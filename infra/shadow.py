"""DEV A. Shadow testing: deploy candidate setups on separate ports, replay recent traffic, score.

run_many([arch_a, arch_b], profile, seconds) -> [ShadowResult, ...]   (in parallel, one slot each)
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from common import config
from common.contracts import Arch, ShadowResult, TrafficProfile
from common.routing import Balancer, pick_pool
from infra import deployer
from traffic.generator import make_prompt, run_load

MAX_SLOTS = 4


async def _drive(arch: Arch, eps: dict, profile: TrafficProfile, seconds: float):
    bal = Balancer(eps)
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=500)) as client:
        async def send(tokens: int, out: int):
            url = bal.acquire(pick_pool(arch.split_threshold_tokens, tokens))
            t0 = time.monotonic()
            try:
                r = await client.post(f"{url}/v1/chat/completions", json={
                    "model": "dummy", "max_tokens": out,
                    "messages": [{"role": "user", "content": make_prompt(tokens)}]})
                return (time.monotonic() - t0) * 1000 if r.status_code == 200 else None
            except Exception:  # noqa: BLE001
                return None
            finally:
                bal.release(url)
        # anything still queued 3x the SLO after arrivals stop counts as a failure
        return await run_load(profile, seconds, send, drain_s=config.SLO_P95_MS * 3 / 1000)


def run_shadow(arch: Arch, profile: TrafficProfile, seconds: float, slot: int = 0) -> ShadowResult:
    env = f"shadow{slot}"
    try:
        eps = deployer.deploy(arch, env, startup_delay=0)
        res = asyncio.run(_drive(arch, eps, profile, seconds))
    finally:
        deployer.teardown(env)
    lat = sorted(l for _, l in res if l is not None)
    errors = sum(1 for _, l in res if l is None)
    pct = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))] if lat else float("inf")  # noqa: E731
    return ShadowResult(arch_key=arch.key(), n=len(lat), p50_ms=round(pct(0.5)), p95_ms=round(pct(0.95)),
                        errors=errors, usd_hr=arch.usd_hr())


def run_many(archs: list[Arch], profile: TrafficProfile, seconds: float | None = None) -> list[ShadowResult]:
    seconds = seconds or config.SHADOW_S
    archs = archs[:MAX_SLOTS]
    with ThreadPoolExecutor(len(archs)) as ex:
        futs = [ex.submit(run_shadow, a, profile, seconds, i) for i, a in enumerate(archs)]
        return [f.result() for f in futs]
