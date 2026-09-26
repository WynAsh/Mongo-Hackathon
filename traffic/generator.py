"""DEV C. Poisson load generator + scripted scenario.

python -m traffic.generator                  # plays SCENARIO against the gateway
python -m traffic.generator --phase-s 40     # compressed demo timing
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from typing import Awaitable, Callable

import httpx

from common import config, db
from common.contracts import TrafficProfile


def short_tokens(n=200):
    return [random.randint(20, 120) for _ in range(n)]


def mixed_tokens(long_frac=0.3, n=200):
    return [random.randint(1200, 1800) if random.random() < long_frac else random.randint(20, 120)
            for _ in range(n)]


# name, profile. The demo story: quiet -> long-document surge -> quiet -> surge again (memory kicks in)
SCENARIO = [
    ("quiet chats", TrafficProfile(rps=3, prompt_tokens=short_tokens())),
    ("long-document surge", TrafficProfile(rps=6, prompt_tokens=mixed_tokens(0.3))),
    ("quiet chats", TrafficProfile(rps=3, prompt_tokens=short_tokens())),
    ("long-document surge (again)", TrafficProfile(rps=6, prompt_tokens=mixed_tokens(0.3))),
]


def make_prompt(tokens: int) -> str:
    return " ".join(["word"] * tokens)


Sender = Callable[[int, int], Awaitable[float | None]]   # (prompt_tokens, output_tokens) -> latency_ms or None


async def run_load(profile: TrafficProfile, seconds: float, send: Sender,
                   drain_s: float = 120) -> list[tuple[int, float | None]]:
    """Open-loop Poisson arrivals for `seconds`, then wait up to `drain_s` for stragglers
    (unfinished ones count as failures). Returns [(prompt_tokens, latency_ms|None)]."""
    results: list[tuple[int, float | None]] = []
    tasks = []

    async def one(tok):
        results.append((tok, await send(tok, profile.output_tokens)))

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        tasks.append(asyncio.create_task(one(random.choice(profile.prompt_tokens))))
        await asyncio.sleep(random.expovariate(profile.rps))
    _, pending = await asyncio.wait(tasks, timeout=drain_s)
    for t in pending:
        t.cancel()
    results.extend((0, None) for _ in pending)
    return results


def http_sender(client: httpx.AsyncClient, url: str) -> Sender:
    async def send(tokens: int, out: int):
        t0 = time.monotonic()
        try:
            r = await client.post(f"{url}/v1/chat/completions", json={
                "model": "dummy", "max_tokens": out,
                "messages": [{"role": "user", "content": make_prompt(tokens)}]})
            if r.status_code != 200:
                return None
        except Exception:  # noqa: BLE001
            return None
        return (time.monotonic() - t0) * 1000
    return send


async def play(phase_s: float, loops: int):
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=500)) as client:
        send = http_sender(client, config.GATEWAY_URL)
        for _ in range(loops):
            for name, prof in SCENARIO:
                db.db().state.update_one({"_id": "phase"}, {"$set": {"name": name, "ts": time.time(),
                                         "rps": prof.rps}}, upsert=True)
                db.log_event("phase", f"Traffic phase: {name} ({prof.rps} rps)")
                res = await run_load(prof, phase_s, send)
                lat = sorted(l for _, l in res if l is not None)
                if lat:
                    print(f"  {name}: n={len(lat)} p95={lat[int(len(lat)*0.95)-1]:.0f}ms", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-s", type=float, default=60)
    ap.add_argument("--loops", type=int, default=1)
    a = ap.parse_args()
    asyncio.run(play(a.phase_s, a.loops))
