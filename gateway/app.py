"""DEV B. Data plane: OpenAI-compatible proxy that executes whatever Arch is live in Mongo.

uvicorn gateway.app:app --port 9000
"""
from __future__ import annotations

import asyncio
import datetime as dt
import threading
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from common import config, db
from common.contracts import Arch
from common.routing import Balancer, estimate_tokens, pick_pool
from gateway import metrics
from infra import deployer

app = FastAPI(title="harness-architect gateway")
current: dict = {"arch": None, "bal": None}
_log_buf: list[dict] = []
_log_lock = threading.Lock()


def _on_arch_change(doc: dict):
    arch = Arch(**{k: v for k, v in doc.items() if k != "_id"})
    eps = deployer.endpoints_for(arch, deployer.live_env_for(arch.version))
    urls = [u for us in eps.values() for u in us]
    print(f"[gateway] v{arch.version} announced, waiting for replicas...", flush=True)
    if not deployer.wait_healthy(urls, timeout=180):
        print(f"[gateway] v{arch.version} never became healthy; keeping old config", flush=True)
        return
    # wait for the reconciler's "loading weights" delay to finish too
    for _ in range(120):
        st = db.db().state.find_one({"_id": "deployed"}) or {}
        if st.get("version", 0) >= arch.version:
            break
        time.sleep(0.5)
    current["bal"], current["arch"] = Balancer(eps), arch   # atomic swap
    print(f"[gateway] now serving v{arch.version}: {arch.summary()}", flush=True)


async def _flush_logs():
    while True:
        await asyncio.sleep(1)
        with _log_lock:
            batch, _log_buf[:] = list(_log_buf), []
        if batch:
            await asyncio.to_thread(db.db().requests.insert_many, batch)


@app.on_event("startup")
async def _startup():
    threading.Thread(target=db.watch_architectures, args=(_on_arch_change,), daemon=True).start()
    asyncio.create_task(_flush_logs())
    app.state.client = httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=1000))


@app.post("/v1/chat/completions")
async def chat(req: Request):
    arch, bal = current["arch"], current["bal"]
    if arch is None:
        return JSONResponse({"error": "no live architecture yet"}, status_code=503)
    body = await req.json()
    tokens = estimate_tokens(body.get("messages", []))
    pool = pick_pool(arch.split_threshold_tokens, tokens)
    url = bal.acquire(pool)
    t0 = time.monotonic()
    ok, status, payload = False, 502, {"error": "upstream failed"}
    try:
        r = await app.state.client.post(f"{url}/v1/chat/completions", json=body)
        status, payload, ok = r.status_code, r.json(), r.status_code == 200
    except Exception as e:  # noqa: BLE001
        payload = {"error": str(e)}
    finally:
        bal.release(url)
    with _log_lock:
        _log_buf.append({"t": dt.datetime.now(dt.timezone.utc), "env": "live", "arch_version": arch.version,
                         "pool": pool, "replica": url, "prompt_tokens": tokens,
                         "latency_ms": round((time.monotonic() - t0) * 1000, 1), "ok": ok})
    return JSONResponse(payload, status_code=status)


@app.get("/metrics/window")
def window(seconds: float = config.WINDOW_S, current_only: bool = True):
    v = current["arch"].version if (current_only and current["arch"]) else None
    return metrics.window(seconds, v)


@app.get("/arch")
def arch():
    a = current["arch"]
    return a.model_dump() if a else {}
