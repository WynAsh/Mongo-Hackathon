"""Tiny stand-in for llm-d-inference-sim, for offline dev and laptops without Docker.
Same flags as the real sim (simplified): slots, TTFT, per-token prefill, ITL, slowdown under load.

python -m infra.fakesim --port 8100 --max-num-seqs 8 --ttft-ms 400 --prefill-ms-per-tok 2 --itl-ms 30 --load-factor 3
"""
from __future__ import annotations

import argparse
import asyncio
import time

import uvicorn
from fastapi import FastAPI, Request

args = None
app = FastAPI()
state = {"running": 0, "waiting": 0}
sem: asyncio.Semaphore | None = None


@app.on_event("startup")
async def _startup():
    global sem
    sem = asyncio.Semaphore(args.max_num_seqs)


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "dummy"}]}


@app.get("/metrics")
async def metrics():
    return {"vllm:num_requests_running": state["running"], "vllm:num_requests_waiting": state["waiting"]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))
    out_tokens = int(body.get("max_tokens") or 8)
    state["waiting"] += 1
    async with sem:
        state["waiting"] -= 1
        state["running"] += 1
        try:
            slowdown = 1 + (args.load_factor - 1) * (state["running"] / args.max_num_seqs)
            prefill = (args.ttft_ms + args.prefill_ms_per_tok * prompt_tokens) * slowdown
            decode = args.itl_ms * out_tokens * slowdown
            await asyncio.sleep((prefill + decode) / 1000)
        finally:
            state["running"] -= 1
    return {
        "id": f"fake-{time.time_ns()}", "object": "chat.completion", "model": "dummy",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok " * out_tokens},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": out_tokens},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--ttft-ms", type=float, default=400)
    p.add_argument("--prefill-ms-per-tok", type=float, default=2.0)
    p.add_argument("--itl-ms", type=float, default=30)
    p.add_argument("--load-factor", type=float, default=3.0)
    args = p.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
