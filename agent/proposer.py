"""DEV C. Ask an LLM (OpenRouter) for candidate architectures. Falls back to heuristics offline."""
from __future__ import annotations

import json
import re

import httpx

from common import config
from common.contracts import (GPU_PROFILES, MAX_REPLICAS_PER_POOL, MAX_TOTAL_REPLICAS, Arch, Pool,
                              describe_regime)

SYSTEM = f"""You are an inference infrastructure architect. You redesign how an LLM serving fleet is laid out
so p95 latency stays under the SLO at the lowest $/hr. Every proposal is shadow-tested on replayed traffic
before it goes live, so propose concrete, testable changes.

Knobs you may change (nothing else exists):
- split_threshold_tokens: null (one pool named "shared") or an integer; prompts >= threshold go to pool "long",
  others to pool "short".
- per pool: gpu in {list(GPU_PROFILES)} and replicas 1..{MAX_REPLICAS_PER_POOL}; at most {MAX_TOTAL_REPLICAS} replicas total.

GPU profiles (llm-d simulator flags):
{json.dumps(GPU_PROFILES, indent=1)}

Physics: prefill time grows with prompt length; long prompts hold slots and slow every request on that
replica (load_factor). Mixing long and short prompts on the same replicas causes head-of-line blocking.

Reply with ONLY JSON:
{{"diagnosis": "<one sentence>",
  "candidates": [{{"split_threshold_tokens": <int|null>, "pools": {{"<name>": {{"gpu": "...", "replicas": N}}}},
                  "reason": "<why this should help, one sentence>"}}]}}
Give 2 candidates that are meaningfully different from each other and from the current setup."""


def _llm(user: str) -> dict:
    r = httpx.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                   headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                   json={"model": config.AGENT_MODEL, "temperature": 0.4,
                         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def _heuristic(current: Arch, obs: dict, trigger: str) -> dict:
    long_frac = obs["regime_vector"][3]
    c: list[Arch] = []
    if trigger == "slo_breach":
        if long_frac > 0.05:
            c.append(Arch(split_threshold_tokens=500, pools={"short": Pool(gpu="t4", replicas=2),
                                                              "long": Pool(gpu="a100", replicas=1)},
                          reason="isolate long prefills on a fast-prefill GPU so short chats stop queueing"))
        total = sum(p.replicas for p in current.pools.values())
        c.append(Arch(split_threshold_tokens=None,
                      pools={"shared": Pool(gpu="t4", replicas=min(MAX_REPLICAS_PER_POOL, total + 2))},
                      reason="brute force: add more cheap replicas"))
    else:
        c.append(Arch(split_threshold_tokens=None, pools={"shared": Pool(gpu="t4", replicas=1)},
                      reason="traffic is light; one cheap replica is enough"))
        c.append(Arch(split_threshold_tokens=None, pools={"shared": Pool(gpu="t4", replicas=2)},
                      reason="shrink to a small shared pool"))
    return {"diagnosis": f"(heuristic) {trigger} under {describe_regime(obs['regime_vector'])}",
            "candidates": [a.model_dump() for a in c]}


def propose(current: Arch, obs: dict, trigger: str, memory: dict, past_losers: list[str]) -> tuple[str, list[Arch], list[str]]:
    """Returns (diagnosis, valid candidates, rejected-reasons)."""
    user = json.dumps({
        "trigger": trigger, "slo_p95_ms": config.SLO_P95_MS,
        "current_setup": {"split_threshold_tokens": current.split_threshold_tokens,
                          "pools": {k: v.model_dump() for k, v in current.pools.items()},
                          "usd_hr": current.usd_hr()},
        "observed_last_window": {k: obs[k] for k in ("rps", "p50_ms", "p95_ms", "n", "errors")} |
                                {"traffic": describe_regime(obs["regime_vector"])},
        "lessons_from_memory": [l["text"] for l in memory.get("lessons", [])],
        "already_tried_and_lost_here": past_losers,
    }, indent=1)
    try:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("no OPENROUTER_API_KEY")
        out = _llm(user)
    except Exception as e:  # noqa: BLE001
        print(f"[proposer] LLM unavailable ({e}); heuristic", flush=True)
        out = _heuristic(current, obs, trigger)
    valid, rejected = [], []
    for c in out.get("candidates", []):
        try:
            a = Arch(**{k: c[k] for k in ("split_threshold_tokens", "pools", "reason") if k in c},
                     status="candidate", created_by="agent")
            if a.key() == current.key():
                rejected.append("identical to current")
                continue
            valid.append(a)
        except Exception as e:  # noqa: BLE001
            rejected.append(f"invalid: {str(e).splitlines()[0][:120]}")
    return out.get("diagnosis", ""), valid, rejected
