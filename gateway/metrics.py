"""DEV B. Window stats over the `requests` time-series collection. Used by the agent and the UI."""
from __future__ import annotations

import datetime as dt

from common import db
from common.contracts import TrafficProfile, regime_vector


def window(seconds: float, arch_version: int | None = None) -> dict:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)
    q: dict = {"t": {"$gte": since}}
    if arch_version is not None:
        q["arch_version"] = arch_version
    rows = list(db.db().requests.find(q, {"_id": 0, "prompt_tokens": 1, "latency_ms": 1, "ok": 1, "t": 1}))
    rows.sort(key=lambda r: r["t"])
    # stable = first and second half of the window look like the same traffic (not mid-phase-change)
    half = len(rows) // 2
    stable = False
    if half >= 5:
        a = regime_vector(1, [r["prompt_tokens"] for r in rows[:half]])
        b = regime_vector(1, [r["prompt_tokens"] for r in rows[half:]])
        stable = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 < 0.15
    ok = [r for r in rows if r.get("ok")]
    lat = sorted(r["latency_ms"] for r in ok)
    toks = [r["prompt_tokens"] for r in rows]
    if rows:
        ts = [r["t"] for r in rows]
        span = max((max(ts) - min(ts)).total_seconds(), 1.0)
        span = min(max(span, seconds * 0.5), seconds)
    else:
        span = seconds
    rps = len(rows) / span
    pct = lambda q_: round(lat[min(len(lat) - 1, int(len(lat) * q_))]) if lat else None  # noqa: E731
    return {
        "n": len(rows), "errors": len(rows) - len(ok), "rps": round(rps, 2), "stable": stable,
        "p50_ms": pct(0.5), "p95_ms": pct(0.95),
        "regime_vector": regime_vector(rps, toks),
        "prompt_tokens_sample": toks[-300:],
    }


def profile_from_window(w: dict) -> TrafficProfile:
    """Replay recent traffic: same rate, same prompt-length distribution."""
    return TrafficProfile(rps=max(w["rps"], 0.5), prompt_tokens=w["prompt_tokens_sample"] or [50])
