"""DEV D. Long-term memory: which setups won under which traffic, and plain-English lessons.

record(experiments)            -> updates `experiments`, `regimes` (win/loss per arch per regime), `lessons`
recall(regime_vec)             -> {"winners": [...ranked by Thompson sampling...], "lessons": [...]}

Atlas Vector Search indexes (created by scripts/setup_db.py):
  regimes.regime_vec  on `vector`  (4 dims, euclidean)
  lessons.lesson_vec  on `vector`  (4 dims, euclidean)
"""
from __future__ import annotations

import random
import time

from common import db
from common.contracts import Arch, Experiment, describe_regime

SIMILAR = 0.9   # vectorSearch euclidean score = 1/(1+d^2); 0.9 ~= distance 0.33


def _bucket(vec: list[float]) -> list[float]:
    return [round(x, 1) for x in vec]


def _vsearch(coll: str, index: str, vec: list[float], limit: int = 10) -> list[dict]:
    """Atlas $vectorSearch, with a pure-python fallback for mock mode / index still building."""
    c = db.db()[coll]
    if not db.is_mock():
        try:
            return list(c.aggregate([
                {"$vectorSearch": {"index": index, "path": "vector", "queryVector": vec,
                                   "numCandidates": 100, "limit": limit}},
                {"$addFields": {"sim": {"$meta": "vectorSearchScore"}}},
            ]))
        except Exception as e:  # noqa: BLE001
            print(f"[memory] vectorSearch failed ({e}); python fallback", flush=True)
    docs = list(c.find())
    for d in docs:
        dist2 = sum((a - b) ** 2 for a, b in zip(d["vector"], vec))
        d["sim"] = 1 / (1 + dist2)
    return sorted(docs, key=lambda d: -d["sim"])[:limit]


def record(experiments: list[Experiment], winner_lesson: str | None = None):
    for e in experiments:
        db.db().experiments.insert_one(e.model_dump())
        db.db().regimes.update_one(
            {"bucket": _bucket(e.regime_vector), "arch_key": e.arch.key()},
            {"$set": {"vector": e.regime_vector, "arch": e.arch.model_dump(exclude={"status", "version"}),
                      "ts": time.time()},
             "$inc": {"wins": int(e.won), "losses": int(not e.won)}},
            upsert=True)
        existing = db.db().lessons.find_one({"bucket": _bucket(e.regime_vector), "arch_key": e.arch.key()})
        if e.won:
            text = winner_lesson or (
                f"Under {describe_regime(e.regime_vector)}, '{e.arch.summary()}' beat the incumbent: "
                f"p95 {e.incumbent_p95_ms/1000:.1f}s -> {e.result.p95_ms/1000:.1f}s.")
            db.db().lessons.update_one(
                {"bucket": _bucket(e.regime_vector), "arch_key": e.arch.key()},
                {"$set": {"vector": e.regime_vector, "text": text, "ts": time.time()},
                 "$inc": {"confirmed": 1}, "$setOnInsert": {"contradicted": 0}},
                upsert=True)
        elif existing:
            db.db().lessons.update_one({"_id": existing["_id"]}, {"$inc": {"contradicted": 1}})


def recall(vec: list[float], exclude_key: str | None = None) -> dict:
    """Past winners for similar traffic, ranked by a Thompson-sampling bandit over win/loss counts."""
    hits = [h for h in _vsearch("regimes", "regime_vec", vec) if h["sim"] >= SIMILAR and h.get("wins", 0) > 0]
    by_key: dict[str, dict] = {}
    for h in hits:
        if h["arch_key"] == exclude_key:
            continue
        agg = by_key.setdefault(h["arch_key"], {"arch": h["arch"], "wins": 0, "losses": 0, "sim": h["sim"]})
        agg["wins"] += h.get("wins", 0)
        agg["losses"] += h.get("losses", 0)
    winners = []
    for k, v in by_key.items():
        v["sample"] = random.betavariate(v["wins"] + 1, v["losses"] + 1)   # Thompson sampling
        v["arch_key"] = k
        winners.append(v)
    winners.sort(key=lambda v: -v["sample"])
    lessons = [
        {"text": l["text"], "confirmed": l.get("confirmed", 0), "contradicted": l.get("contradicted", 0),
         "arch_key": l["arch_key"], "sim": round(l["sim"], 3)}
        for l in _vsearch("lessons", "lesson_vec", vec, limit=5) if l["sim"] >= SIMILAR
    ]
    return {"winners": winners, "lessons": lessons}


def arch_from_memory(w: dict) -> Arch:
    d = {k: v for k, v in w["arch"].items() if k not in ("created_at", "_id")}
    d.update(status="candidate", created_by="agent",
             reason=f"recalled from memory: won {w['wins']}x / lost {w['losses']}x in similar traffic")
    return Arch(**d)
