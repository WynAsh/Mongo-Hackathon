"""Mongo access. MONGO_URI=mock gives an in-memory DB for offline dev (no change streams / vector search)."""
from __future__ import annotations

import time
import threading
from functools import lru_cache

from . import config

COLLECTIONS = [
    "architectures", "requests", "regimes", "experiments", "lessons", "events", "state",
    "campaigns", "campaign_checkpoints", "policies", "metric_windows", "replay_plans",
    "trials", "evaluations", "summaries", "context_manifests", "docs",
    "production_plans", "production_evidence", "production_changes", "production_verifications",
    "production_campaigns", "production_campaign_checkpoints", "production_campaign_bases", "production_policies",
    "production_lessons", "production_summaries", "production_experiments", "production_candidates",
    "production_regimes", "production_revisions", "production_metric_windows", "production_replay_plans",
    "production_trials", "production_evaluations", "production_context_manifests", "production_events",
]
_PROMOTION_LOCK = threading.Lock()


class StaleArchitectureError(RuntimeError):
    """The live architecture no longer matches the expected version."""


@lru_cache(maxsize=1)
def client():
    if config.MONGO_URI == "mock":
        import mongomock
        return mongomock.MongoClient()
    from pymongo import MongoClient
    return MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=8000)


def db():
    return client()[config.DB_NAME]


def is_mock() -> bool:
    return config.MONGO_URI == "mock"


def live_arch_doc() -> dict | None:
    return db().architectures.find_one({"status": "live"}, sort=[("version", -1)])


def next_version() -> int:
    d = db().architectures.find_one(sort=[("version", -1)])
    return (d["version"] + 1) if d else 1


def promote(arch_dict: dict) -> int:
    """Retire current live, insert new live. Gateway + reconciler pick it up via change stream."""
    arch_dict = dict(arch_dict)
    arch_dict.pop("_id", None)
    arch_dict["version"] = next_version()
    arch_dict["status"] = "live"
    arch_dict["created_at"] = time.time()
    db().architectures.update_many({"status": "live"}, {"$set": {"status": "retired"}})
    db().architectures.insert_one(arch_dict)
    return arch_dict["version"]


def promote_compare_and_swap(arch_dict: dict, expected_version: int, database=None) -> dict:
    """Atomically promote only if the currently-live architecture is unchanged.

    Atlas uses a transaction. Mock mode uses a process lock and a version check,
    which is sufficient for the single-process demo runtime.
    """
    d = database if database is not None else db()
    architectures = d.architectures

    def commit(session=None):
        kwargs = {"session": session} if session is not None else {}
        current = architectures.find_one({"status": "live"}, sort=[("version", -1)], **kwargs)
        actual = current.get("version") if current else None
        if actual != expected_version:
            raise StaleArchitectureError(f"expected live version {expected_version}, found {actual}")
        retired = architectures.update_one(
            {"_id": current["_id"], "status": "live", "version": expected_version},
            {"$set": {"status": "retired"}}, **kwargs,
        )
        if retired.matched_count != 1:
            raise StaleArchitectureError("live architecture changed during promotion")
        promoted = dict(arch_dict)
        promoted.pop("_id", None)
        promoted.update({"version": expected_version + 1, "status": "live", "created_at": time.time()})
        architectures.insert_one(promoted, **kwargs)
        return promoted

    # Explicit mock mode and mongomock don't implement sessions/transactions.
    if is_mock() or type(architectures).__module__.startswith("mongomock"):
        with _PROMOTION_LOCK:
            return commit()
    with d.client.start_session() as session:
        return session.with_transaction(lambda active: commit(active))


def log_event(kind: str, msg: str, **data):
    """Agent decision log -> UI main panel."""
    db().events.insert_one({"ts": time.time(), "kind": kind, "msg": msg, **data})
    print(f"[{kind}] {msg}", flush=True)


def watch_architectures(on_change, poll_s: float = 1.0):
    """Blocking. Calls on_change(live_doc) whenever the live architecture changes.
    Uses a change stream on Atlas; falls back to polling (mock / no replica set)."""
    last = None
    doc = live_arch_doc()
    if doc:
        last = doc["version"]
        on_change(doc)
    if not is_mock():
        try:
            with db().architectures.watch(full_document="updateLookup") as stream:
                for _ in stream:
                    doc = live_arch_doc()
                    if doc and doc["version"] != last:
                        last = doc["version"]
                        on_change(doc)
        except Exception as e:  # noqa: BLE001
            print(f"change stream unavailable ({e}); polling", flush=True)
    while True:
        time.sleep(poll_s)
        doc = live_arch_doc()
        if doc and doc["version"] != last:
            last = doc["version"]
            on_change(doc)
