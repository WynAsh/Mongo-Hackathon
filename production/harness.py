"""The long-horizon harness from ``agent/`` and ``memory/``, pointed at production collections.

``CampaignStore`` (leases, CAS transitions, immutable checkpoints, budgets),
``ContextCompiler`` (bounded, ordered memory packets), ``MemoryCurator``
(evidence-linked lessons) and ``ExperimentRunner``/``Evaluator`` are reused
unchanged. ``ProductionDB`` maps their collection names onto ``production_*``
so simulator campaigns and lessons can never enter a production decision;
ingested documentation (``docs``) is shared.
"""
import hashlib
import json
import time

from agent.state import CampaignStore
from common import config, db as db_module
from memory.context import ContextCompiler
from memory.curator import MemoryCurator
from memory.embeddings import OpenRouterEmbedder

SHARED = {"docs"}


class ProductionDB:
    """Mapping-style view whose ``db[name]`` resolves to ``production_<name>``."""

    def __init__(self, database=None):
        self.raw = database if database is not None else db_module.db()

    def __getitem__(self, name):
        return self.raw[name if name in SHARED or name.startswith("production_") else f"production_{name}"]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


class ProductionContextCompiler(ContextCompiler):
    """Same ordering and token budget; regime winners are restricted to one plan lineage."""

    def __init__(self, database, lineage, **kwargs):
        super().__init__(database, embedder=kwargs.pop("embedder", None) or OpenRouterEmbedder(),
                         token_budget=kwargs.pop("token_budget", config.AGENT_CONTEXT_TOKENS), **kwargs)
        self.lineage = lineage

    def _many(self, collection, query, limit=10, sort=None):
        if collection == "regimes":
            query = {**query, "lineage": self.lineage}
        return super()._many(collection, query, limit, sort)


def digest(value, n=16):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:n]


def store(database=None):
    return CampaignStore(ProductionDB(database))


def compile_context(database, *, campaign_id, lineage, query, observation=None, regime_vector=None,
                    scope=None, stage="PROPOSE"):
    """Bounded packet from production memory; the manifest is persisted for audit."""
    compiler = ProductionContextCompiler(ProductionDB(database), lineage)
    return compiler.compile(campaign_id=campaign_id, stage=stage, query=query, observation=observation,
                            regime_vector=regime_vector, scope_filter=scope or {"lineage": lineage})


def manifest_view(packet):
    """What the UI shows: which memories were used, why, and at what token cost."""
    m = packet["manifest"]
    items = []
    for item in m["included"]:
        c = item.get("content") or {}
        items.append({"item_id": item["item_id"], "memory_type": item["memory_type"], "reason": item["reason"],
                      "tokens": item["token_estimate"],
                      "detail": str(c.get("claim") or c.get("summary") or c.get("hypothesis") or c.get("title")
                                    or c.get("arch_key") or c.get("stage") or "")[:240]})
    return {"manifest_id": m["manifest_id"], "tokens": m.get("total_tokens", 0), "budget": m["token_budget"],
            "included": items, "excluded": len(m["excluded"])}


def record_lesson(database, *, record, scope, confirmed, vector=None, arch_key=None, curator=None, clock=time.time):
    """Curate one terminal episode into a scoped lesson; a disagreeing prior lesson is superseded, not deleted."""
    pdb = ProductionDB(database)
    curated = (curator or MemoryCurator()).curate(record)
    lesson = curated["lesson"]
    lesson.update({"scope": scope, "confirmed": int(confirmed), "contradicted": int(not confirmed),
                   "ts": clock(), "created_at": clock()})
    if vector is not None:
        lesson.update({"vector": vector, "bucket": [round(x, 1) for x in vector]})
    if arch_key:
        lesson["arch_key"] = arch_key
    query = {f"scope.{k}": v for k, v in scope.items() if k in {"lineage", "fault", "arch_key", "regime_hash"}}
    prior = pdb.lessons.find_one({**query, "lesson_id": {"$ne": lesson["lesson_id"]}, "superseded_by": None},
                                 sort=[("ts", -1)])
    if prior and bool(prior.get("confirmed")) != bool(confirmed):
        lesson["contradictions"] = lesson["supersedes"] = [prior["lesson_id"]]
        pdb.lessons.update_one({"_id": prior["_id"]}, {"$set": {
            "superseded_by": lesson["lesson_id"], "confidence": float(prior.get("confidence", .5)) * .5}})
    pdb.lessons.update_one({"lesson_id": lesson["lesson_id"]}, {"$setOnInsert": lesson}, upsert=True)
    pdb.summaries.update_one({"summary_id": curated["summary"]["summary_id"]},
                             {"$setOnInsert": curated["summary"]}, upsert=True)
    return {k: v for k, v in pdb.lessons.find_one({"lesson_id": lesson["lesson_id"]}).items() if k != "_id"}


def ensure_indexes(database=None):
    pdb = ProductionDB(database)
    pdb.campaigns.create_index("campaign_id", unique=True, sparse=True)
    pdb.campaign_checkpoints.create_index([("campaign_id", 1), ("revision", 1)], unique=True)
    pdb.lessons.create_index("lesson_id", unique=True)
    pdb.lessons.create_index([("scope.lineage", 1), ("ts", -1)])
    pdb.experiments.create_index("experiment_id", unique=True)
    pdb.experiments.create_index("idempotency_key", unique=True, sparse=True)
    pdb.summaries.create_index("summary_id", unique=True)
    pdb.context_manifests.create_index("manifest_id", unique=True)
    pdb.regimes.create_index([("lineage", 1), ("bucket", 1), ("arch_key", 1)])
    pdb.revisions.create_index([("campaign_id", 1), ("revision", 1)], unique=True)
    pdb.metric_windows.create_index("window_id", unique=True)
    pdb.replay_plans.create_index("replay_id", unique=True)
    pdb.trials.create_index("trial_id", unique=True)
    pdb.evaluations.create_index("evaluation_id", unique=True)
