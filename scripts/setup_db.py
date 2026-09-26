"""Run once against the Atlas sandbox: collections, time-series, indexes, vector indexes, baseline arch.

python -m scripts.setup_db            # create everything + seed baseline v1
python -m scripts.setup_db --reset    # wipe collections first (fresh demo run, KEEPS lessons unless --wipe-memory)
"""
from __future__ import annotations

import argparse
import time

from common import db
from common import config
from common.contracts import BASELINE_ARCH, OptimizationPolicy
from agent.state import CampaignStore, ensure_indexes as ensure_campaign_indexes


def main(reset: bool, wipe_memory: bool):
    d = db.db()
    names = set(d.list_collection_names())
    if reset:
        operational = [
            "architectures", "requests", "events", "state", "campaigns",
            "campaign_checkpoints", "replay_plans", "trials", "evaluations", "metric_windows",
            "context_manifests",
        ]
        learned = ["regimes", "experiments", "lessons", "summaries", "docs", "policies"]
        operational += ["production_plans"]
        for c in operational + (learned if wipe_memory else []):
            if c in names:
                d[c].drop()
        names = set(d.list_collection_names())

    if "requests" not in names:
        try:
            d.create_collection("requests", timeseries={"timeField": "t", "metaField": "arch_version",
                                                        "granularity": "seconds"},
                                expireAfterSeconds=6 * 3600)
            print("created time-series collection: requests")
        except Exception as e:  # noqa: BLE001  (mongomock)
            print(f"time-series not available ({e}); plain collection")
    d.architectures.create_index([("status", 1), ("version", -1)])
    d.events.create_index([("ts", -1)])
    d.experiments.create_index([("ts", -1)])
    d.experiments.create_index("experiment_id", unique=True, sparse=True)
    d.experiments.create_index("idempotency_key", unique=True, sparse=True)
    d.experiments.create_index([("campaign_id", 1), ("created_at", -1)])
    d.regimes.create_index([("bucket", 1), ("arch_key", 1)])
    d.lessons.create_index([("bucket", 1), ("arch_key", 1)])
    d.lessons.create_index("lesson_id", unique=True, sparse=True)
    d.lessons.create_index([("scope.regime_hash", 1), ("created_at", -1)])
    d.replay_plans.create_index("replay_id", unique=True)
    d.trials.create_index("trial_id", unique=True)
    d.trials.create_index([("experiment_id", 1), ("repeat", 1), ("role", 1)])
    d.evaluations.create_index("evaluation_id", unique=True)
    d.evaluations.create_index([("campaign_id", 1), ("created_at", -1)])
    d.metric_windows.create_index([("campaign_id", 1), ("ts", -1)])
    d.summaries.create_index("summary_id", unique=True)
    d.summaries.create_index([("level", 1), ("scope.regime_hash", 1), ("version", -1)])
    d.context_manifests.create_index("manifest_id", unique=True)
    d.context_manifests.create_index([("campaign_id", 1), ("created_at", -1)])
    d.docs.create_index("content_hash", unique=True)
    d.docs.create_index([("source", 1), ("chunk_index", 1)])
    d.production_plans.create_index("plan_id", unique=True)
    ensure_campaign_indexes(d)

    if not db.is_mock():
        from pymongo.errors import OperationFailure
        from pymongo.operations import SearchIndexModel
        for coll, name in [("regimes", "regime_vec"), ("lessons", "lesson_vec")]:
            if coll not in d.list_collection_names():
                d.create_collection(coll)
            existing = {i["name"] for i in d[coll].list_search_indexes()}
            if name not in existing:
                try:
                    d[coll].create_search_index(SearchIndexModel(name=name, type="vectorSearch", definition={
                        "fields": [{"type": "vector", "path": "vector", "numDimensions": 4, "similarity": "euclidean"}]}))
                    print(f"vector index {coll}.{name} building (takes ~1 min)")
                except OperationFailure as exc:
                    print(f"vector index {coll}.{name} skipped: {exc.details.get('errmsg', exc)}")
        for coll, name in [("lessons", "lessons_text"), ("experiments", "experiments_text"),
                           ("docs", "docs_text")]:
            if coll not in d.list_collection_names():
                d.create_collection(coll)
            existing = {i["name"] for i in d[coll].list_search_indexes()}
            if name not in existing:
                try:
                    d[coll].create_search_index(SearchIndexModel(name=name, type="vectorSearch", definition={
                        "fields": [
                            {"type": "vector", "path": "embedding", "numDimensions": 1536,
                             "similarity": "cosine"},
                            {"type": "filter", "path": "scope.model"},
                            {"type": "filter", "path": "scope.gpu"},
                            {"type": "filter", "path": "scope.serving_engine"},
                            {"type": "filter", "path": "scope.regime_hash"},
                        ]}))
                    print(f"vector index {coll}.{name} building (takes ~1 min)")
                except OperationFailure as exc:
                    print(f"vector index {coll}.{name} skipped: {exc.details.get('errmsg', exc)}")

    if not db.live_arch_doc():
        a = BASELINE_ARCH.model_dump()
        a["created_at"] = time.time()
        d.architectures.insert_one(a)
        print(f"seeded baseline v1: {BASELINE_ARCH.summary()}")
    policy = OptimizationPolicy(
        policy_id=config.CAMPAIGN_ID,
        slo_p95_ms=config.SLO_P95_MS,
        min_trial_requests=config.MIN_TRIAL_REQUESTS,
        trial_repeats=config.TRIAL_REPEATS,
        post_promotion_verify_s=config.POST_PROMOTION_VERIFY_S,
    ).model_dump()
    d.policies.update_one(
        {"_id": config.CAMPAIGN_ID},
        {"$setOnInsert": {**policy, "campaign_id": config.CAMPAIGN_ID, "scope": "global"}},
        upsert=True,
    )
    live = db.live_arch_doc()
    CampaignStore(d).bootstrap(
        config.CAMPAIGN_ID,
        max_experiments=config.CAMPAIGN_MAX_EXPERIMENTS,
        policy=policy,
        initial_state={
            "summary": "Campaign initialized",
            "incumbent_version": live.get("version") if live else None,
        },
    )
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--wipe-memory", action="store_true")
    a = ap.parse_args()
    main(a.reset, a.wipe_memory)
