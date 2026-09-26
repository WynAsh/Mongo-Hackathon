"""Run once against the Atlas sandbox: collections, time-series, indexes, vector indexes, baseline arch.

python -m scripts.setup_db            # create everything + seed baseline v1
python -m scripts.setup_db --reset    # wipe collections first (fresh demo run, KEEPS lessons unless --wipe-memory)
"""
from __future__ import annotations

import argparse
import time

from common import db
from common.contracts import BASELINE_ARCH


def main(reset: bool, wipe_memory: bool):
    d = db.db()
    names = set(d.list_collection_names())
    if reset:
        for c in ["architectures", "requests", "events", "state"] + (
                ["regimes", "experiments", "lessons"] if wipe_memory else []):
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
    d.regimes.create_index([("bucket", 1), ("arch_key", 1)])
    d.lessons.create_index([("bucket", 1), ("arch_key", 1)])

    if not db.is_mock():
        from pymongo.operations import SearchIndexModel
        for coll, name in [("regimes", "regime_vec"), ("lessons", "lesson_vec")]:
            if coll not in d.list_collection_names():
                d.create_collection(coll)
            existing = {i["name"] for i in d[coll].list_search_indexes()}
            if name not in existing:
                d[coll].create_search_index(SearchIndexModel(name=name, type="vectorSearch", definition={
                    "fields": [{"type": "vector", "path": "vector", "numDimensions": 4, "similarity": "euclidean"}]}))
                print(f"vector index {coll}.{name} building (takes ~1 min)")

    if not db.live_arch_doc():
        a = BASELINE_ARCH.model_dump()
        a["created_at"] = time.time()
        d.architectures.insert_one(a)
        print(f"seeded baseline v1: {BASELINE_ARCH.summary()}")
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--wipe-memory", action="store_true")
    a = ap.parse_args()
    main(a.reset, a.wipe_memory)
