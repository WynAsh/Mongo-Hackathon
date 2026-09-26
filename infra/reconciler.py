"""DEV A. Keeps live replicas matching the live Arch in Mongo (blue/green, no downtime).

python -m infra.reconciler
"""
from __future__ import annotations

import threading
import time

from common import db
from common.contracts import Arch
from infra import deployer

GRACE_S = 15  # let in-flight requests on the old env finish before tearing it down


def on_change(doc: dict):
    arch = Arch(**{k: v for k, v in doc.items() if k != "_id"})
    env = deployer.live_env_for(arch.version)
    old = deployer.live_env_for(arch.version + 1)
    print(f"[reconciler] v{arch.version} -> {env}: {arch.summary()}", flush=True)
    deployer.deploy(arch, env)
    db.db().state.update_one({"_id": "deployed"}, {"$set": {"version": arch.version, "env": env,
                                                              "ts": time.time()}}, upsert=True)
    print(f"[reconciler] v{arch.version} healthy", flush=True)
    threading.Timer(GRACE_S, deployer.teardown, args=[old]).start()


if __name__ == "__main__":
    db.watch_architectures(on_change)
