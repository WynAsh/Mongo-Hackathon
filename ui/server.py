"""DEV D. Demo UI backend. python -m ui.server  -> http://127.0.0.1:9100"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from common import config, db

app = FastAPI()
HERE = Path(__file__).parent


def _clean(d):
    if d is None:
        return None
    d = dict(d)
    d.pop("_id", None)
    return d


def _latest(collection, query=None, sort_field="created_at"):
    """Read the newest optional long-horizon record without changing legacy state."""
    try:
        row = collection.find_one(query or {}, sort=[(sort_field, -1)])
        return _clean(row)
    except Exception:
        # Older/demo databases may not have the additive long-horizon collections.
        return None


def _long_horizon_state(database, events):
    campaign_id = getattr(config, "CAMPAIGN_ID", "default")
    campaign = _latest(database.campaigns, {"_id": campaign_id}, "updated_at")
    if campaign is None:
        campaign = _latest(database.campaigns, {"campaign_id": campaign_id}, "updated_at")

    context_manifest = _latest(database.context_manifests, {"campaign_id": campaign_id})
    included = (context_manifest or {}).get("included", [])
    context_ids = [str(item.get("item_id", "")) for item in included if item.get("item_id")]
    context_tokens = (context_manifest or {}).get("total_tokens")
    if context_tokens is None:
        context_tokens = sum(int(item.get("token_estimate", 0)) for item in included)

    replay = _latest(database.replay_plans, {"campaign_id": campaign_id})
    trial = _latest(database.trials, {}, "finished_at")
    evaluation = _latest(database.evaluations, {"campaign_id": campaign_id})
    if evaluation is None:
        experiment = _latest(database.experiments, {"campaign_id": campaign_id})
        evaluation = experiment.get("evaluation") if experiment else None

    workflow_events = []
    for event in events:
        kind = str(event.get("kind", "")).lower()
        message = str(event.get("msg", "")).lower()
        if ("verif" in kind or "rollback" in kind or "verif" in message or "rollback" in message):
            workflow_events.append(event)

    return {
        "campaign": campaign,
        "stage": campaign.get("stage") if campaign else None,
        "revision": campaign.get("revision") if campaign else None,
        "budget_remaining": max(0, int(campaign.get("max_experiments", 0)) -
                                 int(campaign.get("experiments_spent", 0))) if campaign else None,
        "context_manifest": {
            "manifest_id": (context_manifest or {}).get("manifest_id"),
            "included_ids": context_ids,
            "token_count": context_tokens,
            "token_budget": (context_manifest or {}).get("token_budget"),
        } if context_manifest else None,
        "replay": {
            "replay_id": (replay or {}).get("replay_id"),
            "content_hash": (replay or {}).get("content_hash"),
        } if replay else None,
        "latest_trial": trial,
        "latest_evaluation": evaluation,
        "verification_events": workflow_events[:12],
    }


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/state")
def state():
    d = db.db()
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=240)
    rows = list(d.requests.find({"t": {"$gte": since}}, {"_id": 0, "t": 1, "latency_ms": 1}))
    buckets: dict[int, list[float]] = {}
    for r in rows:
        t = r["t"] if r["t"].tzinfo else r["t"].replace(tzinfo=dt.timezone.utc)
        buckets.setdefault(int(t.timestamp()) // 5 * 5, []).append(r["latency_ms"])
    series = [{"t": k, "p95": sorted(v)[int(len(v) * 0.95) - 1 if len(v) > 1 else 0]}
              for k, v in sorted(buckets.items())]
    history = [_clean(a) for a in d.architectures.find({}, {"_id": 0}).sort("version", -1).limit(8)]
    events = [_clean(e) for e in d.events.find().sort("ts", -1).limit(60)]
    return {
        "slo_ms": config.SLO_P95_MS,
        "live": _clean(db.live_arch_doc()),
        "history": history,
        "phase": _clean(d.state.find_one({"_id": "phase"})),
        "events": events,
        "lessons": [_clean(l) for l in d.lessons.find({}, {"vector": 0, "bucket": 0}).sort("ts", -1).limit(8)],
        "experiments": d.experiments.count_documents({}),
        "series": series,
        "long_horizon": _long_horizon_state(d, events),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.UI_PORT, log_level="warning")
