"""DEV D. Demo UI backend. python -m ui.server  -> http://127.0.0.1:9100"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from common import config, db

app = FastAPI()
from production.api import install
install(app)
HERE = Path(__file__).parent


def _clean(value: Any):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key != "_id"}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value.__class__.__name__ == "ObjectId":
        return str(value)
    return value


def _latest(collection, query=None, sort_field="created_at"):
    """Read the newest optional long-horizon record without changing legacy state."""
    try:
        row = collection.find_one(query or {}, sort=[(sort_field, -1)])
        return _clean(row)
    except Exception:
        # Older/demo databases may not have the additive long-horizon collections.
        return None


def _context_preview(item: dict) -> dict:
    content = item.get("content") or {}
    memory_type = item.get("memory_type", "memory")
    if memory_type == "documentation":
        title = content.get("title") or "Serving documentation"
        detail = content.get("section") or content.get("source") or "Retrieved reference"
    elif memory_type == "observation":
        metrics = content.get("metrics", content)
        title = "Current traffic evidence"
        detail = (f"{metrics.get('rps', '—')} rps · p95 {metrics.get('p95_ms', '—')} ms · "
                  f"{metrics.get('n', 0)} requests")
    elif memory_type == "semantic":
        title = "Learned workload memory"
        detail = content.get("claim") or content.get("text") or content.get("arch_key") or "Similar regime"
    elif memory_type == "episodic":
        title = "Prior experiment"
        detail = content.get("hypothesis") or content.get("summary") or content.get("decision") or "Past evidence"
    elif memory_type == "policy":
        title = "Safety policy"
        detail = f"p95 SLO {content.get('slo_p95_ms', '—')} ms · error ceiling {content.get('max_error_rate', '—')}"
    else:
        title = "Campaign working memory"
        detail = content.get("summary") or content.get("goal") or content.get("stage") or "Durable checkpoint"
    return {
        "item_id": item.get("item_id"), "memory_type": memory_type,
        "reason": item.get("reason"), "tokens": item.get("token_estimate", 0),
        "title": title, "detail": str(detail)[:280],
        "source": content.get("source"),
    }


def _arch_cost(arch: dict | None) -> float | None:
    if not arch:
        return None
    prices = {"t4": .5, "a100": 3.0}
    return round(sum(prices.get(pool.get("gpu"), 0) * pool.get("replicas", 0)
                     for pool in arch.get("pools", {}).values()), 2)


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

    experiment = None
    if campaign and campaign.get("active_experiment_id"):
        experiment = _clean(database.experiments.find_one(
            {"experiment_id": campaign["active_experiment_id"]}))
    experiment = experiment or _latest(database.experiments, {"campaign_id": campaign_id})
    experiment_id = (experiment or {}).get("experiment_id")
    replay = (_clean(database.replay_plans.find_one({"replay_id": experiment.get("replay_id")}))
              if experiment and experiment.get("replay_id") else
              None if experiment else
              _latest(database.replay_plans, {"campaign_id": campaign_id}))
    trial_rows = []
    if experiment_id:
        trial_rows = [_clean(row) for row in database.trials.find(
            {"experiment_id": experiment_id}, {"outcomes": 0}).sort([("repeat", 1), ("execution_order", 1)])]
    trial = trial_rows[-1] if trial_rows else None
    evaluation = (_clean(database.evaluations.find_one({"evaluation_id": experiment.get("evaluation_id")}))
                  if experiment and experiment.get("evaluation_id") else
                  (experiment or {}).get("evaluation"))
    if evaluation is None and experiment is None:
        evaluation = _latest(database.evaluations, {"campaign_id": campaign_id})

    # During an active experiment, present the exact observation that triggered
    # its hypothesis rather than a newer rolling window from a later traffic phase.
    observation = ((campaign or {}).get("checkpoint") or {}).get("observation")
    observation = _clean(observation) or _latest(
        database.metric_windows, {"campaign_id": campaign_id}, "ts")
    lesson = (_clean(database.lessons.find_one({"lesson_id": experiment.get("lesson_id")}))
              if experiment and experiment.get("lesson_id") else
              _latest(database.lessons, {}, "ts"))
    proposal = (experiment or {}).get("proposal") or {}
    candidate = proposal.get("candidate")
    incumbent_cost = _arch_cost((observation or {}).get("architecture"))
    candidate_cost = _arch_cost(candidate)
    potential_savings = (round((1 - candidate_cost / incumbent_cost) * 100, 1)
                         if incumbent_cost and candidate_cost is not None else None)
    doc_sources = sorted(database.docs.distinct("source"))

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
            "items": [_context_preview(item) for item in included],
            "excluded_count": len((context_manifest or {}).get("excluded", [])),
        } if context_manifest else None,
        "replay": {
            "replay_id": (replay or {}).get("replay_id"),
            "content_hash": (replay or {}).get("content_hash"),
        } if replay else None,
        "latest_trial": trial,
        "trials": trial_rows,
        "latest_evaluation": evaluation,
        "experiment": experiment,
        "proposal": proposal or None,
        "observation": observation,
        "lesson": lesson,
        "impact": {"incumbent_cost": incumbent_cost, "candidate_cost": candidate_cost,
                   "potential_savings_pct": potential_savings},
        "knowledge": {"chunks": database.docs.count_documents({}),
                      "sources": doc_sources, "configured_sources": len(config.DOC_SOURCES)},
        "runtime": {"model": config.AGENT_MODEL if config.OPENROUTER_API_KEY else None,
                    "provider": "OpenRouter" if config.OPENROUTER_API_KEY else "Offline fallback",
                    "database": "MongoDB Atlas" if config.MONGO_URI != "mock" else "In-memory demo"},
        "verification_events": workflow_events[:12],
    }


@app.get("/")
@app.get("/production")
def index():
    return FileResponse(HERE / "production.html")


@app.get("/simulation")
def simulation():
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
    uvicorn.run(app, host="127.0.0.1", port=config.UI_PORT, log_level="warning")
