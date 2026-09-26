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
    return {
        "slo_ms": config.SLO_P95_MS,
        "live": _clean(db.live_arch_doc()),
        "history": history,
        "phase": _clean(d.state.find_one({"_id": "phase"})),
        "events": [_clean(e) for e in d.events.find().sort("ts", -1).limit(60)],
        "lessons": [_clean(l) for l in d.lessons.find({}, {"vector": 0, "bucket": 0}).sort("ts", -1).limit(8)],
        "experiments": d.experiments.count_documents({}),
        "series": series,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.UI_PORT, log_level="warning")
