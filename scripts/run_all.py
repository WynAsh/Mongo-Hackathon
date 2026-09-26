"""Everything in one process (needed for MONGO_URI=mock; also handy for the demo laptop).

python -m scripts.run_all                 # reconciler + gateway + agent + UI, then plays the scenario
python -m scripts.run_all --phase-s 60 --no-traffic
"""
from __future__ import annotations

import argparse
import asyncio
import threading
import time

import uvicorn

from common import config, db
from scripts import setup_db


def _serve(app, port):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-s", type=float, default=60)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--no-traffic", action="store_true")
    ap.add_argument("--reset", action="store_true", help="fresh run, keeps memory")
    ap.add_argument("--wipe-memory", action="store_true")
    a = ap.parse_args()

    setup_db.main(a.reset or a.wipe_memory, a.wipe_memory)

    if config.DOC_SOURCES:
        from docs.ingest import DocumentIngestor
        ingestor = DocumentIngestor(db.db().docs)
        chunks = ingestor.ingest(config.DOC_SOURCES)
        print(f"docs: {len(chunks)} chunks from {len(config.DOC_SOURCES) - len(ingestor.errors)} "
              f"of {len(config.DOC_SOURCES)} configured sources", flush=True)
        for failure in ingestor.errors:
            print(f"[docs] skipped {failure['source']}: {failure['error']}", flush=True)

    from agent import loop
    from gateway.app import app as gateway_app
    from infra import reconciler
    from ui.server import app as ui_app

    threading.Thread(target=db.watch_architectures, args=(reconciler.on_change,), daemon=True).start()
    gw_port = int(config.GATEWAY_URL.rsplit(":", 1)[1])
    threading.Thread(target=_serve, args=(gateway_app, gw_port), daemon=True).start()
    threading.Thread(target=_serve, args=(ui_app, config.UI_PORT), daemon=True).start()
    print(f"UI: http://127.0.0.1:{config.UI_PORT}   gateway: {config.GATEWAY_URL}", flush=True)
    time.sleep(3)
    threading.Thread(target=loop.main, daemon=True).start()

    if a.no_traffic:
        while True:
            time.sleep(3600)
    from traffic.generator import play
    asyncio.run(play(a.phase_s, a.loops))
    print("scenario finished; Ctrl-C to exit", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
