"""Vercel entrypoint. Serves the production planner UI and `/api/production/*`.

The simulator's long-running pieces (agent loop, gateway, traffic generator,
Docker sims) are not part of this deployment: `/simulation` renders whatever the
last local run left in Atlas and does not advance on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.server import app  # noqa: E402  (path setup must run first)

__all__ = ["app"]
