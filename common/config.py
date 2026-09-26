"""All settings come from environment variables (see .env.example)."""
import os

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

MONGO_URI = os.getenv("MONGO_URI", "mock")          # "mock" = in-memory mongomock (offline dev only)
DB_NAME = os.getenv("DB_NAME", "harness_architect")

# "docker" = real llm-d-inference-sim containers, "fake" = local python fakesim processes
SIM_MODE = os.getenv("SIM_MODE", "fake")
SIM_IMAGE = os.getenv("SIM_IMAGE", "ghcr.io/llm-d/llm-d-inference-sim:v0.8.0")
SIM_HOST = os.getenv("SIM_HOST", "127.0.0.1")

LIVE_PORT_BASE = int(os.getenv("LIVE_PORT_BASE", "8100"))
SHADOW_PORT_BASE = int(os.getenv("SHADOW_PORT_BASE", "8200"))   # shadow slot i uses base + i*20
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9000")
UI_PORT = int(os.getenv("UI_PORT", "9100"))

# Artificial delay before a new replica is usable (GPUs load weights slowly)
REPLICA_STARTUP_S = float(os.getenv("REPLICA_STARTUP_S", "2"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "anthropic/claude-sonnet-4.5")

SLO_P95_MS = float(os.getenv("SLO_P95_MS", "4000"))
WINDOW_S = int(os.getenv("WINDOW_S", "20"))            # observation window
SHADOW_S = int(os.getenv("SHADOW_S", "20"))            # shadow test duration
CYCLE_S = int(os.getenv("CYCLE_S", "10"))              # agent loop period
PROMOTE_MARGIN = float(os.getenv("PROMOTE_MARGIN", "0.15"))  # candidate must be 15% better
