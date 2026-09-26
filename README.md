# Harness Architect

**Teams running open-source models control every part of their serving stack but don't have time to tune it. Our agent does it for them.**

An agent that watches live LLM-serving traffic, proposes changes to the *whole serving architecture* (pool layout, GPU types, replica counts), **shadow-tests every change on replayed traffic** before touching production, promotes only proven winners, and **remembers which setups won under which traffic** in MongoDB Atlas, so the next time that traffic returns it fixes the problem faster.

> Built entirely during the MongoDB Harness Engineering hackathon (Sep 26, 2026). One of us has built an inference router before; nothing from that repo is used here. The gateway is deliberately simple. All the intelligence is in the agent.

## Architecture (layers)

```
 8  UI (ui/)            decision log + live setup diagram
 7  Agent (agent/)      observe -> recall -> propose (LLM) -> shadow test -> promote -> learn
 6  Memory (memory/)    regimes + lessons (Atlas Vector Search), Thompson-sampling bandit
 5  Shadow (infra/shadow.py)   candidate setups on separate ports, replayed traffic, scored
 4  Control plane       live Arch doc in Mongo -> change stream -> gateway + reconciler hot-swap (blue/green)
 3  Gateway (gateway/)  OpenAI-compatible proxy: pool by prompt length, least-in-flight; logs to time-series
 2  Traffic (traffic/)  Poisson load: quiet -> long-document surge -> quiet -> surge again
 1  Serving (infra/)    llm-d-inference-sim containers (or infra/fakesim.py), GPU profiles t4 / a100
```

## MongoDB Atlas is the harness's state

| Collection | Holds | Feature |
|---|---|---|
| `architectures` | versioned serving setups, `status: live/candidate/retired` | **change streams** drive hot reload |
| `requests` | every request: latency, pool, arch version | **time-series collection** |
| `regimes` | traffic fingerprint -> arch, wins/losses | **Vector Search** (4-dim, euclidean) + bandit |
| `lessons` | plain-English lessons, confirmed/contradicted | **Vector Search** |
| `experiments` | every shadow test result | aggregation |
| `events` | agent decision log (UI) | |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # set MONGO_URI (Atlas sandbox) and OPENROUTER_API_KEY

# offline, no Docker, no Atlas: in-memory Mongo + fake sims + heuristic proposer
MONGO_URI=mock SIM_MODE=fake python -m scripts.run_all --phase-s 60
# open http://127.0.0.1:9100
```

Against Atlas with real llm-d sims:

```bash
docker pull ghcr.io/llm-d/llm-d-inference-sim:v0.8.0
python -m scripts.setup_db                      # collections, time-series, vector indexes, baseline v1
SIM_MODE=docker python -m scripts.run_all --phase-s 60
# fresh demo but keep memory:   python -m scripts.run_all --reset
# fresh demo, empty memory:     python -m scripts.run_all --wipe-memory
```

With Atlas you can also run each piece as its own process (each dev runs their own):

```bash
python -m infra.reconciler
uvicorn gateway.app:app --port 9000
python -m agent.loop
python -m ui.server
python -m traffic.generator --phase-s 60
```

## Measured on fake sims (20s shadow tests)

| Traffic | Setup | p95 | $/hr |
|---|---|---|---|
| surge (6 rps, 30% long) | baseline 2×T4 shared | 20.9s | 1.0 |
| surge | **split: 2×T4 short + 1×A100 long** | **2.2s** | 4.0 |
| surge | brute force 4×T4 shared | 9.7s | 2.0 |
| quiet (3 rps, short) | 2×T4 shared | 1.8s | 1.0 |
| quiet | 1×T4 | 4.3s | 0.5 |

Adding replicas doesn't fix a long-prompt surge; isolating long prefills does. The agent has to discover that, and then remember it.

## Team split

| Dev | Owns | Files |
|---|---|---|
| A: Infra | serving, deploy, shadow tests | `infra/` |
| B: Gateway | data plane, hot reload, metrics | `gateway/`, `common/routing.py` |
| C: Agent | traffic, proposer, loop | `traffic/`, `agent/` |
| D: Memory + demo | memory, UI, video | `memory/`, `ui/`, `scripts/` |

`common/contracts.py` is the shared contract. Change it only as a team.
