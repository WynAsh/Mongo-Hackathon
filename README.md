# Harness Architect

**Teams running open-source models control every part of their serving stack but don't have time to tune it. Our agent does it for them.**

An agent that watches live LLM-serving traffic, proposes changes to the *whole serving architecture* (pool layout, GPU types, replica counts), **shadow-tests every change on identical persisted traffic** before touching production, promotes only proven winners, and **remembers which setups won under which traffic** in MongoDB Atlas. Campaigns survive restarts without replaying an ever-growing chat transcript.

The product goal is continuous, evidence-based optimization: find the lowest GPU cost that still satisfies latency and error SLOs as traffic changes. Live traffic is used to detect and characterize an opportunity; an immutable copy of that traffic is used for safe shadow proof; live traffic then verifies a promoted change and triggers automatic rollback if it regresses.

> Built entirely during the MongoDB Harness Engineering hackathon (Sep 26, 2026). One of us has built an inference router before; nothing from that repo is used here. The gateway is deliberately simple. All the intelligence is in the agent.

## Architecture (layers)

```
 8  UI (ui/)            decision log + live setup diagram
 7  Agent (agent/)      durable controller + bounded-context Strands architect
 6  Memory (memory/)    policy -> checkpoint -> observations -> bandit -> lessons -> evidence
 5  Shadow (infra/shadow.py)   immutable replay, paired serial trials, deterministic gates
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
| `campaigns`, `campaign_checkpoints` | stage, revision, lease, budget, compact working memory | atomic compare-and-swap |
| `replay_plans`, `trials`, `evaluations` | immutable traffic and promotion evidence | resumable/idempotent execution |
| `summaries`, `context_manifests` | evidence-linked compaction and exact model inputs | bounded context |
| `docs` | heading-aware documentation chunks and provenance | semantic + lexical retrieval |

## Long-horizon workflow

```text
OBSERVE -> BUILD_CONTEXT -> PROPOSE -> VALIDATE -> PREPARE_REPLAY
        -> RUN_TRIALS -> EVALUATE -> PROMOTE/REJECT -> VERIFY_LIVE
        -> LEARN -> CHECKPOINT -> OBSERVE
```

The controller, not the model, owns transitions. Every transition increments a MongoDB revision and writes a checkpoint. A renewable lease permits one campaign writer, deterministic IDs reuse completed work after a crash, and promotion checks the expected live architecture version. Post-promotion telemetry can restore the previous architecture as a new version.

Each Strands invocation is stateless. The context compiler fits policy, the campaign checkpoint, the current observation, similar-regime winners, scoped lessons, supporting experiments, and documentation into `AGENT_CONTEXT_TOKENS`. A persisted context manifest records what was included or excluded and why. Raw requests expire after six hours; durable windows, trials, episodes, lessons, regime summaries, and campaign checkpoints form the compaction ladder.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -r requirements-dev.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env        # set OPENROUTER_API_KEY and other local options

# offline, no Docker, no Atlas: in-memory Mongo + fake sims + heuristic proposer
MONGO_URI=mock SIM_MODE=fake python -m scripts.run_all --phase-s 60
# open http://127.0.0.1:9100
```

Leave `OPENROUTER_API_KEY` empty for the deterministic heuristic proposer and lexical retrieval. With a key, Strands uses OpenRouter's OpenAI-compatible endpoint for typed proposals and the same provider for 1,536-dimensional embeddings. Optional docs can be ingested with:

```bash
python -m scripts.ingest_docs
```

`DOC_SOURCES` accepts comma-separated local Markdown/text paths or curated HTTP(S) URLs.
If it is blank, Harness uses the curated vLLM engine/scaling references and llm-d routing, baseline, P/D-disaggregation, and autoscaling guides shown in `.env.example`. `scripts.run_all` ingests them automatically and keeps unchanged embeddings.

For Atlas, either put `MONGODB_URI=...` in a local `atlas-credentials.env` file (already git-ignored), or set `MONGO_URI=...` in `.env`. `MONGODB_URI` takes precedence. Harness creates its collections and indexes additively; on Atlas tiers with a search-index limit, retrieval falls back to deterministic lexical ranking for any index the tier cannot create.

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
