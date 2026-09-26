# Harness Architect

**An infrastructure engineer for teams that own NVIDIA hardware and need an open-source inference platform.**

Harness turns a workload description and hardware inventory into an evidence-backed serving plan. It selects a compatible open model and stack, renders Kubernetes resources, engine configuration, environment templates, ordered commands, validation findings, and rollback instructions. Imported metrics, Kubernetes events, and logs can then drive performance changes or reliability fixes.

The production workflow generates and validates artifacts; it never runs the commands or changes a cluster. Results distinguish offline validation, simulated evidence, and verification imported from an externally applied bundle. The original serving simulator remains at `/simulation` as a separate evidence-loop demonstration.

> Built entirely during the MongoDB Harness Engineering hackathon (Sep 26, 2026). One of us has built an inference router before; nothing from that repo is used here. The gateway is deliberately simple. All the intelligence is in the agent.

## Architecture (layers)

```
 Production UI          workload + inventory -> design -> artifacts -> operations
 Production workflow    assess -> context -> reason -> render -> validate -> publish -> learn
 Production catalog     pinned models + three compatible OSS recipes
 Evidence pipeline      Prometheus, Kubernetes event, and log normalization + diagnosis
 Artifact renderer      YAML, environment templates, commands, validation, diff, rollback
 Durable memory         Atlas tasks, checkpoints, manifests, bundles, evidence, lessons
 Simulation             original gateway, traffic replay, promotion and rollback demo
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
| `production_tasks`, `production_checkpoints`, `production_contexts` | durable provisioning, optimization, and repair stages | leases, revisions, bounded model context |
| `production_environments`, `production_bundles`, `production_evidence`, `production_lessons`, `production_verifications` | proposed plans, immutable artifacts, operational observations, and outcomes | compare-and-swap publication + evidence lineage |

## Long-horizon workflow

Production planning uses a separate durable workflow:

```text
ASSESS -> CONTEXT -> REASON -> RENDER -> VALIDATE -> PUBLISH -> LEARN -> COMPLETE
```

The three initial recipes cover the six requested projects through compatible combinations:

- Envoy AI Gateway + KServe `LLMInferenceService` + llm-d + vLLM.
- NVIDIA Dynamo + vLLM.
- NVIDIA Dynamo + SGLang.

vLLM and SGLang are alternative engines. The catalog prevents combinations that are not documented together. Model quality and runtime performance remain provisional until their evaluation evidence is imported.

`PUBLISH` means the validated proposal becomes the environment's current **review candidate** in Atlas. It does not apply Kubernetes resources. An operator downloads and applies the bundle outside Harness, then imports matching operational evidence before Harness can label the result runtime-verified.

The original simulator workflow remains available:

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
# open http://127.0.0.1:9100 for production planning
# open http://127.0.0.1:9100/simulation for the simulator
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
