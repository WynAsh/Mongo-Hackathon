# Harness Architect

**An infrastructure engineer for teams that own NVIDIA hardware and need an open-source inference platform.**

Give Harness your hardware and workload; it picks a stack from six open-source projects, sizes the model onto your GPUs, and returns a plan: Kubernetes YAML, engine config, and the ordered shell commands (`install.sh`) to provision and deploy it. It never runs the commands or touches a cluster. The original serving simulator remains at `/simulation`.

> Built entirely during the MongoDB Harness Engineering hackathon (Sep 26, 2026). One of us has built an inference router before; nothing from that repo is used here. The gateway is deliberately simple. All the intelligence is in the agent.

## Architecture (layers)

```
 Plan UI                hardware + workload -> plan, YAML, shell commands
 Planner                size -> render YAML -> validate -> order commands (production/planner.py)
 Catalog                pinned models, the six projects, three recipes (production/catalog.py)
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
| `production_plans` | every generated plan with its YAML and commands, keyed by input hash | download as a zip bundle |

## Hardware to plan

`POST /api/production/plan` with `{inventory, workload, recipe_id?, model_id?}` runs one pure function, `build_plan`:

1. **Size**: pick Qwen3-4B or 8B, find the smallest tensor-parallel split that fits one node, and fill every GPU with replicas. List blockers (context too long, no fit, old Kubernetes, license).
2. **Pick the stack**: unless `recipe_id` is given, the rules size all three stacks and the LLM (`AGENT_MODEL` via OpenRouter) picks one and explains why. The code rejects a pick that has blockers when another stack has none; with no key, an invalid answer, or an error, the rules' default is used. `decided_by` records which happened.
3. **Render**: `k8s/*.yaml` (Gateway + `LLMInferenceService`, or `DynamoGraphDeployment`) with engine flags.
4. **Validate**: parse each file and check it against pinned upstream CRD schemas.
5. **Commands**: preflight -> cluster add-ons -> project installs -> namespace/secret -> dry-run + apply + wait -> smoke test, plus rollback. Emitted as `install.sh` and `PLAN.md`.

The three recipes cover the six projects:

- Envoy AI Gateway + KServe `LLMInferenceService` + llm-d + vLLM.
- NVIDIA Dynamo + vLLM.
- NVIDIA Dynamo + SGLang.

vLLM and SGLang are alternative engines. Sizing is a BF16 screening estimate; load-test before production. Chart and image versions are pinned in `production/catalog.py`.

## Long-horizon workflow

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
