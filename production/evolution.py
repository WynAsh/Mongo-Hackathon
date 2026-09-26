"""Long-horizon architecture evolution for one provisioned plan.

A campaign walks a traffic timeline derived from the plan's own workload and
capacity. Each phase advances through the same persisted stages as the
simulator campaign (``agent/workflow.py``): OBSERVE -> BUILD_CONTEXT ->
PROPOSE -> VALIDATE -> PREPARE_REPLAY -> RUN_TRIALS -> EVALUATE ->
PROMOTE/REJECT -> VERIFY_LIVE -> LEARN -> CHECKPOINT, one stage per ``tick``.
Every stage is a CAS transition with an immutable checkpoint, so a restarted
process resumes the same campaign without duplicating experiments.

Candidates come from a menu of architecture moves (scale, batch, tensor
parallelism, model swap, prefill/decode disaggregation, stack migration), each
re-sized for the plan's hardware and rendered/validated by the planner. The
architect picks one using memory recalled from this plan's lineage. Trials
replay the same persisted traffic on simulator backends (``production.sim``)
and the existing ``Evaluator`` decides. Promotion only records a new revision
of the plan; nothing is deployed.
"""
import copy
import json
import math
import random
import socket
import time
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.evaluator import Evaluator
from agent.experiments import ExperimentRunner, make_replay_plan
from common import config
from common.contracts import OptimizationPolicy, ReplayPlan, TrafficProfile, TrialResult
from production.catalog import MODELS, RECIPES
from production.harness import ProductionDB, compile_context, digest, manifest_view, record_lesson, store
from production.operations import render_change, unified_diff
from production.planner import RUNTIME_RESERVE_GIB, WEIGHT_OVERHEAD, size
from production.sim import ProdArch, SimExecutor, gpu_spec, profile

MAX_ATTEMPTS_PER_PHASE = 2
PHASE_SECONDS = 30
CTX_SCALE = 32768


def _plain(row):
    return None if row is None else {k: v for k, v in row.items() if k != "_id"}


# ---------------------------------------------------------------- traffic timeline

def _capacity_rps(arch, inventory, prompt, out):
    """Rough sustainable rps of an architecture for a mean request (sizing the scenario, not a result)."""
    prof = profile(arch, inventory)
    service_s = (prof["ttft_ms"] + prof["prefill_ms_per_tok"] * prompt + prof["itl_ms"] * out) * \
        (1 + (prof["load_factor"] - 1) * 0.8) / 1000
    return arch.workers() * arch.max_num_seqs / max(service_s, 1e-3)


def default_timeline(base_row):
    """Six phases scaled to this plan's context length and measured capacity."""
    wl, inv = base_row["input"]["workload"], base_row["input"]["inventory"]
    arch = ProdArch.from_plan(base_row["plan"])
    ctx = wl["context_tokens"]
    out = min(wl["output_tokens"], 1024)
    short = [max(16, int(ctx * 0.02)), max(32, int(ctx * 0.15))]
    long_ = [int(ctx * 0.6), int(ctx * 0.95)]
    specs = [
        ("Launch", "Steady chat traffic at under half of capacity.", 0.45, short, long_, 0.05, 0.2),
        ("Adoption growth", "Usage grows past what the launch topology can absorb.", 1.0, short, long_, 0.05, 0.2),
        ("Long-document RAG", "Half the requests now carry long retrieved documents.", 0.95, short, long_, 0.5, 0.1),
        ("Agentic assistants", "Long multi-turn sessions that re-send most of their history.", 1.0,
         [int(ctx * 0.4), int(ctx * 0.9)], long_, 0.0, 0.8),
        ("Overnight lull", "Traffic drops to a trickle.", 0.12, short, long_, 0.05, 0.2),
        ("Growth returns", "The adoption-growth regime comes back.", 1.0, short, long_, 0.05, 0.2),
    ]
    phases = []
    for name, story, factor, s_range, l_range, long_share, reuse in specs:
        mean_prompt = (sum(s_range) / 2) * (1 - long_share) + (sum(l_range) / 2) * long_share
        effective = mean_prompt * (1 - 0.75 * reuse)
        rps = round(factor * _capacity_rps(arch, inv, effective, out), 2)
        phases.append({"name": name, "story": story, "rps": max(0.5, rps), "short_tokens": s_range,
                       "long_tokens": l_range, "long_share": long_share, "prefix_reuse": reuse,
                       "output_tokens": out, "duration_s": max(PHASE_SECONDS, math.ceil(45 / max(rps, 0.5)))})
    return phases


def suggested_slo_ms(base_row):
    """About twice this plan's unloaded latency for a typical launch request, rounded up to 500 ms."""
    wl, inv = base_row["input"]["workload"], base_row["input"]["inventory"]
    prof = profile(ProdArch.from_plan(base_row["plan"]), inv)
    unloaded = prof["ttft_ms"] + prof["prefill_ms_per_tok"] * wl["context_tokens"] * 0.1 + \
        prof["itl_ms"] * min(wl["output_tokens"], 1024) * prof["load_factor"]
    return int(math.ceil(unloaded * 2 / 500) * 500)


def traffic_profile(phase, seed):
    rng = random.Random(seed)
    prompts = [rng.randint(*phase["long_tokens"]) if rng.random() < phase["long_share"]
               else rng.randint(*phase["short_tokens"]) for _ in range(200)]
    return TrafficProfile(rps=phase["rps"], prompt_tokens=prompts, output_tokens=phase["output_tokens"])


def regime(phase, prompts):
    s = sorted(prompts)
    p50, p90 = s[len(s) // 2], s[int(len(s) * 0.9)]
    vec = [round(min(phase["rps"] / 100, 1), 3), round(math.log1p(p50) / math.log1p(CTX_SCALE), 3),
           round(math.log1p(p90) / math.log1p(CTX_SCALE), 3),
           round(sum(t >= 4096 for t in s) / len(s), 3), round(phase["prefix_reuse"], 3)]
    return vec, digest([round(x, 1) for x in vec], 12)


# ---------------------------------------------------------------- candidates

def _model(model_id):
    return next(m for m in MODELS.values() if m["model_id"] == model_id)


def plan_for(base_row, arch: ProdArch, reason):
    """Re-size ``arch`` on the plan's hardware and render it; returns (plan, files, checks, blockers)."""
    inv, wl = base_row["input"]["inventory"], base_row["input"]["workload"]
    p = size(inv, wl, arch.recipe_id, arch.model_id)
    blockers = [b for b in p["blockers"] if "No node fits" not in b and "exceeds" not in b]
    model = _model(arch.model_id)
    if arch.max_model_len > model["max_position_embeddings"]:
        blockers.append(f"max_model_len {arch.max_model_len} exceeds {arch.model_id}'s limit.")
    if arch.disaggregated and not arch.recipe_id.startswith("dynamo"):
        blockers.append("Prefill/decode disaggregation is rendered for the Dynamo recipes only.")
    need = (model["parameters_b"] * 2 * WEIGHT_OVERHEAD + arch.max_num_seqs * arch.max_model_len *
            model["kv_bytes_per_token"] / 1024**3 + RUNTIME_RESERVE_GIB)
    tp = arch.tensor_parallel
    placement, remaining = {}, arch.workers()
    for node in inv["nodes"]:
        link_ok = tp == 1 or str(node.get("interconnect", "")).lower() not in {"", "none", "unknown"}
        if node.get("gpu_count", 0) >= tp and node["gpu_count"] % tp == 0 and link_ok and need / tp <= node.get("vram_gb", 0):
            take = min(remaining, node["gpu_count"] // tp)
            if take:
                placement[node["name"]] = take
                remaining -= take
    if remaining > 0:
        blockers.append(f"{arch.workers()} workers × TP{tp} ({need:.1f} GiB each) do not fit the inventory; "
                        f"{remaining} left unplaced.")
    p.update({"engine": {"max_model_len": arch.max_model_len, "max_num_seqs": arch.max_num_seqs},
              "allocation": {"tensor_parallel": tp, "replicas": arch.workers(), "gpus_per_replica": tp,
                             "placement": placement, "estimated_gib_per_replica": round(need, 1)},
              "blockers": blockers, "decided_by": "evolution campaign", "recipe_reason": reason,
              "traits": [], "tradeoffs": []})
    p["notes"] = [f"Estimated {need:.1f} GiB per worker; simulator-evaluated, load-test before production."]
    if arch.disaggregated:
        p["disaggregation"] = {"prefill_replicas": arch.prefill_replicas, "decode_replicas": arch.decode_replicas}
        p["notes"].append("Prefill/decode worker flags follow the Dynamo disaggregation guide; verify for the pinned release.")
    base_p = base_row["plan"]
    if arch.recipe_id == base_p["recipe_id"]:
        for key in ("probes", "gateway_policy"):
            if base_p.get(key):
                p[key] = copy.deepcopy(base_p[key])
    else:
        p["notes"].append(f"Stack migration from {base_p['stack']}: requires operator review and a cut-over plan.")
    files, checks, _ = render_change(p)
    if not all(c["ok"] for c in checks):
        blockers.append("Rendered YAML failed schema validation.")
    return p, files, checks, blockers


def moves(incumbent: ProdArch, obs, trigger, base_row):
    """Architecture moves worth testing for this trigger, most promising first (deterministic)."""
    inv = base_row["input"]["inventory"]
    cap = sum(n.get("gpu_count", 0) for n in inv["nodes"]) // incumbent.tensor_parallel
    free = cap - incumbent.workers()
    out, seen = [], {incumbent.key()}

    def add(name, arch, why, migration=False):
        if arch.key() not in seen:
            seen.add(arch.key())
            out.append({"move": name, "arch": arch, "why": why, "migration": migration})

    a = incumbent
    long_heavy, reuse = obs["long_frac"] >= 0.3, obs["prefix_reuse"] >= 0.5
    if trigger == "slo_breach":
        if reuse and a.recipe_id != "dynamo-sglang":
            add("migrate", a.model_copy(update={"recipe_id": "dynamo-sglang", "replicas": a.workers(),
                                                "prefill_replicas": 0, "decode_replicas": 0}),
                f"{obs['prefix_reuse']:.0%} of each prompt is reused history; SGLang's radix cache serves more of it.",
                migration=True)
        if long_heavy and a.workers() >= 2 and not a.disaggregated:
            p = max(1, round(a.workers() * 0.4))
            target = a.recipe_id if a.recipe_id.startswith("dynamo") else "dynamo-vllm"
            add("disaggregate", a.model_copy(update={"recipe_id": target, "replicas": 0, "prefill_replicas": p,
                                                     "decode_replicas": a.workers() - p}),
                f"{obs['long_frac']:.0%} long prompts: dedicated prefill workers stop long prefills stalling decode.",
                migration=target != a.recipe_id)
        if long_heavy and a.tensor_parallel < 4 and cap // (a.tensor_parallel * 2) >= 1:
            workers = max(1, min(a.workers(), cap // (a.tensor_parallel * 2)))
            add("tensor_parallel", a.model_copy(update={"tensor_parallel": a.tensor_parallel * 2, "replicas": workers,
                                                        "prefill_replicas": 0, "decode_replicas": 0}),
                "Long prompts: doubling tensor parallelism roughly halves prefill time per request.")
        if free > 0:
            grow = {"decode_replicas": a.decode_replicas + free} if a.disaggregated else {"replicas": a.replicas + free}
            add("scale_out", a.model_copy(update=grow), f"{free} GPU(s) are idle in the inventory.")
        add("raise_batch", a.model_copy(update={"max_num_seqs": a.max_num_seqs * 2}),
            "Queueing with memory headroom: double the concurrent sequences per worker.")
        if "8B" in a.model_id:
            add("model_swap", a.model_copy(update={"model_id": "Qwen/Qwen3-4B"}),
                "A smaller model halves per-token cost; confirm quality is acceptable.")
    elif trigger == "overprovisioned":
        if a.disaggregated and not long_heavy:
            add("aggregate", a.model_copy(update={"replicas": a.workers(), "prefill_replicas": 0, "decode_replicas": 0}),
                "Few long prompts remain; aggregated workers are simpler at this load.")
        drop = max(1, a.workers() // 4)
        if a.workers() - drop >= 1:
            shrink = ({"decode_replicas": max(1, a.decode_replicas - drop)} if a.disaggregated
                      else {"replicas": a.replicas - drop})
            add("scale_in", a.model_copy(update=shrink), f"p95 is far under the SLO; release {drop * a.tensor_parallel} GPU(s).")
        if a.tensor_parallel > 1:
            add("tensor_parallel", a.model_copy(update={"tensor_parallel": a.tensor_parallel // 2}),
                "Short prompts no longer need wide tensor parallelism; halve the GPUs per worker.")
    return out


def recalled(packet, base_row):
    """Past winning architectures for a similar regime in this lineage, from the compiled context."""
    wins = []
    for item in packet["context"]:
        c = item.get("content") or {}
        if item["reason"] == "similar_traffic_regime" and c.get("arch"):
            try:
                wins.append((c.get("wins", 0), c.get("losses", 0), item["item_id"], ProdArch(**c["arch"])))
            except Exception:  # noqa: BLE001
                continue
    return wins


# ---------------------------------------------------------------- architect

class ArchitectChoice(BaseModel):
    option: int = Field(ge=0)
    hypothesis: str = Field(max_length=800)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    expected_effect: str = Field(default="", max_length=400)


ARCHITECT_PROMPT = (
    "You are the Performance Engineer running a long-horizon evolution campaign for one provisioned LLM serving "
    "platform on NVIDIA GPUs. Traffic has changed. You receive the observation, the incumbent architecture, a bounded "
    "context packet from MongoDB memory (policy, campaign checkpoint, lessons, past experiments, regime winners, "
    "documentation), and a numbered menu of architecture moves that are already sized for this hardware and rendered. "
    "Pick exactly one option. Prefer a move that memory shows won in a similar regime; avoid ones memory shows lost. "
    "State a falsifiable hypothesis with expected numbers. Cite only item_ids present in the context. The candidate "
    "will be replayed against identical traffic and deterministic gates decide; never claim it is proven. No Markdown."
)


def llm_architect(payload):
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    from strands import Agent
    from strands.models.openai import OpenAIModel
    model = OpenAIModel(client_args={"api_key": config.OPENROUTER_API_KEY, "base_url": config.OPENROUTER_BASE_URL,
                                     "timeout": 90, "max_retries": 1},
                        model_id=config.AGENT_MODEL, params={"temperature": 0.1, "max_tokens": 1200})
    agent = Agent(model=model, callback_handler=None, system_prompt=ARCHITECT_PROMPT)
    result = agent(json.dumps(payload, default=str), structured_output_model=ArchitectChoice).structured_output
    if result is None:
        raise ValueError("empty structured output")
    return result


# ---------------------------------------------------------------- workflow

class EvolutionCampaign:
    """Durable controller; one ``tick`` advances one persisted stage."""

    def __init__(self, database, campaign_id, *, executor_factory=SimExecutor, architect=llm_architect,
                 evaluator=None, clock=time.time, repeats=None):
        self.raw, self.db = database, ProductionDB(database)
        self.campaign_id, self.clock = campaign_id, clock
        self.store = store(database)
        self.executor_factory, self.architect = executor_factory, architect
        self.evaluator = evaluator or Evaluator()
        self.repeats = repeats or config.TRIAL_REPEATS
        self.owner = f"{socket.gethostname()}:{uuid4().hex[:8]}"

    # -- creation / queries
    @staticmethod
    def campaign_id_for(base_row, timeline, policy):
        return "evo-" + digest({"base": base_row.get("change_id") or base_row["plan_id"],
                                "timeline": timeline, "policy": policy}, 14)

    def create(self, base_row, timeline, policy, budget):
        base_id = base_row.get("change_id") or base_row["plan_id"]
        arch = ProdArch.from_plan(base_row["plan"], policy.get("gpu_hour_cost", 1.0))
        campaign = self.store.bootstrap(
            self.campaign_id, goal="Keep the provisioned platform within SLO at the lowest GPU cost as traffic changes",
            max_experiments=budget, policy=policy,
            initial_state={"summary": f"Campaign started from {base_id}", "phase_index": 0, "attempt": 0,
                           "incumbent": arch.model_dump(), "revision": 0, "decisions": []})
        self.db.campaigns.update_one({"_id": self.campaign_id}, {"$set": {
            "lineage": base_row["plan_id"], "base_id": base_id, "timeline": timeline}})
        self.db.campaign_bases.update_one({"_id": self.campaign_id}, {"$setOnInsert": {
            "plan_id": base_row["plan_id"], "input": base_row["input"], "plan": base_row["plan"]}}, upsert=True)
        self.db.policies.update_one({"_id": self.campaign_id}, {"$setOnInsert": {
            "campaign_id": self.campaign_id, **policy}}, upsert=True)
        self.db.revisions.update_one({"campaign_id": self.campaign_id, "revision": 0}, {"$setOnInsert": {
            "campaign_id": self.campaign_id, "revision": 0, "arch": arch.model_dump(), "summary": arch.summary(),
            "phase_index": None, "experiment_id": None, "reason": f"Provisioned plan {base_id}",
            "plan": base_row["plan"], "files": base_row["files"], "diff": "", "created_at": self.clock()}}, upsert=True)
        if campaign["revision"] == 0 and not campaign.get("lineage"):
            self._event("start", f"Campaign created over {len(timeline)} traffic phases from {base_id}")
        return self.get()

    def get(self):
        return _plain(self.db.campaigns.find_one({"_id": self.campaign_id}))

    def tick(self):
        campaign = self.store.acquire_lease(self.campaign_id, self.owner, ttl_s=900)
        if not campaign or campaign["stage"] == "COMPLETE":
            if campaign:
                self.store.release_lease(self.campaign_id, self.owner, lease_token=campaign["lease"]["token"])
            return False
        token = campaign["lease"]["token"]
        try:
            handler = getattr(self, f"_{campaign['stage'].lower()}")
            return handler(campaign, token)
        except Exception as exc:
            current = self.store.get(self.campaign_id)
            if current and (current.get("lease") or {}).get("token") == token:
                self.store.transition(self.campaign_id, expected_revision=current["revision"], stage=current["stage"],
                                      updates={"last_error": f"{type(exc).__name__}: {exc}"[:500]},
                                      checkpoint=current.get("checkpoint", {}), owner=self.owner, lease_token=token)
            self._event("error", f"{campaign['stage']} failed: {exc}")
            raise
        finally:
            self.store.release_lease(self.campaign_id, self.owner, lease_token=token)

    # -- helpers
    def _to(self, campaign, token, stage, checkpoint, **updates):
        updates.setdefault("last_error", None)
        return self.store.transition(self.campaign_id, expected_revision=campaign["revision"], stage=stage,
                                     updates=updates, checkpoint=checkpoint, owner=self.owner, lease_token=token)

    def _event(self, kind, msg, **data):
        self.db.events.insert_one({"campaign_id": self.campaign_id, "ts": self.clock(), "kind": kind, "msg": msg, **data})

    def _policy(self, campaign):
        pol = campaign["policy"]
        return OptimizationPolicy(slo_p95_ms=pol["slo_p95_ms"], max_error_rate=pol["max_error_rate"],
                                  min_trial_requests=pol.get("min_trial_requests", 30), trial_repeats=self.repeats,
                                  min_cost_improvement=pol.get("min_cost_improvement", 0.05),
                                  max_latency_regression=pol.get("max_latency_regression", 0.10))

    def _base_row(self, campaign):
        return {**_plain(self.db.campaign_bases.find_one({"_id": self.campaign_id})), "files": {}}

    def _replay(self, campaign, index, seed_tag="observe"):
        phase = campaign["timeline"][index]
        seed = int(digest({"c": self.campaign_id, "phase": index, "tag": seed_tag}, 8), 16)
        prof = traffic_profile(phase, seed)
        plan = make_replay_plan(prof, phase["duration_s"], seed=seed, campaign_id=self.campaign_id,
                                experiment_id=f"{self.campaign_id}:p{index}:{seed_tag}")
        self.db.replay_plans.update_one({"replay_id": plan.replay_id}, {"$setOnInsert": plan.model_dump()}, upsert=True)
        return ReplayPlan(**_plain(self.db.replay_plans.find_one({"replay_id": plan.replay_id}))), prof

    def _runner(self, campaign, phase):
        executor = self.executor_factory(self._base_row(campaign)["input"]["inventory"])
        executor.prefix_reuse = phase["prefix_reuse"]
        return ExperimentRunner(self.db, executor=executor), executor

    # -- stages
    def _observe(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        index, timeline = cp["phase_index"], campaign["timeline"]
        if index >= len(timeline):
            self._event("complete", f"Timeline finished at revision {cp['revision']}")
            self._to(campaign, token, "COMPLETE", {**cp, "summary": f"Completed {len(timeline)} phases; "
                                                   f"final revision {cp['revision']}"})
            return False
        phase = timeline[index]
        incumbent = ProdArch(**cp["incumbent"])
        replay, prof = self._replay(campaign, index)
        runner, executor = self._runner(campaign, phase)
        try:
            trial = runner.run_trial_sync(f"observe:{self.campaign_id}:p{index}:a{cp['attempt']}:r{cp['revision']}",
                                          incumbent, replay, 0, 0, role="incumbent")
        finally:
            executor.close()
        policy = self._policy(campaign)
        vec, regime_hash = regime(phase, prof.prompt_tokens)
        s = sorted(prof.prompt_tokens)
        obs = {"phase": phase["name"], "rps": phase["rps"], "n": trial.n, "p50_ms": round(trial.p50_ms),
               "p95_ms": round(trial.p95_ms), "error_rate": round(trial.error_rate, 4),
               "prompt_p50_tokens": s[len(s) // 2], "prompt_p90_tokens": s[int(len(s) * .9)],
               "long_frac": round(sum(t >= phase["long_tokens"][0] for t in s) / len(s), 3),
               "prefix_reuse": phase["prefix_reuse"], "gpus": incumbent.gpus(), "architecture": incumbent.summary()}
        effective = sum(s) / len(s) * (1 - 0.75 * phase["prefix_reuse"])
        inventory = self._base_row(campaign)["input"]["inventory"]
        obs["est_utilization"] = round(phase["rps"] / _capacity_rps(incumbent, inventory, effective,
                                                                    phase["output_tokens"]), 3)
        trigger = None
        if trial.p95_ms > policy.slo_p95_ms or trial.error_rate > policy.max_error_rate:
            trigger = "slo_breach"
        elif trial.p95_ms < policy.slo_p95_ms * 0.75 and obs["est_utilization"] < 0.3 and incumbent.workers() > 1:
            trigger = "overprovisioned"
        window_id = f"window:{self.campaign_id}:p{index}:a{cp['attempt']}:r{cp['revision']}"
        self.db.metric_windows.update_one({"window_id": window_id}, {"$setOnInsert": {
            "window_id": window_id, "campaign_id": self.campaign_id, "phase_index": index, "attempt": cp["attempt"],
            "trigger": trigger, "metrics": obs, "regime_vector": vec, "regime_hash": regime_hash,
            "replay_id": replay.replay_id, "trial_id": trial.trial_id, "arch_key": incumbent.key(),
            "summary": f"{phase['name']}: p95 {obs['p95_ms']} ms at {phase['rps']} rps on {incumbent.summary()}",
            "ts": self.clock()}}, upsert=True)
        cp.update({"observation": obs, "window_id": window_id, "trigger": trigger, "regime_vector": vec,
                   "regime_hash": regime_hash, "replay_id": replay.replay_id,
                   "evidence_ids": [window_id], "summary": f"{phase['name']}: {trigger or 'within SLO'}"})
        exhausted = campaign["experiments_spent"] >= campaign["max_experiments"]
        if trigger is None or exhausted:
            why = ("SLO met with healthy headroom; no change" if trigger is None
                   else "experiment budget exhausted; holding")
            self._event("observe", f"{phase['name']}: p95 {obs['p95_ms']} ms vs {policy.slo_p95_ms:g} ms SLO; {why}",
                        phase_index=index)
            cp.update({"hold": why, "advance": True})
            self._to(campaign, token, "CHECKPOINT", cp)
            return True
        self._event("observe", f"{phase['name']}: p95 {obs['p95_ms']} ms vs {policy.slo_p95_ms:g} ms SLO; "
                    f"{trigger.replace('_', ' ')}", phase_index=index, window_id=window_id)
        self._to(campaign, token, "BUILD_CONTEXT", cp)
        return True

    def _build_context(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        inc = ProdArch(**cp["incumbent"])
        packet = compile_context(self.raw, campaign_id=self.campaign_id, lineage=campaign["lineage"],
                                 query=f"{cp['trigger']} {cp['observation']['phase']} {inc.recipe_id} {inc.model_id}",
                                 observation={"window_id": cp["window_id"], **cp["observation"]},
                                 regime_vector=cp["regime_vector"])
        view = manifest_view(packet)
        cp["context"] = view
        cp["recalled"] = [{"item_id": i, "wins": w, "losses": l, "arch": a.model_dump()}
                          for w, l, i, a in recalled(packet, None)]
        cp["evidence_ids"] = list(dict.fromkeys(cp["evidence_ids"] + [x["item_id"] for x in view["included"]]))
        self._event("recall", f"Compiled {len(view['included'])} memories in {view['tokens']} of {view['budget']} tokens",
                    manifest_id=view["manifest_id"], phase_index=cp["phase_index"])
        self._to(campaign, token, "PROPOSE", cp)
        return True

    def _propose(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        signature = f"{cp['window_id']}"
        draft = self.db.experiments.find_one({"campaign_id": self.campaign_id, "proposal_signature": signature})
        base_row = self._base_row(campaign)
        incumbent = ProdArch(**cp["incumbent"])
        if draft is None:
            options = self._options(campaign, cp, incumbent, base_row)
            if not options:
                self._event("hold", "No valid architecture move fits this hardware for the trigger", phase_index=cp["phase_index"])
                cp.update({"hold": "no valid move for this hardware", "advance": True})
                self._to(campaign, token, "CHECKPOINT", cp)
                return True
            choice, run = self._choose(cp, incumbent, options)
            picked = options[choice["option"]]
            experiment_id = f"experiment-{uuid4().hex[:12]}"
            idempotency = f"{self.campaign_id}:{incumbent.key()}:{cp['regime_hash']}:{picked['arch'].key()}"
            existing = self.db.experiments.find_one({"idempotency_key": idempotency})
            if existing and existing.get("status") == "terminal":
                self._event("hold", "An identical experiment already ran in this regime; reusing its lesson",
                            experiment_id=existing["experiment_id"], phase_index=cp["phase_index"])
                cp.update({"hold": "identical experiment already terminal", "advance": True,
                           "experiment_id": existing["experiment_id"]})
                self._to(campaign, token, "CHECKPOINT", cp)
                return True
            draft = {"experiment_id": experiment_id, "campaign_id": self.campaign_id, "idempotency_key": idempotency,
                     "proposal_signature": signature, "phase_index": cp["phase_index"], "attempt": cp["attempt"],
                     "status": "proposed", "trigger": cp["trigger"], "move": picked["move"],
                     "migration": picked["migration"], "hypothesis": choice["hypothesis"],
                     "expected_effect": choice.get("expected_effect", ""), "evidence_ids": choice["evidence_ids"],
                     "candidate": picked["arch"].model_dump(), "candidate_key": picked["arch"].key(),
                     "candidate_summary": picked["arch"].summary(), "incumbent": incumbent.model_dump(),
                     "incumbent_key": incumbent.key(), "incumbent_summary": incumbent.summary(),
                     "recalled": picked.get("recalled", False),
                     "options": [{"move": o["move"], "summary": o["arch"].summary(), "why": o["why"],
                                  "migration": o["migration"], "recalled": o.get("recalled", False)} for o in options],
                     "regime_vector": cp["regime_vector"], "regime_hash": cp["regime_hash"],
                     "architect_run": run, "manifest_id": cp["context"]["manifest_id"], "created_at": self.clock()}
            self.db.experiments.update_one({"experiment_id": experiment_id}, {"$setOnInsert": draft}, upsert=True)
            draft = self.db.experiments.find_one({"experiment_id": experiment_id})
        self.store.debit_budget(self.campaign_id, draft["idempotency_key"])
        cp.update({"experiment_id": draft["experiment_id"], "candidate": draft["candidate"]})
        self._event("propose", draft["hypothesis"], experiment_id=draft["experiment_id"], move=draft["move"],
                    phase_index=cp["phase_index"])
        self._to(self.store.get(self.campaign_id), token, "VALIDATE", cp, active_experiment_id=draft["experiment_id"])
        return True

    def _options(self, campaign, cp, incumbent, base_row):
        losers = {r["arch_key"] for r in self.db.regimes.find({
            "lineage": campaign["lineage"], "bucket": [round(x, 1) for x in cp["regime_vector"]]})
            if r.get("losses", 0) > r.get("wins", 0)}
        menu = moves(incumbent, cp["observation"], cp["trigger"], base_row)
        rng = random.Random(cp["window_id"])
        for item in sorted(cp.get("recalled", []), key=lambda r: -rng.betavariate(r["wins"] + 1, r["losses"] + 1)):
            arch = ProdArch(**item["arch"])
            if arch.key() != incumbent.key() and arch.key() not in losers:
                menu.insert(0, {"move": "recall", "arch": arch, "recalled": True, "migration": arch.recipe_id != incumbent.recipe_id,
                                "why": f"Won {item['wins']}× / lost {item['losses']}× in a similar traffic regime (memory {item['item_id']})."})
                break
        valid = []
        for option in menu:
            if option["arch"].key() in losers:
                continue
            _, _, _, blockers = plan_for(base_row, option["arch"], option["why"])
            if not blockers:
                valid.append(option)
        return valid[:5]

    def _choose(self, cp, incumbent, options):
        ids = set(cp["evidence_ids"])
        fallback = {"option": 0, "hypothesis": f"{options[0]['why']} Expect p95 to return under the SLO on the same replay.",
                    "evidence_ids": [cp["window_id"]] + [i for i in cp["evidence_ids"] if i.startswith("lesson")][:3],
                    "expected_effect": ""}
        if self.architect is None:
            return fallback, {"source": "rules", "error": None}
        payload = {"observation": cp["observation"], "trigger": cp["trigger"], "incumbent": incumbent.summary(),
                   "context": cp["context"]["included"],
                   "options": [{"option": i, "move": o["move"], "architecture": o["arch"].summary(), "why": o["why"],
                                "stack_migration": o["migration"], "recalled_from_memory": o.get("recalled", False)}
                               for i, o in enumerate(options)]}
        try:
            result = ArchitectChoice.model_validate(self.architect(payload))
            if result.option >= len(options):
                raise ValueError(f"option {result.option} is not on the menu")
            cited = [e for e in result.evidence_ids if e in ids] or [cp["window_id"]]
            return ({"option": result.option, "hypothesis": result.hypothesis, "evidence_ids": cited,
                     "expected_effect": result.expected_effect}, {"source": f"llm:{config.AGENT_MODEL}", "error": None})
        except Exception as exc:  # noqa: BLE001
            return fallback, {"source": "rules", "error": f"{type(exc).__name__}: {exc}"[:300]}

    def _validate(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        exp = self.db.experiments.find_one({"experiment_id": cp["experiment_id"]})
        candidate, incumbent = ProdArch(**exp["candidate"]), ProdArch(**cp["incumbent"])
        p, files, checks, blockers = plan_for(self._base_row(campaign), candidate, exp["hypothesis"])
        reasons = list(blockers)
        if candidate.key() == incumbent.key():
            reasons.append("candidate is identical to the incumbent")
        unresolved = [e for e in exp["evidence_ids"] if e not in set(cp["evidence_ids"])]
        if unresolved:
            reasons.append(f"cites evidence outside the context: {unresolved}")
        self.db.experiments.update_one({"experiment_id": cp["experiment_id"]}, {"$set": {
            "status": "rejected" if reasons else "validated", "validation_reasons": reasons, "validation": checks}})
        self.db.candidates.update_one({"experiment_id": cp["experiment_id"]}, {"$setOnInsert": {
            "experiment_id": cp["experiment_id"], "plan": p, "files": files}}, upsert=True)
        cp["validation_reasons"] = reasons
        self._event("validate", "rejected: " + "; ".join(reasons) if reasons else
                    f"{len(checks)}/{len(checks)} rendered files pass schema validation; fits the inventory",
                    experiment_id=cp["experiment_id"], phase_index=cp["phase_index"])
        self._to(campaign, token, "REJECT" if reasons else "PREPARE_REPLAY", cp)
        return True

    def _prepare_replay(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        self.db.experiments.update_one({"experiment_id": cp["experiment_id"]},
                                       {"$set": {"status": "replay_ready", "replay_id": cp["replay_id"]}})
        replay = self.db.replay_plans.find_one({"replay_id": cp["replay_id"]})
        self._event("test", f"Incumbent and candidate will replay the same {len(replay['events'])} persisted requests",
                    replay_id=cp["replay_id"], experiment_id=cp["experiment_id"], phase_index=cp["phase_index"])
        self._to(campaign, token, "RUN_TRIALS", cp)
        return True

    def _run_trials(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        phase = campaign["timeline"][cp["phase_index"]]
        replay = ReplayPlan(**_plain(self.db.replay_plans.find_one({"replay_id": cp["replay_id"]})))
        runner, executor = self._runner(campaign, phase)
        try:
            inc, cand = runner.run_repeated_trials(cp["experiment_id"], ProdArch(**cp["incumbent"]),
                                                   ProdArch(**cp["candidate"]), replay, self.repeats)
        finally:
            executor.close()
        cp["incumbent_trial_ids"] = [t.trial_id for t in inc]
        cp["candidate_trial_ids"] = [t.trial_id for t in cand]
        self.db.experiments.update_one({"experiment_id": cp["experiment_id"]}, {"$set": {
            "status": "trials_complete", "trials": {
                "incumbent": [{"repeat": t.repeat, "p50_ms": round(t.p50_ms), "p95_ms": round(t.p95_ms),
                               "error_rate": t.error_rate, "n": t.n} for t in inc],
                "candidate": [{"repeat": t.repeat, "p50_ms": round(t.p50_ms), "p95_ms": round(t.p95_ms),
                               "error_rate": t.error_rate, "n": t.n} for t in cand]}}})
        self._event("results", f"Completed {len(inc) + len(cand)} paired trials on simulator backends",
                    experiment_id=cp["experiment_id"], phase_index=cp["phase_index"])
        self._to(campaign, token, "EVALUATE", cp)
        return True

    def _evaluate(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        load = lambda ids: [TrialResult(**_plain(self.db.trials.find_one({"trial_id": i}))) for i in ids]  # noqa: E731
        evaluation = self.evaluator.evaluate_sync(cp["experiment_id"], load(cp["incumbent_trial_ids"]),
                                                  load(cp["candidate_trial_ids"]), self._policy(campaign), cp["revision"])
        self.db.evaluations.update_one({"evaluation_id": evaluation.evaluation_id},
                                       {"$setOnInsert": {**evaluation.model_dump(), "campaign_id": self.campaign_id}},
                                       upsert=True)
        decision = evaluation.decision.value
        self.db.experiments.update_one({"experiment_id": cp["experiment_id"]}, {"$set": {
            "status": decision, "decision": decision, "evaluation": evaluation.model_dump()}})
        cp["evaluation_id"] = evaluation.evaluation_id
        self._event("evaluate", f"Gates decided: {decision}", experiment_id=cp["experiment_id"],
                    phase_index=cp["phase_index"])
        self._to(campaign, token, "PROMOTE" if decision == "promote" else "REJECT", cp)
        return True

    def _new_revision(self, campaign, cp, arch_dict, plan, files, reason, experiment_id, status):
        prev = self.db.revisions.find_one({"campaign_id": self.campaign_id}, sort=[("revision", -1)])
        number = prev["revision"] + 1
        self.db.revisions.update_one({"campaign_id": self.campaign_id, "revision": number}, {"$setOnInsert": {
            "campaign_id": self.campaign_id, "revision": number, "arch": arch_dict,
            "summary": ProdArch(**arch_dict).summary(), "phase_index": cp["phase_index"],
            "phase": campaign["timeline"][cp["phase_index"]]["name"], "experiment_id": experiment_id,
            "reason": reason, "status": status, "plan": plan, "files": files,
            "diff": unified_diff(prev["files"], files), "created_at": self.clock()}}, upsert=True)
        return number

    def _promote(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        exp = self.db.experiments.find_one({"experiment_id": cp["experiment_id"]})
        bundle = self.db.candidates.find_one({"experiment_id": cp["experiment_id"]})
        existing = self.db.revisions.find_one({"campaign_id": self.campaign_id, "experiment_id": cp["experiment_id"]})
        number = existing["revision"] if existing else self._new_revision(
            campaign, cp, exp["candidate"], bundle["plan"], bundle["files"], exp["hypothesis"],
            cp["experiment_id"], "offline_evaluated")
        cp.update({"previous_incumbent": cp["incumbent"], "previous_revision": cp["revision"],
                   "incumbent": exp["candidate"], "revision": number, "promoted": True})
        self.db.experiments.update_one({"experiment_id": cp["experiment_id"]},
                                       {"$set": {"status": "promoted", "revision": number}})
        self._event("promote", f"Revision {number}: {exp['candidate_summary']}"
                    + (" (stack migration: operator review required)" if exp.get("migration") else ""),
                    experiment_id=cp["experiment_id"], phase_index=cp["phase_index"])
        self._to(campaign, token, "VERIFY_LIVE", cp)
        return True

    def _verify_live(self, campaign, token):
        """No live cluster: re-check the promoted revision on a fresh hold-out replay of the same phase."""
        cp = dict(campaign["checkpoint"])
        phase, policy = campaign["timeline"][cp["phase_index"]], self._policy(campaign)
        replay, _ = self._replay(campaign, cp["phase_index"], seed_tag=f"holdout-{cp['experiment_id']}")
        runner, executor = self._runner(campaign, phase)
        try:
            trial = runner.run_trial_sync(f"holdout:{cp['experiment_id']}", ProdArch(**cp["incumbent"]),
                                          replay, 0, 0, role="candidate")
        finally:
            executor.close()
        passed = trial.p95_ms <= policy.slo_p95_ms and trial.error_rate <= policy.max_error_rate
        cp["holdout"] = {"p95_ms": round(trial.p95_ms), "error_rate": trial.error_rate, "passed": passed,
                         "replay_id": replay.replay_id}
        self.db.experiments.update_one({"experiment_id": cp["experiment_id"]}, {"$set": {"holdout": cp["holdout"]}})
        if passed:
            self._event("verify", f"Hold-out replay passed: p95 {round(trial.p95_ms)} ms", phase_index=cp["phase_index"],
                        experiment_id=cp["experiment_id"])
        else:
            prev = self.db.revisions.find_one({"campaign_id": self.campaign_id, "revision": cp["previous_revision"]})
            number = self._new_revision(campaign, cp, prev["arch"], prev["plan"], prev["files"],
                                        f"Rolled back: hold-out p95 {round(trial.p95_ms)} ms", cp["experiment_id"],
                                        "rolled_back")
            cp.update({"incumbent": cp["previous_incumbent"], "revision": number, "promoted": False, "rolled_back": True})
            self.db.experiments.update_one({"experiment_id": cp["experiment_id"]}, {"$set": {"status": "rolled_back"}})
            self._event("rollback", f"Hold-out replay failed (p95 {round(trial.p95_ms)} ms); restored as revision {number}",
                        phase_index=cp["phase_index"], experiment_id=cp["experiment_id"])
        self._to(campaign, token, "LEARN", cp)
        return True

    def _reject(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        self._event("hold", "Candidate rejected; incumbent kept", experiment_id=cp.get("experiment_id"),
                    phase_index=cp["phase_index"])
        cp["promoted"] = False
        self._to(campaign, token, "LEARN", cp)
        return True

    def _learn(self, campaign, token):
        cp = dict(campaign["checkpoint"])
        exp = _plain(self.db.experiments.find_one({"experiment_id": cp["experiment_id"]}))
        won = bool(cp.get("promoted")) and not cp.get("rolled_back")
        reasons = (exp.get("evaluation") or {}).get("reasons") or exp.get("validation_reasons") or []
        gpu = gpu_spec(self._base_row(campaign)["input"]["inventory"])[0]
        cand = ProdArch(**exp["candidate"])
        lesson = record_lesson(self.raw, record={
            "experiment_id": exp["experiment_id"], "hypothesis": f"{exp['move']}: {exp['hypothesis']}",
            "decision": "promoted" if won else ("rolled back" if cp.get("rolled_back") else "rejected"),
            "reason": "; ".join(reasons), "candidate_key": exp["candidate_key"],
            "confidence": 0.95 if exp.get("evaluation") else 0.5},
            scope={"lineage": campaign["lineage"], "recipe": cand.recipe_id, "model": cand.model_id, "gpu": gpu,
                   "regime_hash": exp["regime_hash"], "trigger": exp["trigger"], "arch_key": exp["candidate_key"]},
            confirmed=won, vector=exp["regime_vector"], arch_key=exp["candidate_key"], clock=self.clock)
        self.db.experiments.update_one({"experiment_id": exp["experiment_id"]}, {"$set": {
            "status": "terminal", "outcome_won": won, "lesson_id": lesson["lesson_id"]}})
        bucket = [round(x, 1) for x in exp["regime_vector"]]
        scope = {"campaign_id": self.campaign_id, "regime_hash": exp["regime_hash"],
                 "candidate_key": exp["candidate_key"], "status": "terminal"}
        self.db.regimes.update_one({"lineage": campaign["lineage"], "bucket": bucket, "arch_key": exp["candidate_key"]}, {"$set": {
            "lineage": campaign["lineage"], "vector": exp["regime_vector"], "arch": exp["candidate"],
            "wins": self.db.experiments.count_documents({**scope, "outcome_won": True}),
            "losses": self.db.experiments.count_documents({**scope, "outcome_won": False}),
            "summary": f"{cand.summary()} in {campaign['timeline'][cp['phase_index']]['name']}", "ts": self.clock()}},
            upsert=True)
        cp["lesson_id"] = lesson["lesson_id"]
        self._event("learn", lesson["claim"][:300], lesson_id=lesson["lesson_id"], phase_index=cp["phase_index"])
        self._to(campaign, token, "CHECKPOINT", cp)
        return True

    def _checkpoint(self, campaign, token):
        old = campaign["checkpoint"]
        advance = old.get("advance") or old.get("attempt", 0) + 1 >= MAX_ATTEMPTS_PER_PHASE
        index = old["phase_index"] + (1 if advance else 0)
        decision = old.get("hold") or ("promoted revision %d" % old["revision"] if old.get("promoted") else
                                       "rolled back" if old.get("rolled_back") else "rejected")
        phase = campaign["timeline"][old["phase_index"]]["name"]
        cp = {"phase_index": index, "attempt": 0 if advance else old.get("attempt", 0) + 1,
              "incumbent": old["incumbent"], "revision": old["revision"],
              "decisions": (old.get("decisions", []) + [f"{phase}: {decision}"])[-12:],
              "evidence_ids": [x for x in (old.get("window_id"), old.get("experiment_id"), old.get("lesson_id")) if x],
              "summary": f"{phase}: {decision}; incumbent is revision {old['revision']}", "updated_at": self.clock()}
        self._event("checkpoint", cp["summary"], phase_index=old["phase_index"])
        self._to(campaign, token, "OBSERVE", cp, active_experiment_id=None)
        return True


def run_until_idle(campaign: EvolutionCampaign, max_ticks=500):
    for _ in range(max_ticks):
        if not campaign.tick():
            return
