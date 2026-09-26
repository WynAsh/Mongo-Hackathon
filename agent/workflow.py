"""Restart-safe long-horizon optimization workflow.

MongoDB owns all durable state. Each call advances one persisted stage, making
process restarts safe without retaining model conversation history.
"""
from __future__ import annotations

import hashlib
import socket
import time
from typing import Callable
from uuid import uuid4

from agent.architect import StrandsArchitect
from agent.evaluator import Evaluator
from agent.experiments import ExperimentRunner
from agent.promotion import PromotionService
from agent.state import CampaignStore
from common import config, db as db_module
from common.contracts import (
    Arch, Campaign, CampaignStage, Evaluation, EvaluationDecision,
    ExperimentProposal, ObservationSnapshot, OptimizationPolicy, PromotionResult,
    ReplayPlan, TrialResult,
)
from gateway import metrics
from memory.context import ContextCompiler
from memory.curator import MemoryCurator
from memory.embeddings import OpenRouterEmbedder
from memory.store import _bucket


def _plain(document):
    return None if document is None else {k: v for k, v in document.items() if k != "_id"}


def _regime_hash(vector: list[float]) -> str:
    return hashlib.sha256(",".join(f"{x:.4f}" for x in vector).encode()).hexdigest()[:20]


class DurableOptimizationWorkflow:
    """Deterministic controller around the bounded-context Strands architect."""

    def __init__(self, database=None, *, campaign_id=None, owner=None,
                 observer: Callable | None = None, architect=None,
                 context_compiler=None, runner=None, evaluator=None,
                 promotion=None, curator=None, clock=time.time):
        self.database = database if database is not None else db_module.db()
        self.campaign_id = campaign_id or config.CAMPAIGN_ID
        self.owner = owner or f"{socket.gethostname()}:{uuid4().hex[:8]}"
        self.clock = clock
        self.store = CampaignStore(self.database, clock=clock)
        self.observer = observer or metrics.window
        self.architect = architect or StrandsArchitect()
        self.compiler = context_compiler or ContextCompiler(
            self.database, token_budget=config.AGENT_CONTEXT_TOKENS)
        self.runner = runner or ExperimentRunner(self.database)
        self.evaluator = evaluator or Evaluator()
        self.promotion = promotion or PromotionService(self.database)
        self.curator = curator or MemoryCurator()
        self.embedder = OpenRouterEmbedder()

    def bootstrap(self):
        live = self._live_arch()
        return self.store.bootstrap(
            self.campaign_id, max_experiments=config.CAMPAIGN_MAX_EXPERIMENTS,
            policy=self._policy().model_dump(),
            initial_state={"summary": "Campaign initialized",
                           "incumbent_version": live.version if live else None})

    def tick(self) -> bool:
        self.bootstrap()
        campaign = self.store.acquire_lease(self.campaign_id, self.owner, ttl_s=3600)
        if not campaign:
            return False
        token = campaign["lease"]["token"]
        try:
            return self._advance(campaign, token)
        except Exception as exc:
            current = self.store.get(self.campaign_id)
            if current and (current.get("lease") or {}).get("token") == token:
                self.store.transition(
                    self.campaign_id, expected_revision=current["revision"],
                    stage=current["stage"], updates={"last_error": str(exc)},
                    checkpoint=current.get("checkpoint", {}), owner=self.owner,
                    lease_token=token)
            self._event("error", f"long-horizon stage failed: {exc}")
            raise
        finally:
            self.store.release_lease(self.campaign_id, self.owner, lease_token=token)

    def _advance(self, campaign, token):
        handlers = {
            CampaignStage.OBSERVE: self._observe,
            CampaignStage.BUILD_CONTEXT: self._build_context,
            CampaignStage.PROPOSE: self._propose,
            CampaignStage.VALIDATE: self._validate,
            CampaignStage.PREPARE_REPLAY: self._prepare_replay,
            CampaignStage.RUN_TRIALS: self._run_trials,
            CampaignStage.EVALUATE: self._evaluate,
            CampaignStage.PROMOTE: self._promote,
            CampaignStage.REJECT: self._reject,
            CampaignStage.VERIFY_LIVE: self._verify_live,
            CampaignStage.LEARN: self._learn,
            CampaignStage.CHECKPOINT: self._checkpoint,
        }
        handler = handlers.get(CampaignStage(campaign["stage"]))
        return False if handler is None else handler(campaign, token)

    def _transition(self, campaign, token, stage, checkpoint, **updates):
        updates.setdefault("last_error", None)
        return self.store.transition(
            self.campaign_id, expected_revision=campaign["revision"],
            stage=stage.value, updates=updates, checkpoint=checkpoint,
            owner=self.owner, lease_token=token)

    def _observe(self, campaign, token):
        live = self._live_arch()
        if live is None:
            return False
        observation = self.observer(config.WINDOW_S, live.version)
        minimum = min(15, self._policy().min_trial_requests)
        if (observation.get("n", 0) < minimum or observation.get("p95_ms") is None
                or not observation.get("stable")):
            return False
        policy = self._policy()
        error_rate = observation.get("errors", 0) / max(observation["n"], 1)
        trigger = None
        if observation["p95_ms"] > policy.slo_p95_ms or error_rate > policy.max_error_rate:
            trigger = "slo_breach"
        elif observation["p95_ms"] < policy.slo_p95_ms * .5 and live.usd_hr() > .5:
            trigger = "overprovisioned"
        if trigger is None or campaign["experiments_spent"] >= campaign["max_experiments"]:
            return False
        regime = observation["regime_vector"]
        signature = f"{trigger}:{live.version}:{_regime_hash(regime)}"
        if campaign.get("checkpoint", {}).get("last_noop_signature") == signature:
            return False
        snapshot = ObservationSnapshot(architecture=live, regime_vector=regime,
                                       metrics=observation, observed_at=self.clock())
        window_id = f"window:{self.campaign_id}:{int(self.clock())}:{live.version}"
        self.database.metric_windows.update_one(
            {"window_id": window_id}, {"$setOnInsert": {
                "window_id": window_id, "campaign_id": self.campaign_id,
                "ts": self.clock(), **snapshot.model_dump()}}, upsert=True)
        checkpoint = {
            "summary": f"Observed {trigger} on v{live.version}", "trigger": trigger,
            "observation": snapshot.model_dump(), "regime_hash": _regime_hash(regime),
            "signature": signature, "evidence_ids": [window_id],
        }
        self._event("observe", f"{trigger.replace('_', ' ')} detected; compiling context",
                    observation_id=window_id, regime_hash=checkpoint["regime_hash"])
        self._transition(campaign, token, CampaignStage.BUILD_CONTEXT, checkpoint,
                         incumbent_version=live.version, incumbent_key=live.key())
        return True

    def _build_context(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        observation = checkpoint["observation"]
        packet = self.compiler.compile(
            campaign_id=self.campaign_id, stage=CampaignStage.PROPOSE.value,
            query=f"{checkpoint['trigger']} {checkpoint['regime_hash']}",
            observation=observation, regime_vector=observation["regime_vector"],
            scope_filter={"regime_hash": checkpoint["regime_hash"]})
        manifest = packet["manifest"]
        checkpoint["context_manifest_id"] = manifest["manifest_id"]
        checkpoint["evidence_ids"] = list(dict.fromkeys(
            checkpoint.get("evidence_ids", []) +
            [item["item_id"] for item in manifest["included"]]))
        self._event("recall", f"Selected {len(manifest['included'])} memories in {packet['token_estimate']} tokens",
                    manifest_id=manifest["manifest_id"], evidence_ids=checkpoint["evidence_ids"])
        self._transition(campaign, token, CampaignStage.PROPOSE, checkpoint)
        return True

    def _propose(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        manifest = _plain(self.database.context_manifests.find_one(
            {"manifest_id": checkpoint["context_manifest_id"]}))
        if manifest is None:
            raise RuntimeError("context manifest is missing")
        incumbent = Arch(**checkpoint["observation"]["architecture"])
        # A proposal draft is persisted before the budget debit. If the process
        # dies at either boundary, this exact proposal is reused on restart.
        draft = self.database.experiments.find_one({
            "campaign_id": self.campaign_id,
            "proposal_signature": checkpoint["signature"],
            "status": {"$ne": "terminal"},
        }, sort=[("created_at", -1)])
        if draft:
            proposal = ExperimentProposal(**draft["proposal"])
            idempotency = draft["idempotency_key"]
            existing = draft
        else:
            packet = {"campaign_id": self.campaign_id, "context": manifest["included"],
                      "manifest": manifest, "token_estimate": manifest.get("total_tokens", 0)}
            proposal = self.architect.propose(packet, current=incumbent,
                                              policy=self._policy().model_dump())
            proposal.proposal_id = proposal.proposal_id or f"experiment-{uuid4().hex}"
            proposal.campaign_id = self.campaign_id
            idempotency = ":".join((self.campaign_id, str(incumbent.version),
                                    checkpoint["regime_hash"], proposal.candidate.key()))
            existing = self.database.experiments.find_one({"idempotency_key": idempotency})
        if existing and existing.get("status") == "terminal":
            checkpoint.update({
                "experiment_id": existing["experiment_id"],
                "proposal": existing["proposal"],
                "last_noop_signature": checkpoint.get("signature"),
            })
            self._event("hold", "An identical experiment is already terminal; reusing its lesson",
                        experiment_id=existing["experiment_id"],
                        evidence_ids=[existing.get("lesson_id")])
            self._transition(campaign, token, CampaignStage.CHECKPOINT, checkpoint)
            return True
        if existing:
            experiment_id = existing["experiment_id"]
            proposal = ExperimentProposal(**existing["proposal"])
        else:
            experiment_id = proposal.proposal_id
            self.database.experiments.update_one(
                {"experiment_id": experiment_id}, {"$setOnInsert": {
                    "experiment_id": experiment_id, "campaign_id": self.campaign_id,
                    "idempotency_key": idempotency, "schema_version": 1,
                    "proposal_signature": checkpoint["signature"],
                    "status": "proposed", "proposal": proposal.model_dump(),
                    "hypothesis": proposal.hypothesis,
                    "candidate_key": proposal.candidate.key(), "incumbent_key": incumbent.key(),
                    "incumbent_version": incumbent.version,
                    "regime_vector": checkpoint["observation"]["regime_vector"],
                    "regime_hash": checkpoint["regime_hash"], "created_at": self.clock(),
                    "evidence_ids": proposal.evidence_ids}}, upsert=True)
        self.store.debit_budget(self.campaign_id, idempotency)
        self.database.experiments.update_one(
            {"experiment_id": experiment_id}, {"$set": {"budget_debited": True}})
        checkpoint.update({"experiment_id": experiment_id, "proposal": proposal.model_dump()})
        self._event("propose", proposal.hypothesis, experiment_id=experiment_id,
                    candidates=[{"summary": proposal.candidate.summary(),
                                 "reason": proposal.candidate.reason}],
                    evidence_ids=proposal.evidence_ids)
        # debit_budget changes the document without changing its revision.
        current = self.store.get(self.campaign_id)
        self._transition(current, token, CampaignStage.VALIDATE, checkpoint,
                         active_experiment_id=experiment_id)
        return True

    def _validate(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        proposal = ExperimentProposal(**checkpoint["proposal"])
        incumbent = Arch(**checkpoint["observation"]["architecture"])
        policy = self._policy()
        reasons = []
        if proposal.candidate.key() == incumbent.key():
            reasons.append("candidate is identical to incumbent")
        if any(pool.gpu not in policy.allowed_gpus for pool in proposal.candidate.pools.values()):
            reasons.append("candidate uses a forbidden GPU")
        if sum(p.replicas for p in proposal.candidate.pools.values()) > policy.max_total_replicas:
            reasons.append("candidate exceeds replica policy")
        evidence = set(checkpoint.get("evidence_ids", []))
        drilldown = [item for item in proposal.evidence_ids
                     if item not in evidence and self._evidence_exists(item)]
        missing = [item for item in proposal.evidence_ids
                   if item not in evidence and item not in drilldown]
        if missing:
            reasons.append("proposal cites evidence that cannot be resolved")
        if not proposal.evidence_ids:
            reasons.append("proposal must cite at least one evidence item")
        if drilldown:
            checkpoint["evidence_ids"] = list(dict.fromkeys(
                checkpoint.get("evidence_ids", []) + drilldown))
            checkpoint["drilldown_evidence_ids"] = drilldown
            self.database.experiments.update_one(
                {"experiment_id": checkpoint["experiment_id"]},
                {"$set": {"drilldown_evidence_ids": drilldown}})
        next_stage = CampaignStage.REJECT if reasons else CampaignStage.PREPARE_REPLAY
        checkpoint["validation_reasons"] = reasons
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]},
            {"$set": {"status": "rejected" if reasons else "validated",
                      "validation_reasons": reasons}})
        self._event("validate", "proposal rejected: " + "; ".join(reasons)
                    if reasons else "proposal passed policy validation",
                    experiment_id=checkpoint["experiment_id"])
        self._transition(campaign, token, next_stage, checkpoint)
        return True

    def _prepare_replay(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        proposal = ExperimentProposal(**checkpoint["proposal"])
        replay = self.runner.prepare_replay_sync(
            Campaign(**_plain(campaign)), proposal,
            ObservationSnapshot(**checkpoint["observation"]))
        checkpoint["replay_id"] = replay.replay_id
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]},
            {"$set": {"status": "replay_ready", "replay_id": replay.replay_id,
                      "replay_hash": replay.content_hash}})
        self._event("test", f"Persisted replay {replay.replay_id} with {len(replay.events)} requests",
                    replay_id=replay.replay_id, experiment_id=checkpoint["experiment_id"])
        self._transition(campaign, token, CampaignStage.RUN_TRIALS, checkpoint)
        return True

    def _run_trials(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        proposal = ExperimentProposal(**checkpoint["proposal"])
        incumbent = Arch(**checkpoint["observation"]["architecture"])
        replay_doc = _plain(self.database.replay_plans.find_one({"replay_id": checkpoint["replay_id"]}))
        if replay_doc is None:
            raise RuntimeError("persisted replay plan is missing")
        replay = ReplayPlan(**replay_doc)
        incumbent_trials, candidate_trials = self.runner.run_repeated_trials(
            checkpoint["experiment_id"], incumbent, proposal.candidate, replay,
            proposal.test_plan.repeats, warmup_requests=proposal.test_plan.warmup_requests)
        checkpoint["incumbent_trial_ids"] = [t.trial_id for t in incumbent_trials]
        checkpoint["candidate_trial_ids"] = [t.trial_id for t in candidate_trials]
        trial_ids = checkpoint["incumbent_trial_ids"] + checkpoint["candidate_trial_ids"]
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]},
            {"$set": {"status": "trials_complete", "trial_ids": trial_ids}})
        self._event("results", f"Completed {len(trial_ids)} serial paired trials",
                    experiment_id=checkpoint["experiment_id"], replay_id=replay.replay_id)
        self._transition(campaign, token, CampaignStage.EVALUATE, checkpoint)
        return True

    def _evaluate(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        def load(ids):
            return [TrialResult(**_plain(self.database.trials.find_one({"trial_id": value})))
                    for value in ids]
        evaluation = self.evaluator.evaluate_sync(
            checkpoint["experiment_id"], load(checkpoint["incumbent_trial_ids"]),
            load(checkpoint["candidate_trial_ids"]), self._policy(),
            int(campaign["incumbent_version"]))
        self.database.evaluations.update_one(
            {"evaluation_id": evaluation.evaluation_id},
            {"$setOnInsert": {**evaluation.model_dump(), "campaign_id": self.campaign_id}}, upsert=True)
        checkpoint["evaluation_id"] = evaluation.evaluation_id
        next_stage = (CampaignStage.PROMOTE if evaluation.decision == EvaluationDecision.PROMOTE
                      else CampaignStage.REJECT)
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]}, {"$set": {
                "status": evaluation.decision.value, "decision": evaluation.decision.value,
                "evaluation_id": evaluation.evaluation_id,
                "evaluation": evaluation.model_dump()}})
        self._event("evaluate", f"Evaluation decision: {evaluation.decision.value}",
                    experiment_id=checkpoint["experiment_id"],
                    gates=[gate.model_dump() for gate in evaluation.gates])
        self._transition(campaign, token, next_stage, checkpoint)
        return True

    def _promote(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        proposal = ExperimentProposal(**checkpoint["proposal"])
        evaluation = Evaluation(**_plain(self.database.evaluations.find_one(
            {"evaluation_id": checkpoint["evaluation_id"]})))
        expected = int(campaign["incumbent_version"])
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]}, {"$set": {
                "promotion_intent": {"expected_version": expected,
                                     "candidate_key": proposal.candidate.key(),
                                     "evaluation_id": evaluation.evaluation_id}}})
        live = self._live_arch()
        experiment = self.database.experiments.find_one(
            {"experiment_id": checkpoint["experiment_id"]})
        # Recover the narrow crash window after architecture CAS and before the
        # campaign checkpoint. The expected next version and key prove that the
        # intended candidate is already live.
        if (live and live.version == expected + 1 and live.key() == proposal.candidate.key()
                and experiment.get("promotion_intent")):
            result = PromotionResult(
                promoted=True, architecture=live,
                previous_architecture=Arch(**checkpoint["observation"]["architecture"]),
                reason=f"recovered completed promotion as v{live.version}")
        else:
            result = self.promotion.promote_sync(proposal.candidate, evaluation, expected)
        checkpoint["promotion"] = result.model_dump()
        if not result.promoted:
            checkpoint["decision_reason"] = result.reason
            self.database.experiments.update_one(
                {"experiment_id": checkpoint["experiment_id"]},
                {"$set": {"status": "rejected", "decision": "reject", "reason": result.reason}})
            self._event("hold", f"Promotion refused: {result.reason}",
                        experiment_id=checkpoint["experiment_id"])
            self._transition(campaign, token, CampaignStage.LEARN, checkpoint)
            return True
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]},
            {"$set": {"status": "promoted", "promoted_version": result.architecture.version}})
        self._event("promote", f"Promoted v{result.architecture.version}: {result.architecture.summary()}",
                    experiment_id=checkpoint["experiment_id"], version=result.architecture.version)
        self._transition(campaign, token, CampaignStage.VERIFY_LIVE, checkpoint)
        return True

    def _reject(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        self._event("hold", "Candidate rejected; incumbent retained",
                    experiment_id=checkpoint.get("experiment_id"))
        self._transition(campaign, token, CampaignStage.LEARN, checkpoint)
        return True

    def _verify_live(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        promoted = Arch(**checkpoint["promotion"]["architecture"])
        previous = Arch(**checkpoint["promotion"]["previous_architecture"])
        verified = self.promotion.verify_sync(promoted, self._policy())
        checkpoint["verification_passed"] = verified
        if verified:
            self._event("verify", f"Post-promotion verification passed for v{promoted.version}",
                        version=promoted.version)
        else:
            rollback = self.promotion.rollback_sync(previous, promoted, "live SLO or error regression")
            checkpoint["rollback"] = rollback.model_dump()
            self.database.experiments.update_one(
                {"experiment_id": checkpoint["experiment_id"]}, {"$set": {
                    "status": "rolled_back" if rollback.promoted else "rollback_failed",
                    "decision": "reject",
                    "rollback_version": rollback.architecture.version if rollback.architecture else None}})
            self._event("rollback" if rollback.promoted else "error", rollback.reason,
                        experiment_id=checkpoint["experiment_id"])
        self._transition(campaign, token, CampaignStage.LEARN, checkpoint)
        return True

    def _learn(self, campaign, token):
        checkpoint = dict(campaign["checkpoint"])
        experiment = _plain(self.database.experiments.find_one(
            {"experiment_id": checkpoint["experiment_id"]}))
        evaluation = experiment.get("evaluation", {})
        experiment.update({
            "scope": {"regime_hash": experiment.get("regime_hash")},
            "reason": "; ".join(evaluation.get("reasons", experiment.get("validation_reasons", []))),
            "confidence": self._policy().confidence_level if evaluation else .5})
        proposal = ExperimentProposal(**checkpoint["proposal"])
        curated = self.curator.curate(experiment)
        lesson = curated["lesson"]
        try:
            vectors = self.embedder.embed([lesson["claim"]])
            if vectors:
                lesson["embedding"] = vectors[0]
        except Exception:
            pass
        lesson.update({
            "vector": experiment["regime_vector"], "bucket": _bucket(experiment["regime_vector"]),
            "arch_key": experiment["candidate_key"], "ts": self.clock()})
        lesson["confirmed"] = int(experiment.get("decision") == "promote"
                                  and experiment.get("status") != "rolled_back")
        lesson["contradicted"] = int(not lesson["confirmed"])
        prior = self.database.lessons.find_one(
            {"bucket": lesson["bucket"], "arch_key": lesson["arch_key"],
             "lesson_id": {"$ne": lesson["lesson_id"]}}, sort=[("ts", -1)])
        if prior and bool(prior.get("confirmed")) != bool(lesson["confirmed"]):
            prior_id = prior.get("lesson_id", str(prior.get("_id")))
            lesson["contradictions"] = [prior_id]
            lesson["supersedes"] = [prior_id]
            self.database.lessons.update_one(
                {"_id": prior["_id"]}, {"$set": {
                    "superseded_by": lesson["lesson_id"],
                    "confidence": max(0.0, float(prior.get("confidence", .5)) * .5)}})
        self.database.lessons.update_one(
            {"lesson_id": lesson["lesson_id"]}, {"$setOnInsert": lesson}, upsert=True)
        self.database.summaries.update_one(
            {"summary_id": curated["summary"]["summary_id"]},
            {"$setOnInsert": curated["summary"]}, upsert=True)
        # Mark the episode terminal before materializing aggregate bandit
        # counts. Re-running LEARN computes the same totals instead of adding a
        # second contribution after a crash.
        self.database.experiments.update_one(
            {"experiment_id": checkpoint["experiment_id"]}, {"$set": {
                "status": "terminal", "outcome_won": bool(lesson["confirmed"]),
                "lesson_id": lesson["lesson_id"],
                "summary_id": curated["summary"]["summary_id"]}})
        terminal_count = self.database.experiments.count_documents(
            {"campaign_id": self.campaign_id, "regime_hash": experiment["regime_hash"],
             "status": "terminal"})
        regime_summary = {
            "summary_id": f"regime:{experiment['regime_hash']}:v{terminal_count}",
            "schema_version": 1, "level": "regime", "version": terminal_count,
            "scope": {"regime_hash": experiment["regime_hash"]},
            "text": (f"For regime {experiment['regime_hash']}, {proposal.candidate.key()} "
                     f"was {'confirmed' if lesson['confirmed'] else 'rejected'} by deterministic gates."),
            "source_ids": [experiment["experiment_id"], lesson["lesson_id"]],
            "source_hash": hashlib.sha256(
                f"{experiment['experiment_id']}:{lesson['lesson_id']}".encode()).hexdigest(),
            "created_at": self.clock(),
        }
        self.database.summaries.update_one(
            {"summary_id": regime_summary["summary_id"]},
            {"$setOnInsert": regime_summary}, upsert=True)
        aggregate_filter = {
            "campaign_id": self.campaign_id, "regime_hash": experiment["regime_hash"],
            "candidate_key": proposal.candidate.key(), "status": "terminal",
        }
        wins = self.database.experiments.count_documents({**aggregate_filter, "outcome_won": True})
        losses = self.database.experiments.count_documents({**aggregate_filter, "outcome_won": False})
        self.database.regimes.update_one(
            {"bucket": lesson["bucket"], "arch_key": proposal.candidate.key()},
            {"$set": {"vector": experiment["regime_vector"],
                      "arch": proposal.candidate.model_dump(exclude={"status", "version"}),
                      "ts": self.clock(), "wins": wins, "losses": losses}}, upsert=True)
        checkpoint["lesson_id"] = lesson["lesson_id"]
        self._event("learn", f"Stored scoped lesson {lesson['lesson_id']}",
                    experiment_id=checkpoint["experiment_id"], evidence_ids=lesson["evidence_ids"])
        self._transition(campaign, token, CampaignStage.CHECKPOINT, checkpoint)
        return True

    def _checkpoint(self, campaign, token):
        old = campaign["checkpoint"]
        live = self._live_arch()
        promoted = old.get("promotion", {}).get("architecture")
        summary = (f"Experiment {old.get('experiment_id')} completed; live architecture is "
                   f"v{live.version if live else 'unknown'}")
        checkpoint = {
            "summary": summary,
            "decisions": [old.get("decision_reason", "deterministic gates applied")],
            "evidence_ids": list(dict.fromkeys(old.get("evidence_ids", []) + [x for x in
                (old.get("experiment_id"), old.get("lesson_id"), old.get("evaluation_id")) if x])),
            "last_experiment_id": old.get("experiment_id"),
            "last_manifest_id": old.get("context_manifest_id"),
            "last_replay_id": old.get("replay_id"),
            "last_noop_signature": old.get("signature") if not promoted else None,
            "updated_at": self.clock()}
        self._event("checkpoint", summary, evidence_ids=checkpoint["evidence_ids"])
        self._transition(campaign, token, CampaignStage.OBSERVE, checkpoint,
                         active_experiment_id=None,
                         incumbent_version=live.version if live else campaign.get("incumbent_version"),
                         incumbent_key=live.key() if live else campaign.get("incumbent_key"))
        return True

    def _policy(self):
        row = self.database.policies.find_one({"_id": self.campaign_id}) or {}
        return OptimizationPolicy(**{k: v for k, v in row.items()
                                    if k in OptimizationPolicy.model_fields})

    def _live_arch(self):
        row = self.database.architectures.find_one({"status": "live"}, sort=[("version", -1)])
        return Arch(**_plain(row)) if row else None

    def _evidence_exists(self, evidence_id: str) -> bool:
        for collection, field in (
            ("metric_windows", "window_id"), ("lessons", "lesson_id"),
            ("summaries", "summary_id"), ("experiments", "experiment_id"),
            ("trials", "trial_id"), ("docs", "doc_id"),
        ):
            if self.database[collection].find_one({field: evidence_id}, {"_id": 1}):
                return True
        return False

    def _event(self, kind, message, **data):
        self.database.events.insert_one({"ts": self.clock(), "kind": kind, "msg": message,
                                         "campaign_id": self.campaign_id, **data})
        print(f"[{kind}] {message}", flush=True)
