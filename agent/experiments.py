"""Deterministic replay creation and resumable shadow trials."""
from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Callable
from uuid import uuid4

from common import config, db as db_module
from common.contracts import (
    Arch, Campaign, ExperimentProposal, ObservationSnapshot, ReplayEvent,
    ReplayPlan, RequestOutcome, TrafficProfile, TrialResult, TrialRole,
)
from gateway.metrics import profile_from_window
from infra import shadow


def make_replay_plan(
    profile: TrafficProfile,
    duration_s: float,
    *,
    seed: int,
    campaign_id: str = "default",
    experiment_id: str | None = None,
) -> ReplayPlan:
    """Generate a stable Poisson-arrival replay and hash its immutable payload."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if profile.rps <= 0 or not profile.prompt_tokens:
        raise ValueError("profile requires positive rps and at least one prompt sample")
    rng = random.Random(seed)
    events = []
    elapsed = 0.0
    while True:
        elapsed += rng.expovariate(profile.rps)
        if elapsed > duration_s:
            break
        events.append(ReplayEvent(
            offset_ms=round(elapsed * 1000),
            prompt_tokens=int(rng.choice(profile.prompt_tokens)),
            output_tokens=int(profile.output_tokens),
        ))
    if not events:
        events.append(ReplayEvent(offset_ms=0, prompt_tokens=int(rng.choice(profile.prompt_tokens)),
                                  output_tokens=int(profile.output_tokens)))
    payload = [event.model_dump() for event in events]
    content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scope = hashlib.sha256(f"{campaign_id}:{experiment_id or ''}".encode()).hexdigest()[:10]
    return ReplayPlan(replay_id=f"replay-{scope}-{content_hash[:16]}", campaign_id=campaign_id,
                      experiment_id=experiment_id, seed=seed, content_hash=content_hash, events=events)


def _profile_from_observation(observation: ObservationSnapshot) -> TrafficProfile:
    metrics = observation.metrics
    if isinstance(metrics.get("prompt_tokens_sample"), list):
        return profile_from_window({"rps": metrics.get("rps", 1),
                                    "prompt_tokens_sample": metrics["prompt_tokens_sample"]})
    # A compact observation may contain only regime quantiles. Reconstruct a
    # deterministic empirical sample for the first integration iteration.
    rps = max(float(metrics.get("rps", observation.regime_vector[0] * 20 or 0.5)), 0.5)
    p50 = max(1, int(metrics.get("prompt_p50_tokens", 100)))
    p90 = max(p50, int(metrics.get("prompt_p90_tokens", p50)))
    long_fraction = min(max(float(observation.regime_vector[3]), 0), 1)
    prompts = [p90 if index < round(long_fraction * 100) else p50 for index in range(100)]
    return TrafficProfile(rps=rps, prompt_tokens=prompts)


class ExperimentRunner:
    """Persists replay plans and trials; a repeated call reuses completed trial IDs."""

    def __init__(self, database=None, *, executor: Callable | None = None, seed: int | None = None):
        self.database = database if database is not None else db_module.db()
        self.replays = self.database["replay_plans"]
        self.trials = self.database["trials"]
        self.executor = executor or shadow.run_replay
        self.seed = seed
        self.trials.create_index("trial_id", unique=True)
        self.replays.create_index("replay_id", unique=True)

    def prepare_replay_sync(self, campaign: Campaign, proposal: ExperimentProposal,
                            observation: ObservationSnapshot, profile: TrafficProfile | None = None) -> ReplayPlan:
        experiment_id = proposal.proposal_id or uuid4().hex
        seed = self.seed if self.seed is not None else int(hashlib.sha256(
            f"{campaign.campaign_id}:{experiment_id}".encode()).hexdigest()[:8], 16)
        plan = make_replay_plan(profile or _profile_from_observation(observation), proposal.test_plan.duration_s,
                                seed=seed, campaign_id=campaign.campaign_id, experiment_id=experiment_id)
        self.replays.update_one({"replay_id": plan.replay_id}, {"$setOnInsert": plan.model_dump()}, upsert=True)
        saved = self.replays.find_one({"replay_id": plan.replay_id})
        return ReplayPlan(**{key: value for key, value in saved.items() if key != "_id"})

    def run_trial_sync(self, experiment_id: str, architecture: Arch, replay: ReplayPlan,
                       repeat: int, execution_order: int, *, role: TrialRole | str = TrialRole.CANDIDATE,
                       warmup_requests: int = 0) -> TrialResult:
        role = TrialRole(role)
        trial_id = f"{experiment_id}:{replay.replay_id}:{role.value}:{repeat}"
        saved = self.trials.find_one({"trial_id": trial_id})
        if saved:
            return TrialResult(**{key: value for key, value in saved.items() if key != "_id"})
        events = [event.model_dump() for event in replay.events]
        started_at = time.time()
        outcomes = self.executor(architecture, events, 0)
        filtered = [result for result in outcomes if int(result["event_index"]) >= warmup_requests]
        latencies = sorted(float(result["latency_ms"]) for result in filtered if result.get("latency_ms") is not None)
        n = len(filtered)
        errors = n - len(latencies)
        percentile = lambda q: float(latencies[min(len(latencies) - 1, int(len(latencies) * q))]) if latencies else 0.0
        outcome_models = [RequestOutcome(event_index=int(result["event_index"]),
                                         latency_ms=result.get("latency_ms"), error=result.get("error"))
                          for result in outcomes]
        trial = TrialResult(
            trial_id=trial_id, experiment_id=experiment_id, replay_id=replay.replay_id,
            arch_key=architecture.key(), role=role, repeat=repeat, execution_order=execution_order,
            n=n, warmup_excluded=min(warmup_requests, len(replay.events)), p50_ms=percentile(0.5),
            p95_ms=percentile(0.95), errors=errors, error_rate=(errors / n if n else 0),
            usd_hr=architecture.usd_hr(), gpu_seconds=None, outcomes=outcome_models,
            started_at=started_at, finished_at=time.time(),
        )
        self.trials.update_one({"trial_id": trial_id}, {"$setOnInsert": trial.model_dump()}, upsert=True)
        stored = self.trials.find_one({"trial_id": trial_id})
        return TrialResult(**{key: value for key, value in stored.items() if key != "_id"})

    def run_repeated_trials(self, experiment_id: str, incumbent: Arch, candidate: Arch,
                            replay: ReplayPlan, repeats: int, *, warmup_requests: int = 0):
        """Run serially; alternate which side runs first to reduce order bias."""
        incumbent_trials, candidate_trials = [], []
        for repeat in range(repeats):
            ordered = [(TrialRole.INCUMBENT, incumbent), (TrialRole.CANDIDATE, candidate)]
            if repeat % 2:
                ordered.reverse()
            for execution_order, (role, arch) in enumerate(ordered):
                result = self.run_trial_sync(experiment_id, arch, replay, repeat, execution_order,
                                             role=role, warmup_requests=warmup_requests)
                (incumbent_trials if role == TrialRole.INCUMBENT else candidate_trials).append(result)
        return incumbent_trials, candidate_trials

    async def prepare_replay(self, campaign: Campaign, proposal: ExperimentProposal,
                             observation: ObservationSnapshot) -> ReplayPlan:
        import asyncio
        return await asyncio.to_thread(self.prepare_replay_sync, campaign, proposal, observation)

    async def run_trial(self, experiment_id: str, architecture: Arch, replay: ReplayPlan,
                        repeat: int, execution_order: int) -> TrialResult:
        import asyncio
        role = TrialRole.INCUMBENT if architecture.status == "live" else TrialRole.CANDIDATE
        return await asyncio.to_thread(self.run_trial_sync, experiment_id, architecture, replay,
                                       repeat, execution_order, role=role)
