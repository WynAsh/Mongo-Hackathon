"""Shared contracts. Every module codes against these. Change only as a team."""
from __future__ import annotations

import math
import time
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------- GPU profiles
# Flags map 1:1 onto llm-d-inference-sim CLI flags (see infra/deployer.py).
GPU_PROFILES: dict[str, dict] = {
    "t4": {   # cheap, slow prefill, few slots
        "max_num_seqs": 8, "ttft_ms": 400, "prefill_ms_per_tok": 2.0,
        "itl_ms": 30, "load_factor": 3.0, "usd_hr": 0.50,
    },
    "a100": {  # expensive, fast prefill, many slots
        "max_num_seqs": 32, "ttft_ms": 150, "prefill_ms_per_tok": 0.5,
        "itl_ms": 12, "load_factor": 1.5, "usd_hr": 3.00,
    },
}

MAX_REPLICAS_PER_POOL = 4
MAX_TOTAL_REPLICAS = 6


# ---------------------------------------------------------------- architecture
class Pool(BaseModel):
    gpu: Literal["t4", "a100"]
    replicas: int = Field(ge=1, le=MAX_REPLICAS_PER_POOL)


class Arch(BaseModel):
    """The whole serving setup. Stored in `architectures`. The agent may only change these knobs."""
    version: int = 0
    status: Literal["live", "candidate", "retired"] = "candidate"
    split_threshold_tokens: Optional[int] = None      # None = one shared pool named "shared"
    pools: dict[str, Pool]
    reason: str = ""
    created_by: Literal["agent", "human"] = "human"
    created_at: float = Field(default_factory=time.time)

    @field_validator("pools")
    @classmethod
    def _pool_names(cls, v):
        if sum(p.replicas for p in v.values()) > MAX_TOTAL_REPLICAS:
            raise ValueError(f"more than {MAX_TOTAL_REPLICAS} replicas total")
        return v

    def model_post_init(self, _):
        names = set(self.pools)
        if self.split_threshold_tokens is None and names != {"shared"}:
            raise ValueError("no split => exactly one pool named 'shared'")
        if self.split_threshold_tokens is not None and names != {"short", "long"}:
            raise ValueError("split => pools must be exactly 'short' and 'long'")

    def usd_hr(self) -> float:
        return round(sum(GPU_PROFILES[p.gpu]["usd_hr"] * p.replicas for p in self.pools.values()), 2)

    def key(self) -> str:
        """Stable identity used by memory/bandit (ignores version/reason)."""
        parts = [f"split={self.split_threshold_tokens}"]
        for name in sorted(self.pools):
            p = self.pools[name]
            parts.append(f"{name}={p.replicas}x{p.gpu}")
        return "|".join(parts)

    def summary(self) -> str:
        pools = ", ".join(f"{n}: {p.replicas}×{p.gpu}" for n, p in sorted(self.pools.items()))
        split = f"split at {self.split_threshold_tokens} tok" if self.split_threshold_tokens else "no split"
        return f"{split}; {pools}; ${self.usd_hr()}/hr"


BASELINE_ARCH = Arch(
    version=1, status="live", split_threshold_tokens=None,
    pools={"shared": Pool(gpu="t4", replicas=2)},
    reason="initial hand-written setup", created_by="human",
)


# ---------------------------------------------------------------- traffic / regime
class TrafficProfile(BaseModel):
    """What the load generator plays. `prompt_tokens` is sampled uniformly."""
    rps: float
    prompt_tokens: list[int]          # empirical sample to draw from
    output_tokens: int = 8


def regime_vector(rps: float, prompt_tokens: list[int]) -> list[float]:
    """4-dim normalized traffic fingerprint -> `regimes` vector index (numDimensions=4)."""
    if not prompt_tokens:
        return [0.0, 0.0, 0.0, 0.0]
    s = sorted(prompt_tokens)
    p50 = s[len(s) // 2]
    p90 = s[min(len(s) - 1, int(len(s) * 0.9))]
    long_frac = sum(1 for t in s if t >= 500) / len(s)
    return [
        round(min(rps / 20.0, 1.0), 4),
        round(min(math.log1p(p50) / math.log1p(2000), 1.0), 4),
        round(min(math.log1p(p90) / math.log1p(2000), 1.0), 4),
        round(long_frac, 4),
    ]


def describe_regime(vec: list[float]) -> str:
    return f"~{vec[0]*20:.1f} rps, {vec[3]*100:.0f}% long prompts"


# ---------------------------------------------------------------- results
class ShadowResult(BaseModel):
    arch_key: str
    n: int
    p50_ms: float
    p95_ms: float
    errors: int
    usd_hr: float


class Experiment(BaseModel):
    regime_vector: list[float]
    arch: Arch
    result: ShadowResult
    incumbent_key: str
    incumbent_p95_ms: float
    won: bool
    ts: float = Field(default_factory=time.time)


class Lesson(BaseModel):
    text: str
    regime_vector: list[float]
    arch_key: str
    confirmed: int = 1
    contradicted: int = 0

    @property
    def confidence(self) -> float:
        return self.confirmed / (self.confirmed + self.contradicted + 1)


SLO_HEADROOM = 0.6   # a setup only "meets" the SLO in a shadow test if p95 <= 60% of it


def score(result: ShadowResult, slo_ms: float) -> tuple:
    """Lower is better. Meeting the SLO (with headroom) beats everything; then cheaper wins; else faster wins."""
    if result.n == 0 or result.errors > result.n * 0.1:
        return (2, float("inf"), float("inf"))
    if result.p95_ms <= slo_ms * SLO_HEADROOM:
        return (0, result.usd_hr, result.p95_ms)
    return (1, result.p95_ms, result.usd_hr)


# ------------------------------------------------------- long-horizon workflow
# These contracts are additive. The original demo models above remain the wire
# format for the gateway, simulator, UI, and the regime bandit.
class CampaignStage(str, Enum):
    OBSERVE = "OBSERVE"
    BUILD_CONTEXT = "BUILD_CONTEXT"
    PROPOSE = "PROPOSE"
    VALIDATE = "VALIDATE"
    PREPARE_REPLAY = "PREPARE_REPLAY"
    RUN_TRIALS = "RUN_TRIALS"
    EVALUATE = "EVALUATE"
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    VERIFY_LIVE = "VERIFY_LIVE"
    LEARN = "LEARN"
    CHECKPOINT = "CHECKPOINT"
    PAUSED = "PAUSED"


class CampaignLease(BaseModel):
    owner: str = Field(min_length=1)
    token: str = Field(min_length=1)
    until: float


class CampaignCheckpoint(BaseModel):
    """Compact, durable working memory used to resume without chat history."""

    summary: str = ""
    established_facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_work: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)


class Campaign(BaseModel):
    campaign_id: str = Field(default="default", min_length=1)
    goal: str = "Minimize GPU cost while satisfying the latency SLO"
    stage: CampaignStage = CampaignStage.OBSERVE
    incumbent_key: Optional[str] = None
    incumbent_version: Optional[int] = Field(default=None, ge=0)
    active_experiment_id: Optional[str] = None
    experiments_spent: int = Field(default=0, ge=0)
    max_experiments: int = Field(default=20, ge=1)
    revision: int = Field(default=0, ge=0)
    lease: Optional[CampaignLease] = None
    checkpoint: CampaignCheckpoint = Field(default_factory=CampaignCheckpoint)
    last_error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _budget_is_valid(self):
        if self.experiments_spent > self.max_experiments:
            raise ValueError("experiments_spent cannot exceed max_experiments")
        return self

    @property
    def experiments_remaining(self) -> int:
        return self.max_experiments - self.experiments_spent


class OptimizationPolicy(BaseModel):
    policy_id: str = "default"
    slo_p95_ms: float = Field(default=4000.0, gt=0)
    max_error_rate: float = Field(default=0.10, ge=0, le=1)
    min_trial_requests: int = Field(default=30, ge=1)
    trial_repeats: int = Field(default=3, ge=1)
    min_cost_improvement: float = Field(default=0.05, ge=0, lt=1)
    max_latency_regression: float = Field(default=0.10, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    post_promotion_verify_s: int = Field(default=20, ge=1)
    auto_promote: bool = True
    allowed_gpus: list[Literal["t4", "a100"]] = Field(default_factory=lambda: ["t4", "a100"])
    max_total_replicas: int = Field(default=MAX_TOTAL_REPLICAS, ge=1, le=MAX_TOTAL_REPLICAS)


class ExpectedEffect(BaseModel):
    cost_pct: Optional[float] = None
    p95_latency_pct: Optional[float] = None
    throughput_pct: Optional[float] = None


class ExperimentTestPlan(BaseModel):
    duration_s: int = Field(default=20, ge=1)
    repeats: int = Field(default=3, ge=1)
    min_requests: int = Field(default=30, ge=1)
    warmup_requests: int = Field(default=1, ge=0)


class ExperimentProposal(BaseModel):
    proposal_id: Optional[str] = None
    campaign_id: str = Field(default="default", min_length=1)
    hypothesis: str = Field(min_length=1)
    candidate: Arch
    evidence_ids: list[str] = Field(default_factory=list)
    expected_effect: ExpectedEffect = Field(default_factory=ExpectedEffect)
    test_plan: ExperimentTestPlan = Field(default_factory=ExperimentTestPlan)
    stop_conditions: list[str] = Field(default_factory=list)
    rollback_plan: str = "Restore the previous live architecture"
    created_at: float = Field(default_factory=time.time)

    @field_validator("candidate")
    @classmethod
    def _candidate_status(cls, arch: Arch) -> Arch:
        if arch.status != "candidate":
            raise ValueError("proposed architecture must have candidate status")
        return arch


class ReplayEvent(BaseModel):
    offset_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)


class ReplayPlan(BaseModel):
    replay_id: str = Field(min_length=1)
    campaign_id: str = Field(default="default", min_length=1)
    experiment_id: Optional[str] = None
    seed: int
    content_hash: str = Field(min_length=1)
    events: list[ReplayEvent] = Field(min_length=1)
    created_at: float = Field(default_factory=time.time)

    @field_validator("events")
    @classmethod
    def _events_are_ordered(cls, events: list[ReplayEvent]) -> list[ReplayEvent]:
        offsets = [event.offset_ms for event in events]
        if offsets != sorted(offsets):
            raise ValueError("replay events must be ordered by offset_ms")
        return events


class RequestOutcome(BaseModel):
    event_index: int = Field(ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    error: Optional[str] = None


class TrialRole(str, Enum):
    INCUMBENT = "incumbent"
    CANDIDATE = "candidate"


class TrialResult(BaseModel):
    trial_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    replay_id: str = Field(min_length=1)
    arch_key: str = Field(min_length=1)
    role: TrialRole
    repeat: int = Field(ge=0)
    execution_order: int = Field(ge=0)
    n: int = Field(ge=0)
    warmup_excluded: int = Field(default=0, ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    errors: int = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    usd_hr: float = Field(ge=0)
    gpu_seconds: Optional[float] = Field(default=None, ge=0)
    outcomes: list[RequestOutcome] = Field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _counts_are_consistent(self):
        if self.errors > self.n:
            raise ValueError("errors cannot exceed n")
        if self.n and abs(self.error_rate - self.errors / self.n) > 1e-6:
            raise ValueError("error_rate must equal errors / n")
        if not self.n and self.error_rate != 0:
            raise ValueError("error_rate must be zero when n is zero")
        return self


class EvaluationDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class EvaluationGate(BaseModel):
    name: str = Field(min_length=1)
    passed: bool
    detail: str = ""


class Evaluation(BaseModel):
    evaluation_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    decision: EvaluationDecision
    gates: list[EvaluationGate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cost_improvement: Optional[float] = None
    latency_regression: Optional[float] = None
    latency_regression_upper_bound: Optional[float] = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    incumbent_version: int = Field(ge=0)
    created_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _promotion_requires_passing_gates(self):
        if self.decision == EvaluationDecision.PROMOTE and any(not gate.passed for gate in self.gates):
            raise ValueError("promotion requires every evaluation gate to pass")
        return self


class LessonScope(BaseModel):
    model: Optional[str] = None
    gpu: Optional[str] = None
    serving_engine: Optional[str] = None
    serving_engine_version: Optional[str] = None
    regime_hash: Optional[str] = None
    objective: Optional[str] = None


class LessonRecord(BaseModel):
    lesson_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    scope: LessonScope = Field(default_factory=LessonScope)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    contradicts: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    schema_version: int = Field(default=1, ge=1)
    created_at: float = Field(default_factory=time.time)


class ContextItem(BaseModel):
    item_id: str = Field(min_length=1)
    memory_type: Literal["policy", "working", "observation", "episodic", "semantic", "documentation", "raw"]
    reason: str = ""
    version: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    token_estimate: int = Field(default=0, ge=0)
    content: Any = None


class ExcludedContextItem(BaseModel):
    item_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    memory_type: Optional[str] = None
    version: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    token_estimate: int = Field(default=0, ge=0)


class ContextManifest(BaseModel):
    manifest_id: str = Field(min_length=1)
    campaign_id: str = Field(default="default", min_length=1)
    campaign_revision: int = Field(ge=0)
    stage: CampaignStage
    query: str = ""
    token_budget: int = Field(default=12000, ge=1)
    included: list[ContextItem] = Field(default_factory=list)
    excluded: list[ExcludedContextItem] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return sum(item.token_estimate for item in self.included)

    @model_validator(mode="after")
    def _fits_budget(self):
        if self.total_tokens > self.token_budget:
            raise ValueError("included context exceeds token_budget")
        return self


class ObservationSnapshot(BaseModel):
    """Small observation passed across the workflow boundary."""

    architecture: Arch
    regime_vector: list[float]
    # Compact windows also carry the bounded prompt-token sample used to build
    # an immutable replay, so values are not exclusively scalar metrics.
    metrics: dict[str, Any] = Field(default_factory=dict)
    observed_at: float = Field(default_factory=time.time)


class PromotionResult(BaseModel):
    promoted: bool
    architecture: Optional[Arch] = None
    reason: str = ""
    previous_architecture: Optional[Arch] = None
