"""Shared contracts. Every module codes against these. Change only as a team."""
from __future__ import annotations

import math
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
