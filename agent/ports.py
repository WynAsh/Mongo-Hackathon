"""Typed boundaries between reasoning and deterministic infrastructure.

Implementations may use MongoDB, Strands, or the simulator, but the workflow
depends only on these ports. Mutation-capable operations are deliberately not
exposed to the architect model.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from common.contracts import (
    Arch,
    Campaign,
    CampaignCheckpoint,
    ContextManifest,
    Evaluation,
    ExperimentProposal,
    LessonRecord,
    ObservationSnapshot,
    OptimizationPolicy,
    PromotionResult,
    ReplayPlan,
    TrialResult,
)


@runtime_checkable
class ObservationPort(Protocol):
    async def observe(self, campaign: Campaign) -> ObservationSnapshot:
        """Return the current architecture, regime, and compact metrics."""
        ...


@runtime_checkable
class ArchitectPort(Protocol):
    async def propose(self, context: ContextManifest) -> Sequence[ExperimentProposal]:
        """Produce typed proposals; never deploy or mutate durable memory."""
        ...


@runtime_checkable
class MemoryPort(Protocol):
    async def compile_context(
        self,
        campaign: Campaign,
        policy: OptimizationPolicy,
        observation: ObservationSnapshot,
    ) -> ContextManifest:
        ...

    async def record_lesson(self, lesson: LessonRecord) -> None:
        ...

    async def save_checkpoint(self, campaign: Campaign, checkpoint: CampaignCheckpoint) -> Campaign:
        ...


@runtime_checkable
class ExperimentPort(Protocol):
    async def prepare_replay(
        self,
        campaign: Campaign,
        proposal: ExperimentProposal,
        observation: ObservationSnapshot,
    ) -> ReplayPlan:
        ...

    async def run_trial(
        self,
        experiment_id: str,
        architecture: Arch,
        replay: ReplayPlan,
        repeat: int,
        execution_order: int,
    ) -> TrialResult:
        ...

    async def evaluate(
        self,
        experiment_id: str,
        incumbent_trials: Sequence[TrialResult],
        candidate_trials: Sequence[TrialResult],
        policy: OptimizationPolicy,
        incumbent_version: int,
    ) -> Evaluation:
        ...


@runtime_checkable
class PromotionPort(Protocol):
    async def promote(
        self,
        candidate: Arch,
        evaluation: Evaluation,
        expected_incumbent_version: int,
    ) -> PromotionResult:
        ...

    async def verify(self, architecture: Arch, policy: OptimizationPolicy) -> bool:
        ...

    async def rollback(self, previous: Arch, failed: Arch, reason: str) -> PromotionResult:
        ...
