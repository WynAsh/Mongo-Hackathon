"""Compare-and-swap promotion and post-promotion verification/rollback."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from common import db as db_module
from common.contracts import (
    Arch, Evaluation, EvaluationDecision, OptimizationPolicy, Pool,
    PromotionResult,
)


class PromotionService:
    def __init__(self, database=None, *, verifier: Callable | None = None):
        self.database = database if database is not None else db_module.db()
        self.verifier = verifier

    def promote_sync(self, candidate: Arch, evaluation: Evaluation,
                     expected_incumbent_version: int) -> PromotionResult:
        if evaluation.decision != EvaluationDecision.PROMOTE:
            return PromotionResult(promoted=False, reason="evaluation did not approve promotion")
        if evaluation.incumbent_version != expected_incumbent_version:
            return PromotionResult(promoted=False, reason="evaluation was produced for a different incumbent version")
        if any(not gate.passed for gate in evaluation.gates):
            return PromotionResult(promoted=False, reason="one or more evaluation gates failed")
        current = self.database.architectures.find_one(
            {"status": "live"}, sort=[("version", -1)])
        if not current or current.get("version") != expected_incumbent_version:
            actual = current.get("version") if current else None
            return PromotionResult(promoted=False, reason=f"stale incumbent: expected v{expected_incumbent_version}, found {actual}")
        previous = Arch(**{key: value for key, value in current.items() if key != "_id"})
        candidate_doc = candidate.model_dump()
        candidate_doc["reason"] = (candidate.reason + f" | evidence-backed evaluation {evaluation.evaluation_id}").strip(" |")
        try:
            promoted_doc = db_module.promote_compare_and_swap(
                candidate_doc, expected_incumbent_version, database=self.database)
        except db_module.StaleArchitectureError as exc:
            return PromotionResult(promoted=False, reason=str(exc), previous_architecture=previous)
        promoted = Arch(**promoted_doc)
        return PromotionResult(promoted=True, architecture=promoted,
                               previous_architecture=previous, reason=f"promoted as v{promoted.version}")

    def verify_sync(self, architecture: Arch, policy: OptimizationPolicy) -> bool:
        """Observe a stable live window or use an injected deployment verifier."""
        if self.verifier is not None:
            result = self.verifier(architecture, policy)
            if isinstance(result, bool):
                return result
            if isinstance(result, dict):
                n = int(result.get("n", 0))
                errors = int(result.get("errors", 0))
                p95 = result.get("p95_ms")
                stable = bool(result.get("stable", False))
            else:
                n = int(getattr(result, "n", 0))
                errors = int(getattr(result, "errors", 0))
                p95 = getattr(result, "p95_ms", None)
                stable = bool(getattr(result, "stable", False))
            return stable and n >= policy.min_trial_requests and p95 is not None and p95 <= policy.slo_p95_ms and errors / max(n, 1) <= policy.max_error_rate

        # The default adapter verifies the actual promoted version from gateway
        # telemetry, waiting for a complete configured observation interval.
        time.sleep(policy.post_promotion_verify_s)
        from gateway.metrics import window
        observation = window(policy.post_promotion_verify_s, architecture.version)
        return (
            observation.get("stable", False)
            and
            observation.get("n", 0) >= policy.min_trial_requests
            and observation.get("p95_ms") is not None
            and observation["p95_ms"] <= policy.slo_p95_ms
            and observation.get("errors", 0) / max(observation.get("n", 0), 1) <= policy.max_error_rate
        )

    def rollback_sync(self, previous: Arch, failed: Arch, reason: str) -> PromotionResult:
        current = self.database.architectures.find_one({"status": "live"}, sort=[("version", -1)])
        if not current or current.get("version") != failed.version:
            actual = current.get("version") if current else None
            return PromotionResult(promoted=False, reason=f"rollback refused: expected failed live v{failed.version}, found {actual}")
        rollback_arch = previous.model_copy(update={
            "version": 0,
            "status": "candidate",
            "reason": f"rollback of v{failed.version}: {reason}",
            "created_by": "agent",
            "created_at": time.time(),
        })
        try:
            doc = db_module.promote_compare_and_swap(
                rollback_arch.model_dump(), failed.version, database=self.database)
        except db_module.StaleArchitectureError as exc:
            return PromotionResult(promoted=False, reason=f"rollback refused: {exc}")
        restored = Arch(**doc)
        return PromotionResult(promoted=True, architecture=restored, previous_architecture=failed,
                               reason=f"rolled back as new version v{restored.version}: {reason}")

    def verify_and_rollback_sync(self, promoted: Arch, previous: Arch,
                                 policy: OptimizationPolicy) -> PromotionResult:
        if self.verify_sync(promoted, policy):
            return PromotionResult(promoted=True, architecture=promoted,
                                   previous_architecture=previous, reason="post-promotion verification passed")
        return self.rollback_sync(previous, promoted, "post-promotion verification failed")

    async def promote(self, candidate: Arch, evaluation: Evaluation,
                      expected_incumbent_version: int) -> PromotionResult:
        return await asyncio.to_thread(self.promote_sync, candidate, evaluation, expected_incumbent_version)

    async def verify(self, architecture: Arch, policy: OptimizationPolicy) -> bool:
        return await asyncio.to_thread(self.verify_sync, architecture, policy)

    async def rollback(self, previous: Arch, failed: Arch, reason: str) -> PromotionResult:
        return await asyncio.to_thread(self.rollback_sync, previous, failed, reason)
