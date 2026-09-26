"""Deterministic promotion gates and paired bootstrap latency comparison."""
from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from common.contracts import (
    Evaluation, EvaluationDecision, EvaluationGate, OptimizationPolicy,
    TrialResult, TrialRole,
)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else float("inf")


def _paired_latency_regression_upper(
    incumbent_trials: Sequence[TrialResult], candidate_trials: Sequence[TrialResult],
    *, confidence: float, seed_text: str, samples: int = 2000,
) -> float:
    """Upper percentile bootstrap bound for candidate-vs-incumbent p95 regression.

    Pairing is by repeat and replay event index, so each bootstrap sample preserves
    the same traffic demand for both architectures.
    """
    incumbent_by_repeat = {trial.repeat: trial for trial in incumbent_trials}
    candidate_by_repeat = {trial.repeat: trial for trial in candidate_trials}
    pairs: list[tuple[float, float]] = []
    for repeat in sorted(set(incumbent_by_repeat) & set(candidate_by_repeat)):
        incumbent = {outcome.event_index: outcome.latency_ms
                     for outcome in incumbent_by_repeat[repeat].outcomes
                     if outcome.latency_ms is not None and outcome.event_index >= incumbent_by_repeat[repeat].warmup_excluded}
        candidate = {outcome.event_index: outcome.latency_ms
                     for outcome in candidate_by_repeat[repeat].outcomes
                     if outcome.latency_ms is not None and outcome.event_index >= candidate_by_repeat[repeat].warmup_excluded}
        pairs.extend((float(incumbent[index]), float(candidate[index]))
                     for index in sorted(incumbent.keys() & candidate.keys()))
    if not pairs:
        return float("inf")
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    regressions = []
    for _ in range(samples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        incumbent_p95 = _p95([item[0] for item in draw])
        candidate_p95 = _p95([item[1] for item in draw])
        regressions.append(candidate_p95 / max(incumbent_p95, 1e-9) - 1)
    regressions.sort()
    index = min(len(regressions) - 1, int(confidence * len(regressions)))
    return regressions[index]


class Evaluator:
    def evaluate_sync(self, experiment_id: str, incumbent_trials: Sequence[TrialResult],
                      candidate_trials: Sequence[TrialResult], policy: OptimizationPolicy,
                      incumbent_version: int) -> Evaluation:
        gates: list[EvaluationGate] = []
        reasons: list[str] = []
        repeats_ok = len(incumbent_trials) == policy.trial_repeats and len(candidate_trials) == policy.trial_repeats
        gates.append(EvaluationGate(name="repeat_count", passed=repeats_ok,
                                    detail=f"incumbent={len(incumbent_trials)}, candidate={len(candidate_trials)}, required={policy.trial_repeats}"))
        if not repeats_ok:
            reasons.append("required repeat set is incomplete")

        min_samples = all(trial.n >= policy.min_trial_requests
                          for trial in [*incumbent_trials, *candidate_trials])
        gates.append(EvaluationGate(name="minimum_samples", passed=min_samples,
                                    detail=f"minimum {policy.min_trial_requests} requests in every repeat"))
        if not min_samples:
            reasons.append("one or more repeats has too few requests")

        candidate_error_ok = bool(candidate_trials) and all(trial.error_rate <= policy.max_error_rate
                                                             for trial in candidate_trials)
        gates.append(EvaluationGate(name="candidate_error_rate", passed=candidate_error_ok,
                                    detail=f"maximum permitted rate {policy.max_error_rate:.3f}"))
        if not candidate_error_ok:
            reasons.append("candidate error rate exceeded policy")

        candidate_slo_ok = bool(candidate_trials) and all(trial.p95_ms <= policy.slo_p95_ms
                                                          for trial in candidate_trials)
        gates.append(EvaluationGate(name="candidate_slo", passed=candidate_slo_ok,
                                    detail=f"p95 <= {policy.slo_p95_ms:g} ms in every candidate repeat"))
        if not candidate_slo_ok:
            reasons.append("candidate failed the p95 SLO in at least one repeat")

        incumbent_slo_ok = bool(incumbent_trials) and all(
            trial.p95_ms <= policy.slo_p95_ms and trial.error_rate <= policy.max_error_rate
            for trial in incumbent_trials)
        if incumbent_trials and not incumbent_slo_ok:
            reasons.append("incumbent violates SLO; candidate eligibility uses the hard candidate gates")

        cost_improvement = None
        latency_upper = None
        if incumbent_trials and candidate_trials:
            incumbent_cost = sum(t.usd_hr for t in incumbent_trials) / len(incumbent_trials)
            candidate_cost = sum(t.usd_hr for t in candidate_trials) / len(candidate_trials)
            cost_improvement = 1 - candidate_cost / max(incumbent_cost, 1e-9)
            latency_upper = _paired_latency_regression_upper(
                incumbent_trials, candidate_trials, confidence=policy.confidence_level, seed_text=experiment_id)
        cost_ok = cost_improvement is not None and cost_improvement >= policy.min_cost_improvement
        gates.append(EvaluationGate(name="cost_improvement", passed=(cost_ok if incumbent_slo_ok else True),
                                    detail=(f"improvement={cost_improvement:.1%}; required={policy.min_cost_improvement:.1%}"
                                            if cost_improvement is not None else "missing cost comparison")))
        if incumbent_slo_ok and not cost_ok:
            reasons.append("candidate did not meet the required cost improvement")

        latency_ok = latency_upper is not None and latency_upper <= policy.max_latency_regression
        gates.append(EvaluationGate(name="latency_regression_upper_bound",
                                    passed=(latency_ok if incumbent_slo_ok else True),
                                    detail=(f"upper {policy.confidence_level:.0%} bound={latency_upper:.1%}; "
                                            f"maximum={policy.max_latency_regression:.1%}"
                                            if latency_upper is not None else "no paired successful outcomes")))
        if incumbent_slo_ok and not latency_ok:
            reasons.append("upper confidence bound for latency regression exceeded policy")

        hard_ok = repeats_ok and min_samples and candidate_error_ok and candidate_slo_ok
        comparative_ok = not incumbent_slo_ok or (cost_ok and latency_ok)
        decision = EvaluationDecision.PROMOTE if hard_ok and comparative_ok else EvaluationDecision.REJECT
        if decision == EvaluationDecision.PROMOTE:
            reasons.append("all deterministic promotion gates passed")
        return Evaluation(
            evaluation_id=f"evaluation-{experiment_id}", experiment_id=experiment_id,
            decision=decision, gates=gates, reasons=reasons, cost_improvement=cost_improvement,
            latency_regression=(sum(t.p95_ms for t in candidate_trials) / len(candidate_trials) /
                                max(sum(t.p95_ms for t in incumbent_trials) / len(incumbent_trials), 1e-9) - 1
                                if incumbent_trials and candidate_trials else None),
            latency_regression_upper_bound=latency_upper, confidence_level=policy.confidence_level,
            incumbent_version=incumbent_version,
        )


def evaluate(experiment_id: str, incumbent_trials: Sequence[TrialResult],
             candidate_trials: Sequence[TrialResult], policy: OptimizationPolicy,
             incumbent_version: int) -> Evaluation:
    return Evaluator().evaluate_sync(experiment_id, incumbent_trials, candidate_trials, policy, incumbent_version)


async def _async_evaluate(self, experiment_id, incumbent_trials, candidate_trials, policy, incumbent_version):
    return self.evaluate_sync(experiment_id, incumbent_trials, candidate_trials, policy, incumbent_version)


Evaluator.evaluate = _async_evaluate
Evaluator.evaluate_async = _async_evaluate
