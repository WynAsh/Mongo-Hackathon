from common.contracts import (
    EvaluationDecision, OptimizationPolicy, RequestOutcome, TrialResult, TrialRole,
)
import pytest
from agent.evaluator import Evaluator


def trial(role, repeat, *, latency=100, cost=1, n=100, errors=0, outcomes=True):
    results = [RequestOutcome(event_index=i, latency_ms=latency + (i % 7), error=None)
               for i in range(n)] if outcomes else []
    return TrialResult(trial_id=f"{role}-{repeat}", experiment_id="exp", replay_id="r",
                       arch_key=role, role=role, repeat=repeat, execution_order=repeat % 2,
                       n=n, p50_ms=latency, p95_ms=latency + 6, errors=errors,
                       error_rate=errors / n if n else 0, usd_hr=cost, outcomes=results)


def evaluate(inc, cand, **policy_updates):
    policy = OptimizationPolicy(slo_p95_ms=200, min_trial_requests=30, trial_repeats=3,
                                **policy_updates)
    return Evaluator().evaluate_sync("exp", inc, cand, policy, incumbent_version=1)


def test_promote_when_both_pass_and_cost_and_latency_gates_pass():
    result = evaluate([trial(TrialRole.INCUMBENT, i, cost=2) for i in range(3)],
                      [trial(TrialRole.CANDIDATE, i, latency=95, cost=1.8) for i in range(3)])
    assert result.decision == EvaluationDecision.PROMOTE
    assert result.cost_improvement == pytest.approx(0.1)
    assert result.latency_regression_upper_bound <= 0.1


def test_slo_breach_by_incumbent_allows_safe_candidate_without_cost_gain():
    result = evaluate([trial(TrialRole.INCUMBENT, i, latency=250, cost=2) for i in range(3)],
                      [trial(TrialRole.CANDIDATE, i, latency=150, cost=2) for i in range(3)])
    assert result.decision == EvaluationDecision.PROMOTE


def test_incumbent_error_breach_uses_recovery_rule():
    result = evaluate([trial(TrialRole.INCUMBENT, i, cost=2, errors=20) for i in range(3)],
                      [trial(TrialRole.CANDIDATE, i, latency=150, cost=2) for i in range(3)])
    assert result.decision == EvaluationDecision.PROMOTE


def test_every_candidate_repeat_must_meet_sample_error_and_slo_gates():
    incumbent = [trial(TrialRole.INCUMBENT, i, cost=2) for i in range(3)]
    candidate = [trial(TrialRole.CANDIDATE, i, cost=1.8) for i in range(3)]
    candidate[1] = trial(TrialRole.CANDIDATE, 1, cost=1.8, n=29)
    assert evaluate(incumbent, candidate).decision == EvaluationDecision.REJECT
    candidate[1] = trial(TrialRole.CANDIDATE, 1, cost=1.8, errors=11)
    assert evaluate(incumbent, candidate).decision == EvaluationDecision.REJECT
    candidate[1] = trial(TrialRole.CANDIDATE, 1, latency=210, cost=1.8)
    assert evaluate(incumbent, candidate).decision == EvaluationDecision.REJECT


def test_cost_gate_required_when_incumbent_meets_slo():
    result = evaluate([trial(TrialRole.INCUMBENT, i, cost=2) for i in range(3)],
                      [trial(TrialRole.CANDIDATE, i, cost=1.95) for i in range(3)])
    assert result.decision == EvaluationDecision.REJECT
    assert result.cost_improvement < 0.05


def test_upper_confidence_bound_rejects_latency_regression():
    result = evaluate([trial(TrialRole.INCUMBENT, i, cost=2, latency=100) for i in range(3)],
                      [trial(TrialRole.CANDIDATE, i, cost=1.8, latency=113) for i in range(3)])
    assert result.decision == EvaluationDecision.REJECT
    assert result.latency_regression_upper_bound > 0.10
