import pytest
from pydantic import ValidationError

from common.contracts import (
    Arch,
    Campaign,
    CampaignStage,
    ContextItem,
    ContextManifest,
    Evaluation,
    EvaluationDecision,
    EvaluationGate,
    ExperimentProposal,
    LessonRecord,
    Pool,
    ReplayEvent,
    ReplayPlan,
    TrialResult,
    TrialRole,
)


def candidate() -> Arch:
    return Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)})


def test_campaign_budget_and_remaining() -> None:
    campaign = Campaign(experiments_spent=2, max_experiments=5)
    assert campaign.experiments_remaining == 3
    with pytest.raises(ValidationError):
        Campaign(experiments_spent=6, max_experiments=5)


def test_proposal_requires_candidate_status() -> None:
    ExperimentProposal(hypothesis="one GPU may suffice", candidate=candidate())
    with pytest.raises(ValidationError):
        ExperimentProposal(
            hypothesis="invalid live proposal",
            candidate=Arch(status="live", pools={"shared": Pool(gpu="t4", replicas=1)}),
        )


def test_replay_events_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        ReplayPlan(
            replay_id="r1",
            seed=7,
            content_hash="abc",
            events=[
                ReplayEvent(offset_ms=10, prompt_tokens=8, output_tokens=8),
                ReplayEvent(offset_ms=0, prompt_tokens=8, output_tokens=8),
            ],
        )


def test_trial_counts_must_be_consistent() -> None:
    with pytest.raises(ValidationError):
        TrialResult(
            trial_id="t1",
            experiment_id="e1",
            replay_id="r1",
            arch_key="a",
            role=TrialRole.CANDIDATE,
            repeat=0,
            execution_order=0,
            n=10,
            p50_ms=10,
            p95_ms=20,
            errors=2,
            error_rate=0.1,
            usd_hr=1,
        )


def test_promotion_requires_all_gates_to_pass() -> None:
    with pytest.raises(ValidationError):
        Evaluation(
            evaluation_id="v1",
            experiment_id="e1",
            decision=EvaluationDecision.PROMOTE,
            gates=[EvaluationGate(name="slo", passed=False)],
            incumbent_version=1,
        )


def test_context_manifest_enforces_token_budget() -> None:
    with pytest.raises(ValidationError):
        ContextManifest(
            manifest_id="m1",
            campaign_revision=0,
            stage=CampaignStage.PROPOSE,
            token_budget=100,
            included=[ContextItem(item_id="policy", memory_type="policy", token_estimate=101)],
        )


def test_lesson_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        LessonRecord(lesson_id="l1", claim="test", evidence_ids=[], confidence=0.5)
