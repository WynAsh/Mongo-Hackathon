from __future__ import annotations

import mongomock

from agent.experiments import ExperimentRunner
from agent.promotion import PromotionService
from agent.workflow import DurableOptimizationWorkflow
from common.contracts import (
    BASELINE_ARCH, Arch, ExperimentProposal, ExperimentTestPlan, OptimizationPolicy, Pool,
)


class FixedArchitect:
    def propose(self, packet, *, current, policy):
        evidence = next(item["item_id"] for item in packet["context"]
                        if item["memory_type"] == "observation")
        return ExperimentProposal(
            campaign_id="test", hypothesis="one T4 is sufficient",
            candidate=Arch(status="candidate", created_by="agent",
                           pools={"shared": Pool(gpu="t4", replicas=1)},
                           reason="lower cost"),
            evidence_ids=[evidence],
            test_plan=ExperimentTestPlan(duration_s=1, repeats=1,
                                         min_requests=1, warmup_requests=0),
        )


def observation(_seconds, _version):
    return {
        "n": 30, "errors": 0, "rps": 10.0, "stable": True,
        "p50_ms": 800, "p95_ms": 1000,
        "regime_vector": [0.5, 0.2, 0.2, 0.0],
        "prompt_tokens_sample": [50] * 30,
    }


def execute(arch, events, _slot):
    latency = 100.0 if arch.usd_hr() < 1 else 110.0
    return [{"event_index": i, "latency_ms": latency, "error": None}
            for i, _event in enumerate(events)]


def make_workflow(database):
    return DurableOptimizationWorkflow(
        database, campaign_id="test", observer=observation,
        architect=FixedArchitect(),
        runner=ExperimentRunner(database, executor=execute, seed=7),
        promotion=PromotionService(database, verifier=lambda *_: True),
    )


class CrashAfterPromotion(PromotionService):
    def promote_sync(self, *args, **kwargs):
        super().promote_sync(*args, **kwargs)
        raise RuntimeError("simulated crash after architecture CAS")


def seed(database):
    database.architectures.insert_one(BASELINE_ARCH.model_dump())
    policy = OptimizationPolicy(
        policy_id="test", slo_p95_ms=4000, min_trial_requests=1,
        trial_repeats=1, post_promotion_verify_s=1,
    ).model_dump()
    database.policies.insert_one({"_id": "test", **policy})


def test_workflow_resumes_without_duplicate_trial_or_budget_debit():
    database = mongomock.MongoClient().db
    seed(database)
    workflow = make_workflow(database)
    for _ in range(5):
        assert workflow.tick()
    assert database.campaigns.find_one({"_id": "test"})["stage"] == "RUN_TRIALS"

    # Reconstructing the controller simulates a process restart.
    workflow = make_workflow(database)
    for _ in range(10):
        workflow.tick()
        if database.campaigns.find_one({"_id": "test"})["stage"] == "OBSERVE":
            break

    campaign = database.campaigns.find_one({"_id": "test"})
    assert campaign["stage"] == "OBSERVE"
    assert campaign["experiments_spent"] == 1
    assert database.experiments.count_documents({}) == 1
    assert database.trials.count_documents({}) == 2
    assert database.lessons.count_documents({}) == 1
    assert database.architectures.find_one({"status": "live"})["version"] == 2


def test_identical_terminal_experiment_is_not_recharged():
    database = mongomock.MongoClient().db
    seed(database)
    workflow = make_workflow(database)
    for _ in range(15):
        workflow.tick()
        campaign = database.campaigns.find_one({"_id": "test"})
        if campaign["stage"] == "OBSERVE" and campaign["revision"]:
            break
    spent = database.campaigns.find_one({"_id": "test"})["experiments_spent"]
    # The promoted architecture no longer triggers over-provisioning at $0.50/hr.
    assert workflow.tick() is False
    assert database.campaigns.find_one({"_id": "test"})["experiments_spent"] == spent


def test_resume_recovers_promotion_completed_before_checkpoint():
    database = mongomock.MongoClient().db
    seed(database)
    workflow = make_workflow(database)
    for _ in range(7):
        assert workflow.tick()
    assert database.campaigns.find_one({"_id": "test"})["stage"] == "PROMOTE"
    workflow.promotion = CrashAfterPromotion(database, verifier=lambda *_: True)
    try:
        workflow.tick()
    except RuntimeError:
        pass
    assert database.architectures.find_one({"status": "live"})["version"] == 2
    assert database.campaigns.find_one({"_id": "test"})["stage"] == "PROMOTE"

    resumed = make_workflow(database)
    assert resumed.tick()
    assert database.campaigns.find_one({"_id": "test"})["stage"] == "VERIFY_LIVE"
    assert resumed.tick()


def test_learning_retry_does_not_double_count_bandit_result():
    database = mongomock.MongoClient().db
    seed(database)
    workflow = make_workflow(database)
    for _ in range(9):
        assert workflow.tick()
    assert database.campaigns.find_one({"_id": "test"})["stage"] == "LEARN"
    original = workflow.store.transition
    failed = {"once": False}

    def fail_after_learning(*args, **kwargs):
        if kwargs.get("stage") == "CHECKPOINT" and not failed["once"]:
            failed["once"] = True
            raise RuntimeError("simulated crash before learn checkpoint")
        return original(*args, **kwargs)

    workflow.store.transition = fail_after_learning
    try:
        workflow.tick()
    except RuntimeError:
        pass
    assert database.regimes.find_one()["wins"] == 1

    resumed = make_workflow(database)
    assert resumed.tick()
    assert database.regimes.find_one()["wins"] == 1
