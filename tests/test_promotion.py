import mongomock

from agent.promotion import PromotionService
from common.contracts import (
    Arch, Evaluation, EvaluationDecision, EvaluationGate, Pool, OptimizationPolicy,
)


def seed_database():
    database = mongomock.MongoClient().db
    live = Arch(version=1, status="live", pools={"shared": Pool(gpu="t4", replicas=2)})
    database.architectures.insert_one(live.model_dump())
    return database, live


def approved():
    return Evaluation(evaluation_id="ev1", experiment_id="e1", decision=EvaluationDecision.PROMOTE,
                      gates=[EvaluationGate(name="all", passed=True)], incumbent_version=1)


def test_compare_and_swap_promotion_and_rollback_create_new_versions():
    database, previous = seed_database()
    service = PromotionService(database, verifier=lambda *_: True)
    candidate = Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)})
    promoted = service.promote_sync(candidate, approved(), 1)
    assert promoted.promoted and promoted.architecture.version == 2
    assert database.architectures.find_one({"status": "live"})["version"] == 2
    rolled_back = service.rollback_sync(previous, promoted.architecture, "verification failed")
    assert rolled_back.promoted and rolled_back.architecture.version == 3
    assert rolled_back.architecture.key() == previous.key()
    assert database.architectures.count_documents({"status": "live"}) == 1
    assert database.architectures.count_documents({"version": 1, "status": "retired"}) == 1


def test_stale_incumbent_is_rejected_without_mutating_live_architecture():
    database, _ = seed_database()
    database.architectures.update_one({"version": 1}, {"$set": {"version": 2}})
    service = PromotionService(database)
    result = service.promote_sync(
        Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)}), approved(), 1)
    assert not result.promoted
    assert "stale incumbent" in result.reason
    assert database.architectures.find_one({"status": "live"})["version"] == 2


def test_verification_failure_can_rollback_to_previous_setup():
    database, previous = seed_database()
    service = PromotionService(database, verifier=lambda *_: False)
    promoted = service.promote_sync(
        Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)}), approved(), 1)
    outcome = service.verify_and_rollback_sync(promoted.architecture, previous,
                                                OptimizationPolicy(post_promotion_verify_s=1))
    assert outcome.promoted and outcome.architecture.version == 3
    assert "rolled back" in outcome.reason


def test_verification_requires_stable_metric_window():
    database, _ = seed_database()
    service = PromotionService(database, verifier=lambda *_: {
        "n": 100, "errors": 0, "p95_ms": 100, "stable": False,
    })
    assert not service.verify_sync(
        Arch(version=2, status="live", pools={"shared": Pool(gpu="t4", replicas=1)}),
        OptimizationPolicy(min_trial_requests=30, post_promotion_verify_s=1),
    )


def test_promotion_requires_approved_evaluation():
    database, _ = seed_database()
    service = PromotionService(database)
    rejected = approved().model_copy(update={"decision": EvaluationDecision.REJECT})
    result = service.promote_sync(
        Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)}), rejected, 1)
    assert not result.promoted
    assert database.architectures.count_documents({"status": "live"}) == 1
