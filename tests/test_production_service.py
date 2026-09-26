import copy
import io
import zipfile
import mongomock
import pytest
from production.service import PlanningService, apply_patch_to_plan
from production.reasoning import ProductionReasoner, compile_context


def request(recipe="kserve-llmd-vllm"):
    return {"environment_id": "test-env", "recipe_id": recipe,
            "workload": {"task": "general chat", "context_tokens": 4096, "concurrency": 2},
            "inventory": {"os": "Ubuntu 24.04", "driver_version": "580.95.05",
                          "cuda_version": "13.0", "kubernetes_version": "1.34.0",
                          "nodes": [{"name": "gpu-1", "gpu_model": "NVIDIA H100", "gpu_count": 4,
                                     "vram_gb": 80, "ram_gb": 256, "storage_gb": 1000,
                                     "interconnect": "NVLink"}]}}


def finish(service, task_id):
    for _ in range(15):
        service.tick()
        row = service.get(task_id)
        if row["stage"] == "COMPLETE" or row["status"] in {"failed", "retrying"}:
            return row
    raise AssertionError("workflow did not finish")


@pytest.fixture
def service():
    return PlanningService(mongomock.MongoClient().test, ProductionReasoner(enabled=False))


def test_restart_idempotency_and_no_deploy(service, monkeypatch):
    import infra.deployer
    import common.db
    def forbidden(*args, **kwargs):
        raise AssertionError("production planning must not deploy")
    monkeypatch.setattr(infra.deployer, "deploy", forbidden)
    monkeypatch.setattr(common.db, "promote_compare_and_swap", forbidden)
    task = service.create(request())
    assert service.create(request())["task_id"] == task["task_id"]
    for _ in range(4):
        service.tick()
    resumed = PlanningService(service.db, ProductionReasoner(enabled=False))
    completed = finish(resumed, task["task_id"])
    assert completed["stage"] == "COMPLETE", completed
    assert completed["status"] == "offline_validated", completed
    assert not completed["runtime_verified"]
    assert resumed.db.production_bundles.count_documents({}) == 1
    archive = zipfile.ZipFile(io.BytesIO(resumed.bundle_zip(completed["bundle_id"])))
    assert "validation-report.json" in archive.namelist()
    assert any(name.endswith(".yaml") for name in archive.namelist())


def test_environment_cas_rejects_competing_plan(service):
    first = service.create(request())
    second_input = request()
    second_input["idempotency_key"] = "second"
    second = service.create(second_input)
    finish(service, first["task_id"])
    other = finish(service, second["task_id"])
    assert other["status"] == "stale"


def test_memory_budget_and_environment_isolation(service):
    task = service.create(request())
    plan = {"recipe_id": "x", "model": {"revision": "abc"}, "sources": []}
    service.db.production_lessons.insert_one({"environment_id": "other", "lesson_id": "secret"})
    service.db.lessons.insert_one({"lesson_id": "simulator", "text": "untrusted simulator recommendation"})
    packet = compile_context(service.db, task, plan)
    assert all(x["item_id"] not in {"secret", "simulator"} for x in packet["included"])
    assert packet["token_count"] <= packet["token_budget"]
    with pytest.raises(ValueError, match="mandatory context"):
        compile_context(service.db, task, plan, budget=10)


def test_plan_patch_rejects_shell_and_unbounded_changes():
    plan = {"config": {"max_num_seqs": 16}, "allocation": {"replicas": 1}}
    original = copy.deepcopy(plan)
    with pytest.raises(ValueError):
        apply_patch_to_plan(plan, {"config": {"shell": "do things"}})
    with pytest.raises(ValueError):
        apply_patch_to_plan(plan, {"config": {"retry_max_attempts": 10}})
    candidate = apply_patch_to_plan(plan, {"config": {"max_num_seqs": 8}})
    assert plan == original
    assert candidate["config"]["max_num_seqs"] == 8


def test_unrelated_evidence_and_stale_base_rejected(service):
    task = service.create(request())
    base = finish(service, task["task_id"])
    ev = service.import_evidence({"environment_id": "different", "plan_revision": 1,
                                 "format": "normalized", "data": {"queue_depth": 20}})
    with pytest.raises(ValueError, match="environment"):
        service.create({"kind": "repair", "environment_id": "test-env",
                        "base_task_id": base["task_id"], "evidence_id": ev["evidence_id"]})


def _health_samples(value):
    return {"samples": [{"timestamp": index, "value": value} for index in range(3)]}


def test_runtime_verification_requires_matching_bundle_revision_and_coverage(service):
    completed = finish(service, service.create(request())["task_id"])
    evidence = service.import_evidence({
        "environment_id": "test-env",
        "plan_revision": completed["published_revision"],
        "applied_bundle_id": completed["bundle_id"],
        "format": "normalized",
        "coverage": {"window_seconds": 120},
        "data": {
            "ready": _health_samples(1),
            "error_rate": _health_samples(0.01),
            "latency_p95_ms": _health_samples(500),
        },
    })
    result = service.verification(completed["task_id"], evidence["evidence_id"])
    assert result["outcome"] == "confirmed"
    assert service.get(completed["task_id"])["runtime_verified"] is True

    insufficient = service.import_evidence({
        "environment_id": "test-env",
        "plan_revision": completed["published_revision"],
        "applied_bundle_id": completed["bundle_id"],
        "format": "normalized",
        "coverage": {"window_seconds": 10},
        "data": {
            "ready": _health_samples(1),
            "error_rate": _health_samples(0.01),
            "latency_p95_ms": _health_samples(500),
        },
    })
    waiting = service.verification(completed["task_id"], insufficient["evidence_id"])
    assert waiting["outcome"] == "awaiting_evidence"


def test_fixture_can_simulate_confirmation_but_never_runtime_verify(service):
    completed = finish(service, service.create(request())["task_id"])
    fixture = service.import_evidence({
        "environment_id": "test-env",
        "plan_revision": completed["published_revision"],
        "applied_bundle_id": completed["bundle_id"],
        "provenance": "fixture",
        "format": "normalized",
        "coverage": {"window_seconds": 120},
        "data": {
            "ready": _health_samples(1),
            "error_rate": _health_samples(0.01),
            "latency_p95_ms": _health_samples(500),
        },
    })
    result = service.verification(completed["task_id"], fixture["evidence_id"])
    assert result["outcome"] == "confirmed"
    assert result["provenance"] == "fixture"
    assert service.get(completed["task_id"])["runtime_verified"] is False
