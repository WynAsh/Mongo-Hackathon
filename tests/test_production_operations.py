import io
import zipfile

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from production.evidence import fixtures, normalize_evidence
from production.operations import OpsExplanation, analyze, apply_patch, verify_change
from production.planner import build_plan


def _plan(recipe="kserve-llmd-vllm"):
    return build_plan({"inventory": {"os": "Ubuntu 24.04", "kubernetes_version": "1.32",
                                     "nodes": [{"name": "gpu-1", "gpu_model": "H100", "gpu_count": 4,
                                                "vram_gb": 80, "interconnect": "NVLink"}]},
                       "workload": {"task": "chat", "allowed_licenses": ["Apache-2.0"]}, "recipe_id": recipe})


def _ev(name):
    return normalize_evidence(fixtures()[name])


@pytest.mark.parametrize("role,fixture,fault,field,after", [
    ("performance", "overload", "overload", "max_num_seqs", 8),       # every GPU used: KV headroom instead
    ("performance", "underutilized", "underutilized", "replicas", 3),
    ("reliability", "oom", "oom", "max_num_seqs", 2),
    ("reliability", "readiness", "readiness_timeout", "readiness_initial_delay_seconds", 60),
    ("reliability", "slow-startup", "slow_startup", "startup_probe_failure_threshold", 30),
    ("reliability", "gateway", "gateway_failure", "upstream_timeout_seconds", 30),
])
def test_fixtures_produce_validated_change_bundles(role, fixture, fault, field, after):
    change = analyze(_plan(), _ev(fixture), role)
    assert change["diagnosis"]["fault"] == fault
    assert change["status"] == "offline_validated"
    assert {c["field"]: c["after"] for c in change["changes"]}[field] == after
    assert all(c["ok"] for c in change["validation"])
    assert change["diff"].startswith("--- a/") and "CHANGE.md" in change["files"]
    for path, text in change["files"].items():
        if path.endswith(".yaml"):
            yaml.safe_load(text)


def test_spare_gpu_scales_out_before_raising_batch_size():
    base = _plan()
    base["plan"]["allocation"]["replicas"] = 2
    change = analyze(base, _ev("overload"), "performance")
    assert change["changes"] == [{"field": "replicas", "before": 2, "after": 3}]
    assert "replicas: 3" in change["files"]["k8s/30-model.yaml"]


def test_performance_hands_incidents_to_reliability_and_ignores_healthy_evidence():
    handoff = analyze(_plan(), _ev("oom"), "performance")
    assert handoff["status"] == "handoff" and handoff["changes"] == [] and handoff["diff"] == ""
    healthy = normalize_evidence({"data": {"queue_depth": 0, "utilization": 0.5, "p95_ms": 300}})
    for role in ("performance", "reliability"):
        result = analyze(_plan(), healthy, role)
        assert result["status"] == "awaiting_evidence" and result["changes"] == []


def test_reliability_never_proposes_scale_down():
    assert analyze(_plan(), _ev("underutilized"), "reliability")["status"] == "awaiting_evidence"


def test_dynamo_blocks_fields_its_renderer_does_not_support():
    for fixture in ("readiness", "gateway"):
        change = analyze(_plan("dynamo-vllm"), _ev(fixture), "reliability")
        assert change["status"] == "blocked" and change["blockers"] and change["changes"] == []
    assert analyze(_plan("dynamo-sglang"), _ev("oom"), "reliability")["status"] == "offline_validated"


def test_patch_rejects_unsupported_fields_and_out_of_bounds_values():
    p = _plan()["plan"]
    with pytest.raises(ValueError):
        apply_patch(p, {"config": {"image": 1}})
    with pytest.raises(ValueError):
        apply_patch(p, {"config": {"retry_max_attempts": 3}})
    assert apply_patch(p, {"allocation": {"replicas": 2}})["allocation"]["replicas"] == 2
    assert p["allocation"]["replicas"] == 4  # the base plan is never mutated


def test_verification_confirms_or_contradicts_and_needs_samples():
    change = analyze(_plan(), _ev("overload"), "reliability")
    assert verify_change(change, _ev("overload-recovered"))["outcome"] == "confirmed"
    assert verify_change(change, _ev("overload-persists"))["outcome"] == "contradicted"
    sparse = verify_change(change, normalize_evidence({"data": {"queue_depth": 0}}))
    assert sparse["outcome"] == "awaiting_evidence" and not sparse["operationally_verified"]
    fixture_confirmed = verify_change(change, _ev("overload-recovered"))
    assert not fixture_confirmed["operationally_verified"]  # fixtures are simulated evidence


def test_changes_chain_on_earlier_changes():
    first = analyze(_plan(), _ev("slow-startup"), "reliability")
    second = analyze(first, _ev("overload"), "performance")
    assert second["base_id"] == first["change_id"]
    assert second["plan"]["probes"] == first["plan"]["probes"]
    assert "failureThreshold: 30" in second["files"]["k8s/30-model.yaml"]


def test_llm_explains_but_cannot_change_the_patch():
    def advisor(packet):
        assert packet["role"] == "Reliability Engineer" and "excerpts" in packet["evidence"]
        return OpsExplanation(summary="OOMKilled twice at 95% of the limit.", hypotheses=["memory leak"])
    change = analyze(_plan(), _ev("oom"), "reliability", advisor=advisor)
    assert change["explanation"]["summary"].startswith("OOMKilled")
    assert change["explanation"]["decided_by"].startswith("llm:")
    assert change["changes"] == analyze(_plan(), _ev("oom"), "reliability")["changes"]
    broken = analyze(_plan(), _ev("oom"), "reliability", advisor=lambda packet: 1 / 0)
    assert broken["explanation"]["decided_by"] == "rules" and "ZeroDivisionError" in broken["explanation"]["fallback_reason"]


def test_api_imports_analyzes_verifies_without_duplicates(monkeypatch):
    import mongomock
    from production import ops_api
    database = mongomock.MongoClient().db
    monkeypatch.setattr(ops_api.db, "db", lambda: database)
    monkeypatch.setattr(ops_api, "llm_ops_advisor", lambda packet: 1 / 0)
    base = _plan()
    database.production_plans.insert_one(dict(base))
    app = FastAPI()
    ops_api.install(app)
    client = TestClient(app)

    fx = client.get("/api/production/ops/fixtures", params={"role": "reliability"}).json()
    ev = client.post("/api/production/ops/evidence", json={"base_id": base["plan_id"], "payload": fx["diagnose"]["oom"]}).json()
    body = {"base_id": base["plan_id"], "evidence_id": ev["evidence_id"]}
    change = client.post("/api/production/ops/reliability/analyze", json=body).json()
    assert change["status"] == "offline_validated"
    assert client.post("/api/production/ops/reliability/analyze", json=body).json()["change_id"] == change["change_id"]
    assert database.production_changes.count_documents({}) == 1

    # Evidence must be imported for the change it verifies.
    cid = change["change_id"]
    assert client.post(f"/api/production/ops/changes/{cid}/verify", json={"evidence_id": ev["evidence_id"]}).status_code == 422
    after = client.post("/api/production/ops/evidence", json={"base_id": cid, "payload": fx["verify"]["oom-recovered"]}).json()
    verified = client.post(f"/api/production/ops/changes/{cid}/verify", json={"evidence_id": after["evidence_id"]}).json()
    assert verified["verifications"][0]["outcome"] == "confirmed"
    assert verified["status"] == "offline_validated"  # the proposal itself is never rewritten

    logs = client.post("/api/production/ops/evidence", json={"base_id": base["plan_id"], "payload": "token=abc upstream timeout 504"}).json()
    assert "abc" not in str(logs["excerpts"]) and logs["events"][0]["type"] == "gateway_failure"

    download = client.get(f"/api/production/ops/changes/{cid}/download")
    names = zipfile.ZipFile(io.BytesIO(download.content)).namelist()
    assert f"{cid}/change.diff" in names and f"{cid}/k8s/30-model.yaml" in names
    assert client.get("/api/production/ops/changes", params={"role": "performance"}).json() == []
