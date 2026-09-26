import json

from production.evidence import diagnose, fixtures, normalize_evidence, verify


def test_fixtures_are_normalized_with_stable_ids_and_provenance():
    payload = fixtures()["oom"]
    first = normalize_evidence(payload)
    second = normalize_evidence(payload)
    assert first["evidence_id"] == second["evidence_id"]
    assert first["provenance"] == "fixture"
    assert first["environment_id"] == "demo-cluster"
    assert first["plan_revision"] == 7
    assert first["applied_bundle_id"] == "bundle-7"
    assert first["metrics"]["memory_bytes"]["unit"] == "bytes"


def test_prometheus_api_matrix_preserves_counter_samples_and_quantiles():
    record = normalize_evidence({
        "format": "prometheus", "environment_id": "prod-a", "plan_revision": 3,
        "data": {"resultType": "matrix", "result": [
            {"metric": {"__name__": "container_oom_events_total", "pod": "p1"},
             "values": [[1, "0"], [2, "1"]]},
            {"metric": {"__name__": "http_request_duration_ms", "quantile": "0.95"},
             "value": [2, "120"]},
            {"metric": {"__name__": "http_request_duration_ms", "quantile": "0.99"},
             "value": [2, "240"]},
        ]},
    })
    samples = record["metrics"]["oom_kills"]["samples"]
    assert [sample["value"] for sample in samples] == [0, 1]
    # Quantile-labelled observations remain individual samples, never averaged.
    quantiles = record["metrics"]["http_request_duration_ms"]["samples"]
    assert [sample["labels"]["quantile"] for sample in quantiles] == ["0.95", "0.99"]
    assert [sample["value"] for sample in quantiles] == [120, 240]


def test_secret_redaction_applies_to_structured_data_and_log_excerpts():
    record = normalize_evidence({
        "format": "logs", "data": {"password": "dont-save-me", "note": "authorization: Bearer abc123"},
        "logs": "gateway upstream timeout token=super-secret", "sources": ["test"],
    })
    encoded = json.dumps(record)
    assert "dont-save-me" not in encoded
    assert "abc123" not in encoded
    assert "super-secret" not in encoded
    assert "[REDACTED]" in encoded
    assert record["events"][0]["type"] == "gateway_failure"


def test_unknown_metrics_and_units_are_retained_and_input_is_bounded():
    payload = {"format": "normalized", "data": {"vendor_custom_metric": {"value": 3, "unit": "widgets"}},
               "logs": "x" * 1_100_000}
    record = normalize_evidence(payload)
    assert record["metrics"]["vendor_custom_metric"] == {"value": 3, "unit": "widgets"}
    assert any("size limit" in warning for warning in record["warnings"])
    assert len(json.dumps(record)) < 100_000


def test_diagnosis_proposes_conservative_overload_patch_and_waits_for_missing_metrics():
    plan = {"allocation": {"replicas": 2, "gpus_per_replica": 1},
            "inventory": {"nodes": [{"name": "gpu-node", "gpu_count": 4}]},
            "config": {"max_num_seqs": 8, "max_model_len": 4096}, "workload": {"slo_p95_ms": 1000}}
    good = normalize_evidence({"data": {"queue_depth": 8, "utilization": 0.95, "p95_ms": 1200}})
    result = diagnose(plan, good, "overload")
    assert result["status"] == "remediation_proposed"
    assert result["patch"]["allocation"] == {"replicas": 3}
    assert result["conditions"]["metrics"]["latency_p95_ms"] == {"lte": 1000}
    assert good["evidence_id"] in result["evidence_ids"]

    sparse = normalize_evidence({"data": {"queue_depth": 8}})
    result = diagnose(plan, sparse, "overload")
    assert result["status"] == "awaiting_evidence"
    assert any("utilization samples" in item for item in result["missing_evidence"])
    assert result["patch"] == {}


def test_repair_and_optimize_infer_only_correlated_unhealthy_evidence():
    healthy = normalize_evidence({"data": {"queue_depth": 0, "utilization": 0.45, "p95_ms": 300}})
    plan = {"allocation": {"replicas": 1}, "inventory": {"nodes": [{"gpu_count": 4}]},
            "config": {"max_num_seqs": 8, "max_model_len": 4096}, "workload": {"slo_p95_ms": 1000}}
    for kind in ("repair", "optimize"):
        result = diagnose(plan, healthy, kind)
        assert result["status"] == "awaiting_evidence"
        assert result["patch"] == {}

    broken = normalize_evidence({"format": "logs", "logs": "OOMKilled container", "data": {}})
    result = diagnose(plan, broken, "repair")
    assert result["status"] == "remediation_proposed"
    assert result["patch"] == {"config": {"max_num_seqs": 4}, "allocation": {}}


def test_renderer_supported_oom_and_readiness_fields_and_hardware_bounds():
    plan = {"allocation": {"replicas": 2, "gpus_per_replica": 2},
            "inventory": {"nodes": [{"gpu_count": 4}]},
            "config": {"max_num_seqs": 4, "max_model_len": 2048}}
    oom = normalize_evidence(fixtures()["oom"])
    result = diagnose(plan, oom, "repair")
    assert result["patch"]["config"] == {"max_num_seqs": 2}
    assert set(result["patch"]) == {"config", "allocation"}

    readiness = normalize_evidence(fixtures()["readiness"])
    result = diagnose(plan, readiness, "repair")
    assert result["patch"]["config"] == {"readiness_initial_delay_seconds": 60, "readiness_timeout_seconds": 10}

    overload = normalize_evidence({"data": {"queue_depth": 10, "utilization": 0.99}})
    result = diagnose(plan, overload, "overload")
    assert result["status"] == "awaiting_evidence"  # Two replicas use all four GPUs.
    assert result["patch"] == {}


def test_verify_waits_for_two_counter_samples_then_confirms_or_contradicts():
    conditions = {"window_samples": 2, "coverage_window_seconds": 60,
                  "plan_revision": 7, "metrics": {"restarts": {"delta_eq": 0}, "ready": {"eq": 1}}}
    base = {"observed_at": "2026-09-26T12:20:00Z", "plan_revision": 7,
            "coverage": {"window_seconds": 60}}
    sparse = normalize_evidence({**base, "data": {"restarts": {"samples": [{"value": 4}]},
                                               "ready": {"samples": [{"value": 1}, {"value": 1}]}}})
    assert verify(conditions, sparse)["outcome"] == "awaiting_evidence"

    stable = normalize_evidence({**base, "data": {"restarts": {"samples": [{"value": 4}, {"value": 4}]},
                                                   "ready": {"samples": [{"value": 1}, {"value": 1}]}}})
    assert verify(conditions, stable)["outcome"] == "confirmed"
    failed = normalize_evidence({**base, "data": {"restarts": {"samples": [{"value": 4}, {"value": 5}]},
                                                   "ready": {"samples": [{"value": 1}, {"value": 0}]}}})
    assert verify(conditions, failed)["outcome"] == "contradicted"

    reset = normalize_evidence({**base, "data": {"restarts": {"samples": [{"value": 4}, {"value": 1}]},
                                                  "ready": {"samples": [{"value": 1}, {"value": 1}]}}})
    result = verify(conditions, reset)
    assert result["outcome"] == "awaiting_evidence"
    assert any("counter reset" in item for item in result["missing_evidence"])


def test_verify_requires_coverage_matching_revision_and_freshness_when_requested():
    conditions = {"plan_revision": 4, "coverage_window_seconds": 300, "max_age_seconds": 60,
                  "now": 1_800_000_000, "metrics": {"ready": {"eq": 1}}}
    record = normalize_evidence({"plan_revision": 3, "observed_at": "2020-01-01T00:00:00Z",
                                 "coverage": {"window_seconds": 5}, "data": {"ready": 1}})
    result = verify(conditions, record)
    assert result["outcome"] == "awaiting_evidence"
    assert len(result["missing_evidence"]) >= 3


def test_kubernetes_metrics_api_quantities_and_event_objects_normalize():
    record = normalize_evidence({"format": "kubernetes", "data": {"items": [
        {"kind": "PodMetrics", "metadata": {"name": "worker"}, "containers": [
            {"name": "model", "usage": {"memory": "512Mi", "cpu": "250m"}}]},
        {"kind": "Event", "reason": "OOMKilled", "message": "container out of memory", "count": 2},
    ]}})
    assert record["metrics"]["memory_bytes"]["samples"][0]["value"] == 512 * 1024 * 1024
    assert record["metrics"]["memory_bytes"]["samples"][0]["unit"] == "bytes"
    assert record["metrics"]["cpu_utilization"]["samples"][0]["value"] == 0.25
    assert record["events"][0]["type"] == "oom"


def test_fault_fixtures_extract_readiness_and_gateway_events():
    readiness = normalize_evidence(fixtures()["readiness"])
    gateway = normalize_evidence(fixtures()["gateway"])
    assert {event["type"] for event in readiness["events"]} >= {"readiness_timeout", "restart"}
    assert gateway["events"][0]["type"] == "gateway_failure"
    assert "demo-secret" not in json.dumps(gateway)
