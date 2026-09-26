"""Bounded, deterministic normalization and diagnosis of production evidence.

This module deliberately has no infrastructure clients. It accepts snapshots or
exported text and returns small, redacted, content-addressed evidence records.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

MAX_INPUT_BYTES = 1_000_000
MAX_RECORDS = 2_000
MAX_TEXT = 4_000
MAX_EXCERPTS = 40

_SECRET_PATTERNS = [
    (re.compile(r"(?i)(\b(?:authorization|proxy-authorization)\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(\b(?:password|passwd|token|api[_-]?key|secret|client[_-]?secret)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(://[^:/\s]+:)[^@/\s]+(@)"), r"\1[REDACTED]\2"),
]

_ALIASES = {
    "cpu_utilization": ("cpu_utilization", "cpu_usage", "container_cpu_usage"),
    "memory_bytes": ("memory_bytes", "memory_usage_bytes", "container_memory_working_set_bytes", "container_memory_usage_bytes", "process_resident_memory_bytes"),
    "memory_limit_bytes": ("memory_limit_bytes", "container_spec_memory_limit_bytes"),
    "request_rate": ("request_rate", "requests_per_second", "http_requests_per_second", "rate_http_requests_total"),
    "error_rate": ("error_rate", "http_error_rate", "5xx_rate"),
    "queue_depth": ("queue_depth", "queue_length", "waiting_requests", "vllm:num_requests_waiting", "num_requests_waiting"),
    "utilization": ("utilization", "gpu_utilization", "replica_utilization"),
    "latency_p50_ms": ("latency_p50_ms", "p50_ms", "request_latency_p50_ms", "http_request_duration_p50_ms"),
    "latency_p95_ms": ("latency_p95_ms", "p95_ms", "request_latency_p95_ms", "http_request_duration_p95_ms"),
    "latency_p99_ms": ("latency_p99_ms", "p99_ms", "request_latency_p99_ms", "http_request_duration_p99_ms"),
    "restarts": ("restarts", "restart_count", "container_restarts", "kube_pod_container_status_restarts_total"),
    "ready": ("ready", "readiness", "pod_ready", "kube_pod_status_ready"),
    "startup_seconds": ("startup_seconds", "startup_duration_seconds", "time_to_ready_seconds"),
    "oom_kills": ("oom_kills", "oomkill", "oom_kill_count", "container_oom_events_total"),
}
_ALIAS_TO_CANON = {alias.lower(): key for key, aliases in _ALIASES.items() for alias in aliases}


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        text = value[:MAX_TEXT]
        for pattern, replacement in _SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:MAX_RECORDS]:
            name = str(key)[:160]
            if re.search(r"(?i)(password|passwd|token|secret|api[_-]?key|credential)", name):
                result[name] = "[REDACTED]"
            else:
                result[name] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value[:MAX_RECORDS]]
    return value


def _bounded(payload: dict) -> dict:
    try:
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    except Exception:
        return {}
    truncated = len(raw) > MAX_INPUT_BYTES
    # Trim collections and text before normalization; only bounded material is
    # ever retained, and a warning records oversized input.
    safe = _redact(payload)
    state = {"nodes": 0, "text": MAX_INPUT_BYTES // 3}
    safe = _clip_payload(safe, state)
    if truncated:
        safe["_truncated"] = True
    return safe


def _clip_payload(value: Any, state: dict[str, int]) -> Any:
    """Limit aggregate nodes and text, rather than only each individual list."""
    state["nodes"] += 1
    if state["nodes"] > 6_000:
        return "[TRUNCATED]"
    if isinstance(value, str):
        limit = max(0, min(MAX_TEXT, state["text"]))
        result = value[:limit]
        state["text"] -= len(result)
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:MAX_RECORDS]:
            if state["nodes"] >= 6_000 or state["text"] <= 0:
                result["_truncated"] = True
                break
            result[str(key)[:120]] = _clip_payload(item, state)
        return result
    if isinstance(value, list):
        result = []
        for item in value[:MAX_RECORDS]:
            if state["nodes"] >= 6_000 or state["text"] <= 0:
                break
            result.append(_clip_payload(item, state))
        return result
    return value


def _metric_name(name: Any) -> str:
    text = str(name).strip()
    key = text.lower().replace("-", "_").replace(".", "_")
    return _ALIAS_TO_CANON.get(key, text)


def _metric_entry(value: Any, unit: str | None = None, timestamp: Any = None) -> dict:
    if isinstance(value, dict) and ("value" in value or "samples" in value):
        out = {k: v for k, v in value.items() if k in {"value", "unit", "samples", "labels", "timestamp", "type"}}
        if "value" in out:
            out["value"] = _number(out["value"])
        if isinstance(out.get("samples"), list):
            out["samples"] = [_normalize_sample(sample) for sample in out["samples"][:MAX_RECORDS]]
        if unit and "unit" not in out:
            out["unit"] = unit
        return out
    if isinstance(value, list):
        samples = []
        if len(value) >= 2 and not isinstance(value[0], (list, tuple, dict)):
            samples.append({"timestamp": value[0], "value": _number(value[1])})
            return {"samples": samples, **({"unit": unit} if unit else {})}
        for sample in value[:MAX_RECORDS]:
            if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                samples.append({"timestamp": sample[0], "value": _number(sample[1])})
            else:
                samples.append(_normalize_sample(sample))
        return {"samples": samples, **({"unit": unit} if unit else {})}
    out = {"value": value}
    if unit:
        out["unit"] = unit
    if timestamp is not None:
        out["timestamp"] = timestamp
    return out


def _normalize_sample(sample: Any) -> Any:
    if isinstance(sample, dict):
        sample = dict(sample)
        if "value" in sample:
            sample["value"] = _number(sample["value"])
        return sample
    return _number(sample)


def _quantity(value: Any) -> tuple[Any, str | None]:
    """Parse common Kubernetes resource quantities while retaining their unit."""
    if not isinstance(value, str):
        return _number(value), None
    text = value.strip()
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(Ki|Mi|Gi|Ti|K|M|G|T|m|n|u)?", text)
    if not match:
        return value, None
    suffix = match.group(2) or ""
    multiplier = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
                  "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4,
                  "m": 0.001, "u": 0.000001, "n": 0.000000001}.get(suffix, 1)
    number = float(match.group(1)) * multiplier
    if number.is_integer():
        number = int(number)
    unit = "cores" if suffix in {"m", "u", "n"} else ("bytes" if suffix in {"Ki", "Mi", "Gi", "Ti", "K", "M", "G", "T"} else None)
    return number, unit


def _normalize_metrics(source: Any, fmt: str) -> dict:
    metrics: dict[str, Any] = {}
    if isinstance(source, dict) and isinstance(source.get("data"), (dict, list)):
        source = source["data"]
    # Prometheus HTTP API envelope (both instant vectors and range matrices).
    if isinstance(source, dict) and isinstance(source.get("result"), list):
        source = source["result"]
    if isinstance(source, dict):
        for name, value in list(source.items())[:MAX_RECORDS]:
            key = _metric_name(name)
            if isinstance(value, list) and value and isinstance(value[0], dict) and ("metric" in value[0] or "values" in value[0]):
                samples = []
                for item in value[:MAX_RECORDS]:
                    labels = item.get("metric", {})
                    raw_values = item.get("values", [item.get("value")])
                    for sample in (raw_values[:MAX_RECORDS] if isinstance(raw_values, list) else [raw_values]):
                        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                            samples.append({"timestamp": sample[0], "value": _number(sample[1]), "labels": labels})
                        elif sample is not None:
                            samples.append({"value": _number(sample), "labels": labels})
                metrics[key] = {"samples": samples, "type": "counter" if "_total" in str(name) else "gauge"}
            else:
                entry = _metric_entry(value)
                if str(name).endswith("_total") or key in {"oom_kills", "restarts"}:
                    entry.setdefault("type", "counter")
                metrics[key] = entry
    elif isinstance(source, list):
        for item in source[:MAX_RECORDS]:
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("metric", item.get("__name__", "unknown")))
            if isinstance(name, dict):
                metric_labels = name
                name = metric_labels.get("__name__", "unknown")
            else:
                metric_labels = item.get("labels", {})
            key = _metric_name(name)
            value = item.get("value", item.get("values", item.get("samples")))
            entry = _metric_entry(value, item.get("unit"), item.get("timestamp"))
            if str(name).endswith("_total") or key in {"oom_kills", "restarts"}:
                entry.setdefault("type", "counter")
            if metric_labels:
                if isinstance(entry.get("samples"), list):
                    entry["samples"] = [({**sample, "labels": sample.get("labels", metric_labels)}
                                          if isinstance(sample, dict) else sample)
                                         for sample in entry["samples"]]
                entry.setdefault("labels", metric_labels)
            if key in metrics:
                previous = metrics[key].setdefault("samples", [])
                incoming = entry.get("samples") if isinstance(entry.get("samples"), list) else [entry]
                for sample in incoming:
                    if isinstance(sample, dict):
                        sample = dict(sample)
                        if metric_labels:
                            sample.setdefault("labels", metric_labels)
                    previous.append(sample)
            else:
                metrics[key] = entry
    return metrics


def _number(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value
    return value


def _extract_text_events(text: str) -> tuple[list[dict], list[str]]:
    events, excerpts = [], []
    for line in text.splitlines()[:MAX_RECORDS]:
        safe = _redact(line.strip())
        if not safe:
            continue
        lowered = safe.lower()
        kind = None
        if re.search(r"oom.?kill|out of memory|oomkilled", lowered):
            kind = "oom"
        elif re.search(r"readiness.*(?:timeout|fail)|probe.*timeout|not ready", lowered):
            kind = "readiness_timeout"
        elif re.search(r"crashloopbackoff|back-off restarting|container.*restarted", lowered):
            kind = "restart"
        elif re.search(r"gateway.*(?:502|503|504)|upstream.*(?:timeout|reset|unavailable)|http[ /]5\d\d", lowered):
            kind = "gateway_failure"
        elif re.search(r"slow startup|startup.*(?:slow|timeout)|started after", lowered):
            kind = "slow_startup"
        elif re.search(r"overload|queue.*(?:full|saturat)|too many requests", lowered):
            kind = "overload"
        if kind:
            events.append({"type": kind, "message": safe[:MAX_TEXT]})
            if len(excerpts) < MAX_EXCERPTS:
                excerpts.append(safe[:MAX_TEXT])
    return events, excerpts


def _normalize_kubernetes(data: Any) -> tuple[dict, list[dict], list[str]]:
    """Extract quantities, pod state, and Event objects from Kubernetes exports."""
    metrics: dict[str, Any] = {}
    events: list[dict] = []
    excerpts: list[str] = []
    objects = data.get("items", []) if isinstance(data, dict) and isinstance(data.get("items"), list) else [data]
    for obj in objects[:MAX_RECORDS]:
        if not isinstance(obj, dict):
            continue
        kind = str(obj.get("kind", "")).lower()
        meta = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
        status = obj.get("status", {}) if isinstance(obj.get("status"), dict) else {}
        reason = obj.get("reason", status.get("reason", ""))
        message = obj.get("message", status.get("message", ""))
        if kind == "event" or reason or message:
            text = _redact(f"{reason} {message}".strip())
            low = text.lower()
            event_type = ("oom" if re.search(r"oom.?kill|out of memory|oomkilled", low) else
                          "readiness_timeout" if re.search(r"readiness|probe.*timeout|not ready", low) else
                          "restart" if re.search(r"crashloop|back-off|restart", low) else None)
            if event_type:
                events.append({"type": event_type, "reason": _redact(reason), "message": _redact(message),
                               "count": _number(obj.get("count", 1)), "timestamp": obj.get("lastTimestamp", obj.get("eventTime"))})
                if len(excerpts) < MAX_EXCERPTS:
                    excerpts.append(text[:MAX_TEXT])
        for condition in status.get("conditions", []) if isinstance(status.get("conditions"), list) else []:
            if isinstance(condition, dict) and condition.get("type") == "Ready":
                metrics.setdefault("ready", {"samples": []})["samples"].append(
                    {"value": 1 if str(condition.get("status")).lower() == "true" else 0,
                     "timestamp": condition.get("lastTransitionTime"), "labels": {"pod": meta.get("name", "")}})
        for container in status.get("containerStatuses", []) if isinstance(status.get("containerStatuses"), list) else []:
            if isinstance(container, dict) and "restartCount" in container:
                restart_metric = metrics.setdefault("restarts", {"samples": [], "type": "counter"})
                restart_metric["type"] = "counter"
                restart_metric["samples"].append(
                    {"value": _number(container["restartCount"]), "timestamp": status.get("startTime"),
                     "labels": {"pod": meta.get("name", ""), "container": container.get("name", "")}})
            waiting = container.get("state", {}).get("waiting", {}) if isinstance(container.get("state"), dict) else {}
            waiting_reason = str(waiting.get("reason", ""))
            if waiting_reason.lower() == "crashloopbackoff":
                events.append({"type": "restart", "reason": waiting_reason, "pod": meta.get("name", "")})
        # Metrics.k8s.io PodMetrics shape: containers[].usage.{cpu,memory}.
        for container in obj.get("containers", []) if isinstance(obj.get("containers"), list) else []:
            if not isinstance(container, dict):
                continue
            usage = container.get("usage", {}) if isinstance(container.get("usage"), dict) else {}
            for resource, metric in (("memory", "memory_bytes"), ("cpu", "cpu_utilization")):
                if resource in usage:
                    value, unit = _quantity(usage[resource])
                    metrics.setdefault(metric, {"samples": []})["samples"].append(
                        {"value": value, "unit": unit or ("bytes" if resource == "memory" else "cores"),
                         "timestamp": obj.get("timestamp"), "labels": {"pod": meta.get("name", ""), "container": container.get("name", "")}})
    return metrics, events, excerpts


def _timestamp(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value
        except ValueError:
            return value[:100]
    return None


def normalize_evidence(payload: dict) -> dict:
    """Normalize one exported evidence bundle into a bounded redacted record."""
    if not isinstance(payload, dict):
        payload = {"data": payload}
    safe = _bounded(payload)
    fmt = str(safe.get("format", safe.get("type", "normalized"))).lower()
    provenance = "fixture" if safe.get("provenance") == "fixture" or safe.get("fixture") else "imported"
    data = safe.get("data", safe.get("metrics", {}))
    metrics_source = data
    if fmt in {"logs", "log", "text"}:
        metrics_source = safe.get("metrics", {})
    is_kubernetes = fmt in {"kubernetes", "k8s", "kube"}
    metrics, kube_events, kube_excerpts = _normalize_kubernetes(data) if is_kubernetes else ({}, [], [])
    normalized = {} if is_kubernetes and isinstance(data, dict) and isinstance(data.get("items"), list) else _normalize_metrics(metrics_source, fmt)
    metrics = {**normalized, **metrics}
    events = list(safe.get("events", []))[:MAX_RECORDS] if isinstance(safe.get("events"), list) else []
    events.extend(kube_events[:MAX_RECORDS - len(events)])
    excerpts = list(safe.get("excerpts", []))[:MAX_EXCERPTS] if isinstance(safe.get("excerpts"), list) else []
    excerpts.extend(kube_excerpts[:max(0, MAX_EXCERPTS - len(excerpts))])
    raw_text = safe.get("logs", safe.get("text", data if isinstance(data, str) else ""))
    if isinstance(raw_text, str):
        found, snippets = _extract_text_events(raw_text[:MAX_INPUT_BYTES])
        events.extend(found[:MAX_RECORDS - len(events)])
        excerpts.extend(snippets[:max(0, MAX_EXCERPTS - len(excerpts))])
    events = [_redact(e) for e in events[:MAX_RECORDS] if isinstance(e, (dict, str))]
    excerpts = [_redact(e) for e in excerpts[:MAX_EXCERPTS] if isinstance(e, str)]
    warnings = list(safe.get("warnings", [])) if isinstance(safe.get("warnings"), list) else []
    if safe.get("_truncated"):
        warnings.append("input exceeded size limit; collections and text were bounded")
    # Record source names and format only; do not copy arbitrary source payloads.
    sources = safe.get("sources", [])
    if isinstance(sources, str):
        sources = [sources]
    sources = [str(x)[:160] for x in sources[:40]] if isinstance(sources, list) else []
    coverage = safe.get("coverage", {})
    if not isinstance(coverage, dict):
        coverage = {"description": str(coverage)[:MAX_TEXT]}
    record = {
        "provenance": provenance,
        "environment_id": safe.get("environment_id", safe.get("cluster", safe.get("namespace"))),
        "plan_revision": safe.get("plan_revision", safe.get("revision")),
        "observed_at": _timestamp(safe.get("observed_at", safe.get("timestamp"))),
        "metrics": metrics,
        "events": events,
        "excerpts": excerpts,
        "coverage": _redact(coverage),
        "sources": sources,
        "warnings": [str(_redact(w))[:MAX_TEXT] for w in warnings[:100]],
    }
    if safe.get("applied_bundle_id") is not None:
        record["applied_bundle_id"] = str(safe["applied_bundle_id"])[:160]
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    record["evidence_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return record


def _metric_value(evidence: dict, key: str) -> Any:
    metric = evidence.get("metrics", {}).get(key)
    if isinstance(metric, dict):
        value = metric.get("value")
        if value is None and metric.get("samples"):
            samples = metric["samples"]
            if isinstance(samples, list):
                values = [s.get("value") for s in samples if isinstance(s, dict) and isinstance(s.get("value"), (int, float))]
                value = values[-1] if values else None
        return value
    return metric


def _arch(plan: dict) -> dict:
    return plan if isinstance(plan, dict) else {}


def _values(evidence: dict, key: str) -> list[float]:
    entry = evidence.get("metrics", {}).get(key)
    if not isinstance(entry, dict):
        entry = {"value": entry}
    values = []
    for sample in entry.get("samples", []) if isinstance(entry.get("samples"), list) else []:
        value = sample.get("value") if isinstance(sample, dict) else sample
        if isinstance(value, (int, float)):
            values.append(float(value))
    if isinstance(entry.get("value"), (int, float)):
        values.append(float(entry["value"]))
    return values


def _event_types(evidence: dict) -> set[str]:
    return {str(event.get("type", "")) if isinstance(event, dict) else str(event)
            for event in evidence.get("events", []) if isinstance(event, (dict, str))}


def _replica_capacity(plan: dict) -> int:
    allocation = plan.get("allocation", {}) if isinstance(plan.get("allocation"), dict) else {}
    inventory = plan.get("inventory", {}) if isinstance(plan.get("inventory"), dict) else {}
    nodes = inventory.get("nodes", []) if isinstance(inventory.get("nodes"), list) else []
    gpus_per_replica = max(1, int(allocation.get("gpus_per_replica", allocation.get("tensor_parallel", 1))))
    declared_max = allocation.get("max_replicas", plan.get("max_replicas", 6))
    try:
        declared_max = max(1, int(declared_max))
    except (TypeError, ValueError):
        declared_max = 6
    # Each replica's tensor-parallel GPUs must fit on one node; don't combine
    # fragments across nodes when estimating safe scale-out capacity.
    capacity = sum(max(0, int(node.get("gpu_count", 0))) // gpus_per_replica
                   for node in nodes if isinstance(node, dict))
    return min(6, declared_max, capacity) if capacity else 0


def diagnose(plan: dict, evidence: dict, kind: str) -> dict:
    """Infer supported faults or check an explicit fault, then propose bounded knobs."""
    plan = plan if isinstance(plan, dict) else {}
    evidence = evidence if isinstance(evidence, dict) else {}
    requested = str(kind or "repair").lower().replace("-", "_").replace(" ", "_")
    aliases = {"oomkill": "oom", "oom_kill": "oom", "queue_saturation": "overload",
               "readiness": "readiness_timeout", "crashloop": "readiness_timeout",
               "startup": "slow_startup", "gateway": "gateway_failure", "5xx": "gateway_failure"}
    requested = aliases.get(requested, requested)
    config = plan.get("config", {}) if isinstance(plan.get("config"), dict) else {}
    allocation = plan.get("allocation", {}) if isinstance(plan.get("allocation"), dict) else {}
    workload = plan.get("workload", {}) if isinstance(plan.get("workload"), dict) else {}
    events = _event_types(evidence)
    metrics = evidence.get("metrics", {}) if isinstance(evidence.get("metrics"), dict) else {}
    metric_names = set(metrics)
    memory, mem_limit = _metric_value(evidence, "memory_bytes"), _metric_value(evidence, "memory_limit_bytes")
    memory_hot = isinstance(memory, (int, float)) and isinstance(mem_limit, (int, float)) and mem_limit > 0 and memory / mem_limit >= 0.90
    queue_vals, util_vals = _values(evidence, "queue_depth"), _values(evidence, "utilization")
    queue_hot = bool(queue_vals) and queue_vals[-1] > 0
    util_hot = bool(util_vals) and util_vals[-1] >= 0.80
    slo = workload.get("slo_p95_ms", 4000)
    latency_vals = _values(evidence, "latency_p95_ms")
    latency_hot = bool(latency_vals) and isinstance(slo, (int, float)) and latency_vals[-1] > slo
    error_vals = _values(evidence, "error_rate")
    error_limit = workload.get("max_error_rate", 0.05)
    error_hot = bool(error_vals) and isinstance(error_limit, (int, float)) and error_vals[-1] > error_limit
    startup_vals = _values(evidence, "startup_seconds")
    startup_budget = config.get("startup_budget_seconds", 180)
    startup_hot = bool(startup_vals) and isinstance(startup_budget, (int, float)) and startup_vals[-1] > startup_budget

    detected = []
    if "oom" in events or memory_hot:
        detected.append("oom")
    if queue_hot and (util_hot or latency_hot):
        detected.append("overload")
    # A measured startup overrun is the more specific explanation for readiness failures.
    if ("slow_startup" in events or startup_hot) and ("readiness_timeout" in events or startup_hot):
        detected.append("slow_startup")
    if events.intersection({"readiness_timeout", "restart"}):
        detected.append("readiness_timeout")
    if "gateway_failure" in events or (error_hot and latency_hot):
        detected.append("gateway_failure")
    # Idle capacity is an optimization finding, never an incident.
    replicas_now = int(allocation.get("replicas", 1) or 1)
    idle = (bool(queue_vals) and max(queue_vals) == 0 and bool(util_vals) and max(util_vals) < 0.30
            and not latency_hot and not error_hot and replicas_now > 1)
    if idle and not detected:
        detected.append("underutilized")
    supported = {"oom", "overload", "readiness_timeout", "slow_startup", "gateway_failure", "underutilized"}
    fault = requested if requested in supported else (detected[0] if detected else None)
    if requested in {"repair", "optimize"}:
        candidates = [f for f in detected if requested == "optimize" or f != "underutilized"]
        fault = candidates[0] if candidates else None

    evidence_id = evidence.get("evidence_id")
    result = {"status": "awaiting_evidence", "diagnosis": "Evidence does not show a supported, correlated production fault.",
              "alternatives": [], "missing_evidence": [], "patch": {}, "conditions": {},
              "tradeoffs": [], "evidence_ids": [evidence_id] if evidence_id else [], "fault": fault}
    if fault is None:
        result["missing_evidence"] = ["correlated fault event or metric samples; healthy observations do not trigger a remediation"]
        return result

    missing: list[str] = []
    patch_config: dict[str, Any] = {}
    patch_allocation: dict[str, Any] = {}
    alternatives: list[dict] = []
    tradeoffs: list[str] = []
    if fault == "oom":
        if "oom" not in events and not memory_hot:
            missing.append("OOM event or memory usage at least 90% of its limit")
        current_seqs = config.get("max_num_seqs")
        current_len = config.get("max_model_len")
        if isinstance(current_seqs, int) and current_seqs > 1:
            patch_config["max_num_seqs"] = max(1, current_seqs // 2)
        elif isinstance(current_len, int) and current_len > 128:
            patch_config["max_model_len"] = max(128, int(current_len * 0.8))
        else:
            missing.append("current max_num_seqs or max_model_len to choose a renderer-supported memory reduction")
        diagnosis = "OOM or near-limit memory evidence supports reducing per-replica KV-cache demand."
        alternatives = [{"config": {"max_model_len": max(128, int(current_len * 0.8))} if isinstance(current_len, int) and current_len > 128 else {},
                         "when": "long contexts dominate memory after concurrency is reduced"}]
        tradeoffs = ["Lower concurrency or context length reduces memory demand but can reduce throughput or truncate accepted workloads."]
        conditions = {"metrics": {"memory_bytes": {"lt_ratio_of": {"metric": "memory_limit_bytes", "ratio": 0.85}},
                                   "oom_kills": {"delta_eq": 0}}, "window_samples": 2}
    elif fault == "overload":
        if not queue_hot:
            missing.append("positive queue-depth sample")
        if not (util_hot or latency_hot):
            missing.append("utilization samples or p95 latency above the workload SLO, correlated with queued requests")
        current_replicas = int(allocation.get("replicas", 1))
        capacity = _replica_capacity(plan)
        seqs, seqs_fit = config.get("max_num_seqs"), allocation.get("max_num_seqs_fit")
        if capacity > current_replicas:
            patch_allocation["replicas"] = current_replicas + 1
            diagnosis = "Queued demand coincides with saturated utilization or an SLO breach; one hardware-bounded replica is the proposed capacity step."
            tradeoffs = ["One more replica raises GPU cost; confirm scheduling, per-replica memory, and traffic behavior before rollout."]
        elif isinstance(seqs, int) and isinstance(seqs_fit, int) and seqs_fit > seqs:
            # Every GPU already hosts a replica; use estimated KV-cache headroom for a larger batch instead.
            patch_config["max_num_seqs"] = min(seqs_fit, seqs * 2)
            diagnosis = ("Queued demand coincides with saturated utilization or an SLO breach, and every GPU already hosts a replica; "
                         f"estimated KV-cache headroom allows raising max_num_seqs from {seqs} to {patch_config['max_num_seqs']}.")
            tradeoffs = ["A larger batch raises throughput but can increase inter-token latency; confirm GPU memory headroom under load."]
        else:
            missing.append("inventory-confirmed spare GPU capacity for one additional replica")
            diagnosis = "Queued demand coincides with saturated utilization or an SLO breach, but no spare GPU or KV-cache headroom remains; add GPU nodes."
            tradeoffs = ["More capacity requires more hardware."]
        conditions = {"metrics": {"queue_depth": {"eq": 0}, "utilization": {"lt": 0.8},
                                   "latency_p95_ms": {"lte": slo}}, "window_samples": 3}
    elif fault == "readiness_timeout":
        if not events.intersection({"readiness_timeout", "restart"}):
            missing.append("readiness timeout or restart event")
        patch_config = {"readiness_initial_delay_seconds": 60, "readiness_timeout_seconds": 10}
        diagnosis = "Readiness timeout or restart evidence supports a larger startup probe budget."
        alternatives = [{"config": {"max_num_seqs": max(1, int(config.get("max_num_seqs", 1)) // 2)},
                         "when": "startup delay correlates with memory or CPU saturation"}]
        tradeoffs = ["A longer readiness window can delay detection of an unhealthy process; keep liveness checks independent."]
        conditions = {"metrics": {"ready": {"eq": 1}, "restarts": {"delta_eq": 0}}, "window_samples": 2}
    elif fault == "slow_startup":
        if not startup_hot and "slow_startup" not in events:
            missing.append("startup duration over configured budget or slow-startup event")
        patch_config = {"startup_probe_failure_threshold": 30, "readiness_initial_delay_seconds": 90}
        diagnosis = "Measured startup exceeds its budget; allow additional startup time while retaining readiness gating."
        tradeoffs = ["A larger startup budget delays failure detection and should be revisited after startup timings improve."]
        conditions = {"metrics": {"ready": {"eq": 1}, "startup_seconds": {"lte": startup_budget}}, "window_samples": 2}
    elif fault == "underutilized":
        if not idle:
            missing.append("empty queue with utilization below 30% across samples, latency within SLO, and more than one replica")
        if replicas_now > 1:
            patch_allocation["replicas"] = replicas_now - 1
        gpus = max(1, int(allocation.get("gpus_per_replica", allocation.get("tensor_parallel", 1)) or 1))
        diagnosis = (f"Utilization stayed below 30% with an empty queue; removing one replica frees {gpus} GPU(s), "
                     f"{gpus * 24 * 30} GPU-hours per 30 days.")
        tradeoffs = ["Less headroom for traffic bursts; keep the recovery conditions below as a scale-back-up trigger."]
        conditions = {"metrics": {"queue_depth": {"eq": 0}, "latency_p95_ms": {"lte": slo}}, "window_samples": 3}
    else:  # gateway_failure
        if not ("gateway_failure" in events or error_hot and latency_hot):
            missing.append("gateway failure event or elevated error rate correlated with SLO latency breach")
        patch_config = {"upstream_timeout_seconds": 30, "retry_max_attempts": 1}
        diagnosis = "Gateway or upstream failures support a bounded timeout increase and at most one retry."
        alternatives = [{"config": {"retry_max_attempts": 0}, "when": "requests are non-idempotent or retries amplify overload"}]
        tradeoffs = ["Retries can amplify overload or repeat unsafe requests; apply only to idempotent operations."]
        conditions = {"metrics": {"error_rate": {"lte": error_limit}, "latency_p95_ms": {"lte": slo}}, "window_samples": 3}

    if missing:
        result.update({"diagnosis": diagnosis, "missing_evidence": missing, "alternatives": alternatives,
                       "tradeoffs": tradeoffs, "conditions": conditions})
        return result
    result.update({"status": "remediation_proposed", "diagnosis": diagnosis, "alternatives": alternatives,
                   "patch": {"config": patch_config, "allocation": patch_allocation},
                   "conditions": conditions, "tradeoffs": tradeoffs})
    return result


def verify(conditions: dict, evidence: dict) -> dict:
    """Verify all numeric gates; incomplete, stale, or reset-counter windows wait."""
    reasons, missing = [], []
    conditions = conditions if isinstance(conditions, dict) else {}
    evidence = evidence if isinstance(evidence, dict) else {}
    metrics = evidence.get("metrics", {}) if isinstance(evidence.get("metrics"), dict) else {}
    checks = conditions.get("metrics", {}) if isinstance(conditions.get("metrics"), dict) else {}
    coverage = evidence.get("coverage", {}) if isinstance(evidence.get("coverage"), dict) else {}
    if not checks:
        missing.append("recovery metric conditions")
    if "environment_id" in conditions and evidence.get("environment_id") != conditions["environment_id"]:
        missing.append("matching environment evidence")
    if "plan_revision" in conditions and evidence.get("plan_revision") != conditions["plan_revision"]:
        missing.append("matching plan revision evidence")
    required_coverage = conditions.get("coverage_window_seconds")
    if required_coverage is not None:
        observed_coverage = coverage.get("window_seconds")
        if not isinstance(observed_coverage, (int, float)) or observed_coverage < required_coverage:
            missing.append(f"coverage window of at least {required_coverage} seconds")
    observed_at = evidence.get("observed_at")
    if conditions.get("max_age_seconds") is not None:
        max_age = conditions["max_age_seconds"]
        now = conditions.get("now")
        try:
            observed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00")).timestamp()
            current = float(now) if isinstance(now, (int, float)) else datetime.now().astimezone().timestamp()
            if current - observed > float(max_age) or observed > current:
                missing.append("fresh evidence within the permitted age")
        except (TypeError, ValueError, OverflowError):
            missing.append("timestamped evidence for freshness validation")

    def samples_for(name: str) -> tuple[list[float], list[Any]]:
        entry = metrics.get(name)
        if not isinstance(entry, dict):
            entry = {"value": entry}
        values, timestamps = [], []
        for sample in entry.get("samples", []) if isinstance(entry.get("samples"), list) else []:
            value = sample.get("value") if isinstance(sample, dict) else sample
            if isinstance(value, (int, float)):
                values.append(float(value))
                timestamps.append(sample.get("timestamp") if isinstance(sample, dict) else None)
        if not values and isinstance(entry.get("value"), (int, float)):
            values.append(float(entry["value"]))
            timestamps.append(entry.get("timestamp"))
        return values, timestamps

    for name, predicate in checks.items():
        if name not in metrics:
            missing.append(f"metric: {name}")
            continue
        values, timestamps = samples_for(name)
        metric_entry = metrics.get(name)
        metric_type = metric_entry.get("type") if isinstance(metric_entry, dict) else None
        if not values:
            missing.append(f"numeric samples: {name}")
            continue
        if not isinstance(predicate, dict):
            predicate = {"eq": predicate}
        op = next((key for key in ("lte", "lt", "gte", "gt", "eq", "delta_eq", "lt_ratio_of") if key in predicate), None)
        required = max(1, int(conditions.get("required_samples", conditions.get("window_samples", 1))))
        if op == "delta_eq" or metric_type == "counter":
            required = max(2, required)
        if len(values) < required:
            missing.append(f"{name}: {required} numeric samples required")
            continue
        window_values = values[-required:]
        window_times = timestamps[-required:]
        if conditions.get("require_timestamps") and any(ts is None for ts in window_times):
            missing.append(f"{name}: timestamped samples required")
            continue
        if metric_type == "counter" and any(right < left for left, right in zip(window_values, window_values[1:])):
            missing.append(f"{name}: counter reset makes comparison unknown")
            continue
        if op == "delta_eq":
            if any(right < left for left, right in zip(window_values, window_values[1:])):
                missing.append(f"{name}: counter reset makes delta unknown")
                continue
            passed = window_values[-1] - window_values[0] == predicate[op]
        elif op == "lt_ratio_of":
            relation = predicate[op]
            other_name = relation.get("metric") if isinstance(relation, dict) else None
            other_values, _ = samples_for(other_name) if other_name else ([], [])
            if not other_values:
                missing.append(f"numeric samples: {other_name}")
                continue
            passed = window_values[-1] < other_values[-1] * float(relation.get("ratio", 1))
        else:
            threshold = predicate.get(op) if op else None
            if threshold is None or not isinstance(threshold, (int, float)):
                missing.append(f"numeric threshold for {name}")
                continue
            compare = {"lte": lambda x: x <= threshold, "lt": lambda x: x < threshold,
                       "gte": lambda x: x >= threshold, "gt": lambda x: x > threshold,
                       "eq": lambda x: x == threshold}.get(op)
            if compare is None:
                missing.append(f"supported numeric predicate for {name}")
                continue
            passed = all(compare(value) for value in window_values)
        reasons.append(f"{name} {'met' if passed else 'not met'} recovery condition")
    if missing:
        outcome = "awaiting_evidence"
    elif any("not met" in reason for reason in reasons):
        outcome = "contradicted"
    else:
        outcome = "confirmed"
    return {"outcome": outcome, "reasons": reasons, "missing_evidence": missing,
            "evidence_ids": [evidence["evidence_id"]] if evidence.get("evidence_id") else []}


def fixtures() -> dict[str, dict]:
    """Small representative exports for deterministic, offline demonstrations."""
    return {
        "oom": {"format": "kubernetes", "provenance": "fixture", "environment_id": "demo-cluster",
                "plan_revision": 7, "applied_bundle_id": "bundle-7", "observed_at": "2026-09-26T12:00:00Z",
                "data": {"memory_bytes": {"value": 950000000, "unit": "bytes"},
                         "memory_limit_bytes": {"value": 1000000000, "unit": "bytes"}, "oom_kills": 2},
                "events": [{"type": "oom", "reason": "OOMKilled"}], "coverage": {"window_seconds": 300}, "sources": ["kubernetes metrics"]},
        "overload": {"format": "prometheus", "provenance": "fixture", "environment_id": "demo-cluster",
                     "plan_revision": 7, "observed_at": "2026-09-26T12:05:00Z",
                     "data": {"vllm:num_requests_waiting": {"samples": [{"timestamp": 1, "value": 12}, {"timestamp": 2, "value": 18}], "unit": "requests"},
                              "gpu_utilization": {"value": 0.97, "unit": "ratio"}, "p95_ms": {"value": 2100, "unit": "ms"}},
                     "coverage": {"window_seconds": 300}, "sources": ["prometheus"]},
        "readiness": {"format": "logs", "provenance": "fixture", "environment_id": "demo-cluster",
                      "plan_revision": 7, "observed_at": "2026-09-26T12:10:00Z",
                      "logs": "2026-09-26T12:09:00Z readiness probe timeout pod=worker-1\n2026-09-26T12:09:30Z CrashLoopBackOff",
                      "coverage": {"window_seconds": 180}, "sources": ["pod logs"]},
        "gateway": {"format": "logs", "provenance": "fixture", "environment_id": "demo-cluster",
                    "plan_revision": 7, "observed_at": "2026-09-26T12:15:00Z",
                    "logs": "upstream timeout returned HTTP 504; authorization: Bearer demo-secret",
                    "coverage": {"window_seconds": 60}, "sources": ["gateway logs"]},
        "slow-startup": {"format": "normalized", "provenance": "fixture", "observed_at": "2026-09-26T12:20:00Z",
                         "data": {"startup_seconds": {"value": 420, "unit": "s"}},
                         "logs": "2026-09-26T12:19:00Z readiness probe timeout pod=worker-0 (model still loading)",
                         "coverage": {"window_seconds": 600}, "sources": ["pod logs", "kube-state-metrics"]},
        "underutilized": {"format": "prometheus", "provenance": "fixture", "observed_at": "2026-09-26T12:25:00Z",
                          "data": {"vllm:num_requests_waiting": {"samples": [{"timestamp": t, "value": 0} for t in (1, 2, 3)]},
                                   "gpu_utilization": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 0.14), (2, 0.11), (3, 0.18))]},
                                   "p95_ms": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 610), (2, 580), (3, 640))], "unit": "ms"}},
                          "coverage": {"window_seconds": 3600}, "sources": ["prometheus"]},
        # Exports observed after a change was applied, for verification.
        "overload-recovered": {"format": "prometheus", "provenance": "fixture", "observed_at": "2026-09-26T13:05:00Z",
                               "data": {"vllm:num_requests_waiting": {"samples": [{"timestamp": t, "value": 0} for t in (1, 2, 3)]},
                                        "gpu_utilization": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 0.71), (2, 0.68), (3, 0.74))]},
                                        "p95_ms": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 1400), (2, 1320), (3, 1380))], "unit": "ms"}},
                               "coverage": {"window_seconds": 900}, "sources": ["prometheus"]},
        "overload-persists": {"format": "prometheus", "provenance": "fixture", "observed_at": "2026-09-26T13:05:00Z",
                              "data": {"vllm:num_requests_waiting": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 9), (2, 14), (3, 11))]},
                                       "gpu_utilization": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 0.96), (2, 0.98), (3, 0.97))]},
                                       "p95_ms": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 4600), (2, 5100), (3, 4800))], "unit": "ms"}},
                              "coverage": {"window_seconds": 900}, "sources": ["prometheus"]},
        "underutilized-recovered": {"format": "prometheus", "provenance": "fixture", "observed_at": "2026-09-26T14:25:00Z",
                                    "data": {"vllm:num_requests_waiting": {"samples": [{"timestamp": t, "value": 0} for t in (1, 2, 3)]},
                                             "p95_ms": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 720), (2, 690), (3, 750))], "unit": "ms"}},
                                    "coverage": {"window_seconds": 3600}, "sources": ["prometheus"]},
        "oom-recovered": {"format": "normalized", "provenance": "fixture", "observed_at": "2026-09-26T13:00:00Z",
                          "data": {"memory_bytes": {"samples": [{"timestamp": 1, "value": 700000000}, {"timestamp": 2, "value": 720000000}], "unit": "bytes"},
                                   "memory_limit_bytes": {"value": 1000000000, "unit": "bytes"},
                                   "oom_kills": {"samples": [{"timestamp": 1, "value": 2}, {"timestamp": 2, "value": 2}]}},
                          "coverage": {"window_seconds": 900}, "sources": ["kubernetes metrics"]},
        "readiness-recovered": {"format": "normalized", "provenance": "fixture", "observed_at": "2026-09-26T13:10:00Z",
                                "data": {"ready": {"samples": [{"timestamp": 1, "value": 1}, {"timestamp": 2, "value": 1}]},
                                         "restarts": {"samples": [{"timestamp": 1, "value": 3}, {"timestamp": 2, "value": 3}]}},
                                "coverage": {"window_seconds": 900}, "sources": ["kube-state-metrics"]},
        "slow-startup-recovered": {"format": "normalized", "provenance": "fixture", "observed_at": "2026-09-26T13:20:00Z",
                                   "data": {"ready": {"samples": [{"timestamp": 1, "value": 1}, {"timestamp": 2, "value": 1}]},
                                            "startup_seconds": {"samples": [{"timestamp": 1, "value": 150}, {"timestamp": 2, "value": 140}], "unit": "s"}},
                                   "coverage": {"window_seconds": 900}, "sources": ["kube-state-metrics"]},
        "gateway-recovered": {"format": "normalized", "provenance": "fixture", "observed_at": "2026-09-26T13:15:00Z",
                              "data": {"error_rate": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 0.004), (2, 0.002), (3, 0.003))]},
                                       "p95_ms": {"samples": [{"timestamp": t, "value": v} for t, v in ((1, 1900), (2, 1750), (3, 1820))], "unit": "ms"}},
                              "coverage": {"window_seconds": 900}, "sources": ["envoy metrics"]},
    }
