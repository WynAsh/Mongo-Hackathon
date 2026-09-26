"""Deterministic production bundle renderer with honest offline validation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .catalog import catalog as get_catalog


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return (name or "model")[:50].rstrip("-")


def _kserve_manifest(plan: dict, recipe: dict) -> str:
    model = plan["model"]
    config = plan["config"]
    name = _safe_name(model["model_id"].split("/")[-1])
    return f'''apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: {name}
  namespace: kserve-models
spec:
  model:
    name: {_yaml_scalar(model["model_id"])}
    uri: {_yaml_scalar("hf://" + model["model_id"] + ":" + model["revision"])}
  replicas: {plan["allocation"]["replicas"]}
  router:
    gateway:
      refs:
        - group: gateway.networking.k8s.io
          kind: Gateway
          name: model-gateway
          namespace: kserve-models
    route: {{}}
    scheduler: {{}}
  parallelism:
    tensor: {plan["allocation"]["tensor_parallel"]}
  template:
    containers:
      - name: main
        image: {recipe["image"]}
        imagePullPolicy: IfNotPresent
        args:
          - --max-model-len
          - {_yaml_scalar(str(config["max_model_len"]))}
          - --max-num-seqs
          - {_yaml_scalar(str(config["max_num_seqs"]))}
        command:
          - /bin/bash
          - -c
          - {_yaml_scalar('exec vllm serve /mnt/models --served-model-name ' + model["model_id"].split("/")[-1] + ' --port 8000 --tensor-parallel-size ' + str(plan["allocation"]["tensor_parallel"]) + ' "$@"')}
          - --
        envFrom:
          - secretRef:
              name: hf-token-secret
        resources:
          limits:
            nvidia.com/gpu: {_yaml_scalar(str(plan["allocation"]["gpus_per_replica"]))}
          requests:
            nvidia.com/gpu: {_yaml_scalar(str(plan["allocation"]["gpus_per_replica"]))}
'''


def _dynamo_manifest(plan: dict, recipe_id: str, recipe: dict) -> str:
    model = plan["model"]
    config = plan["config"]
    backend = "sglang" if recipe_id == "dynamo-sglang" else "vllm"
    runtime_module = "dynamo.sglang" if backend == "sglang" else "dynamo.vllm"
    model_flag = "--model-path" if backend == "sglang" else "--model"
    context_flag = "--context-length" if backend == "sglang" else "--max-model-len"
    concurrency_flag = "--max-running-requests" if backend == "sglang" else "--max-num-seqs"
    name = _safe_name(model["model_id"].split("/")[-1])
    return f'''apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: {name}
  namespace: dynamo
spec:
  backendFramework: {backend}
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: {recipe["image"]}
              command:
                - python3
                - -m
                - dynamo.frontend
              args:
                - --http-port
                - "8000"
              envFrom:
                - secretRef:
                    name: hf-token-secret
    - name: worker
      type: worker
      replicas: {plan["allocation"]["replicas"]}
      podTemplate:
        spec:
          containers:
            - name: main
              image: {recipe["image"]}
              command:
                - python3
                - -m
                - {runtime_module}
              args:
                - {model_flag}
                - {_yaml_scalar(model["model_id"])}
                - --revision
                - {_yaml_scalar(model["revision"])}
                - {context_flag}
                - {_yaml_scalar(str(config["max_model_len"]))}
                - {concurrency_flag}
                - {_yaml_scalar(str(config["max_num_seqs"]))}
                - {('--tp-size' if backend == 'sglang' else '--tensor-parallel-size')}
                - {_yaml_scalar(str(plan["allocation"]["tensor_parallel"]))}
              envFrom:
                - secretRef:
                    name: hf-token-secret
              resources:
                limits:
                  nvidia.com/gpu: {_yaml_scalar(str(plan["allocation"]["gpus_per_replica"]))}
                requests:
                  nvidia.com/gpu: {_yaml_scalar(str(plan["allocation"]["gpus_per_replica"]))}
'''


def _gateway_manifest() -> str:
    return '''apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: model-gateway
  namespace: kserve-models
spec:
  gatewayClassName: eg
  listeners:
    - name: http
      port: 80
      protocol: HTTP
      allowedRoutes:
        namespaces:
          from: Same
'''


def _manifest_for_plan(plan: dict) -> str:
    recipes = get_catalog()["recipes"]
    recipe_id = plan.get("recipe_id")
    if recipe_id not in recipes:
        raise ValueError(f"Unsupported recipe_id in previous plan: {recipe_id!r}")
    if recipe_id == "kserve-llmd-vllm":
        return _kserve_manifest(plan, recipes[recipe_id])
    return _dynamo_manifest(plan, recipe_id, recipes[recipe_id])


def _change_summary(previous: dict, current: dict) -> str:
    lines = ["# Plan change summary", "", "Fields that differ from the supplied previous plan:", ""]
    for key in ("recipe_id", "model", "allocation", "config"):
        old = previous.get(key)
        new = current.get(key)
        if old != new:
            lines.extend([f"- `{key}`: `{json.dumps(old, sort_keys=True)}` → `{json.dumps(new, sort_keys=True)}`"])
    if len(lines) == 4:
        lines.append("- No compared fields changed.")
    return "\n".join(lines) + "\n"


def _commands(plan: dict, recipe_id: str, recipe: dict) -> str:
    model = plan["model"]["model_id"]
    if recipe_id == "kserve-llmd-vllm":
        prereq = [
            "# Prerequisite: KServe v0.20.0, Gateway API v1.5.0 standard CRDs, Envoy Gateway 1.5.0+ with GatewayClass 'eg', and the llm-d/EPP controller installed.",
            "kubectl get crd llminferenceservices.serving.kserve.io",
            "kubectl get gatewayclass eg",
            "kubectl create namespace kserve-models --dry-run=client -o yaml | kubectl apply -f -",
        ]
        namespace = "kserve-models"
        name = _safe_name(model.split("/")[-1])
    else:
        prereq = [
            "# Install exact Dynamo CRDs and platform version from the official NVIDIA chart repository.",
            'helm upgrade --install dynamo-crds oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds --version 1.5.0 --namespace default',
            'helm upgrade --install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version 1.5.0 --namespace dynamo-system --create-namespace --wait',
            "kubectl get crd dynamographdeployments.nvidia.com",
            "kubectl create namespace dynamo --dry-run=client -o yaml | kubectl apply -f -",
        ]
        namespace = "dynamo"
        name = _safe_name(model.split("/")[-1])
    commands = prereq + [
        "# Supply HF_TOKEN in the shell only when required by your model access policy; never commit the token.",
        f'kubectl create secret generic hf-token-secret -n {namespace} --from-literal=HF_TOKEN="${{HF_TOKEN}}" --dry-run=client -o yaml | kubectl apply -f -',
        "# Review k8s/deployment.yaml and the blockers in VALIDATION.json before applying.",
        *((["kubectl apply -f k8s/gateway.yaml"] if recipe_id == "kserve-llmd-vllm" else [])),
        "kubectl apply -f k8s/deployment.yaml",
        f"kubectl wait --for=condition=Ready --timeout=900s {('llmisvc' if recipe_id == 'kserve-llmd-vllm' else 'dgd')}/{name} -n {namespace}",
        "# After readiness, port-forward the service and run your approved smoke and load checks.",
        "# Roll back by applying k8s/rollback.yaml when it is present; otherwise restore the saved prior manifest.",
    ]
    return "\n".join(commands) + "\n"


def _runbook(plan: dict, recipe_id: str, previous_present: bool) -> str:
    rollback = "A prior bundle was supplied; `k8s/rollback.yaml` contains its deployment manifest. Apply it to restore the previous desired state." if previous_present else "No prior manifest was supplied. Before rollout, save the current known-good manifest; rollback is `kubectl apply -f <saved-known-good-manifest>`."
    return f'''# Production deployment runbook

This is a rendered deployment candidate for `{recipe_id}` serving `{plan['model']['model_id']}` at immutable model revision `{plan['model']['revision']}`. It does not deploy anything.

## Before rollout

1. Review `VALIDATION.json`; the bundle's offline validation checks its pinned schema projections, while target-cluster admission and image digest resolution still need completion.
2. Confirm the installed CRDs and operator versions match `VERSION_LOCK.json`.
3. Verify the image tags resolve to approved registry digests, then record those digests in your release record.
4. Confirm the cluster has enough same-node NVIDIA GPU memory for the model, context, and concurrency. The catalog estimate is a screening heuristic, not benchmark evidence.
5. Confirm the Apache-2.0 model license is acceptable for your use and review workload output quality against your acceptance set.
6. Fill `HF_TOKEN` only in a protected shell if access policy requires it. No credential is stored in this bundle.

## Ordered rollout

Use the commands in `commands.txt` in order. Inspect each generated resource and wait for readiness before routing production traffic. Start with low or shadow traffic, compare quality, latency, errors, and GPU memory to the current service, then increase traffic only under your normal change process.

## Rollback

{rollback}

For KServe, restore the prior `LLMInferenceService` desired state. For Dynamo, restore the prior `DynamoGraphDeployment`. Verify readiness and route traffic back only after health and smoke checks pass.
'''


def _yaml_check(text: str) -> tuple[bool, str]:
    try:
        import yaml  # PyYAML is supplied by the production application dependencies.
    except ImportError:
        return False, "PyYAML unavailable; YAML syntax was not checked."
    try:
        docs = list(yaml.safe_load_all(text))
        if not docs or not isinstance(docs[0], dict):
            return False, "YAML parsed but did not contain a Kubernetes object mapping."
        if not {"apiVersion", "kind", "metadata", "spec"}.issubset(docs[0]):
            return False, "YAML parsed but is missing a Kubernetes object's standard top-level fields."
        return True, "YAML parses and contains standard Kubernetes object fields."
    except Exception as exc:  # parser exceptions vary across PyYAML versions
        return False, f"YAML parse failed: {exc}"


def _schema_check(text: str, schema_name: str) -> tuple[bool, str, str | None]:
    try:
        import yaml
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        return False, f"Schema validator dependency unavailable: {exc}", None
    schema_path = Path(__file__).parent / "assets" / schema_name
    try:
        schema_bytes = schema_path.read_bytes()
        schema = json.loads(schema_bytes)
        obj = yaml.safe_load(text)
        errors = sorted(Draft202012Validator(schema).iter_errors(obj), key=lambda e: list(map(str, e.absolute_path)))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return False, f"Could not load or apply pinned schema projection {schema_name}: {exc}", None
    if errors:
        error = errors[0]
        path = ".".join(str(bit) for bit in error.absolute_path) or "$"
        return False, f"{schema_name} rejected {path}: {error.message}", hashlib.sha256(schema_bytes).hexdigest()
    return True, f"Manifest fields passed pinned upstream schema projection {schema_name}; target-cluster admission remains to be checked.", hashlib.sha256(schema_bytes).hexdigest()


def render_bundle(plan: dict, previous: dict | None = None) -> dict:
    """Render deterministic text files plus a schema-honest validation report."""
    catalog_data = get_catalog()
    recipe_id = plan.get("recipe_id")
    if recipe_id not in catalog_data["recipes"]:
        raise ValueError(f"Unsupported recipe_id: {recipe_id!r}")
    recipe = catalog_data["recipes"][recipe_id]
    manifest = _manifest_for_plan(plan)

    files: dict[str, str] = {
        "k8s/deployment.yaml": manifest,
        ".env.example": "# Copy into a protected environment; never commit a real token.\nHF_TOKEN=\n",
        "commands.txt": _commands(plan, recipe_id, recipe),
        "RUNBOOK.md": _runbook(plan, recipe_id, previous is not None),
    }
    if recipe_id == "kserve-llmd-vllm":
        files["k8s/gateway.yaml"] = _gateway_manifest()
    prev_manifest = _manifest_for_plan(previous) if previous and "recipe_id" in previous else None
    if isinstance(prev_manifest, str) and prev_manifest.strip():
        files["k8s/rollback.yaml"] = prev_manifest
        files["CHANGELOG.md"] = _change_summary(previous, plan)
    schema_name = "kserve-llmisvc-v0.20.0.schema.json" if recipe_id == "kserve-llmd-vllm" else "dynamo-dgd-v1.5.0.schema.json"
    manifest_ok, manifest_check = _yaml_check(manifest)
    schema_ok, schema_detail, schema_digest = _schema_check(manifest, schema_name)
    gateway_ok = True
    gateway_detail = "Not applicable."
    gateway_digest = None
    if recipe_id == "kserve-llmd-vllm":
        gateway_ok, gateway_detail, gateway_digest = _schema_check(_gateway_manifest(), "gateway-api-v1.5.0-gateway.schema.json")
    schemas_ok = manifest_ok and schema_ok and gateway_ok
    source_set = sorted(set(plan.get("sources", []) + [recipe["source"]]))
    manifest_hash = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    lock = {
        "recipe_id": recipe_id,
        "stack": recipe["stack"],
        "stack_version": recipe["stack_version"],
        "container_image": recipe["image"],
        "container_digest": None,
        "container_digest_status": "not_resolved; verify against approved registry before rollout",
        "model": plan["model"],
        "manifest_sha256": manifest_hash,
        "manifest_path": "k8s/deployment.yaml",
        "artifact_sha256": {path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in sorted(files.items()) if path.endswith((".yaml", ".yml"))},
        "previous_manifest_sha256": hashlib.sha256(prev_manifest.encode("utf-8")).hexdigest() if isinstance(prev_manifest, str) else None,
        "schema_projection": {"file": schema_name, "sha256": schema_digest, "upstream": ("https://github.com/kserve/kserve/blob/v0.20.0/config/crd/full/llmisvc/serving.kserve.io_llminferenceservices.yaml" if recipe_id == "kserve-llmd-vllm" else "https://github.com/ai-dynamo/dynamo/blob/v1.5.0/deploy/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml")},
        "gateway_schema_projection": ({"file": "gateway-api-v1.5.0-gateway.schema.json", "sha256": gateway_digest, "upstream": "https://github.com/kubernetes-sigs/gateway-api/blob/v1.5.0/config/crd/standard/gateway.networking.k8s.io_gateways.yaml"} if recipe_id == "kserve-llmd-vllm" else None),
        "sources": source_set,
    }
    files["VERSION_LOCK.json"] = _json(lock)
    checks = [
        {"name": "manifest-yaml-syntax", "status": "passed" if manifest_ok else "failed", "detail": manifest_check},
        {"name": "upstream-crd-schema-projection", "status": "passed" if schema_ok else "failed", "detail": schema_detail},
        {"name": "gateway-api-schema-projection", "status": ("passed" if gateway_ok else "failed") if recipe_id == "kserve-llmd-vllm" else "not_applicable", "detail": gateway_detail},
        {"name": "target-cluster-admission", "status": "pending", "detail": "Run kubectl apply --dry-run=server against the target cluster after confirming its installed CRD versions."},
        {"name": "container-digest", "status": "skipped", "detail": "Image tag is version pinned; registry digest has not been resolved or verified."},
        {"name": "model-fit", "status": "passed" if not plan.get("blockers") else "failed", "detail": "; ".join(plan.get("blockers", [])) or "No catalog fit blockers. Capacity remains a heuristic and needs a workload benchmark."},
    ]
    blockers = list(plan.get("blockers", []))
    if not manifest_ok:
        blockers.append(manifest_check)
    if not schema_ok:
        blockers.append(schema_detail)
    if not gateway_ok:
        blockers.append(gateway_detail)
    blockers.append("Target-cluster server-side dry-run is pending; the target cluster must have the pinned CRDs and GatewayClass installed.")
    blockers.append("Container image digest is unresolved; resolve it in the target registry before production rollout.")
    if not prev_manifest:
        blockers.append("No previous deployment manifest was supplied; save a known-good manifest before rollout for a concrete rollback artifact.")
    status = "validation_failed" if not manifest_ok or not schemas_ok or plan.get("blockers") else "offline_validated"
    files["VALIDATION.json"] = _json({"status": status, "checks": checks, "blockers": blockers})
    return {"files": files, "validation": {"status": status, "checks": checks, "blockers": blockers}, "sources": source_set}
