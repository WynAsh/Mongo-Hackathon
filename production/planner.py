"""Hardware + workload -> serving plan, Kubernetes YAML, and ordered shell commands.

One pure function, ``build_plan``. Nothing here runs a command or touches a
cluster; the operator reviews the output and runs ``install.sh`` themselves.
GPU sizing is a conservative BF16 screening estimate, not a benchmark.
"""
import hashlib
import json
import re
from pathlib import Path

from production.advisor import choose
from production.catalog import MODELS, PREREQS, PROJECTS, RECIPES
from production.contracts import PlanRequest

RUNTIME_RESERVE_GIB = 6.0
WEIGHT_OVERHEAD = 1.18  # BF16 weights plus loader/activation slack
LINUX = ("linux", "ubuntu", "rhel", "rocky", "debian", "suse", "cos")
SCHEMAS = Path(__file__).parent / "assets"


# ---------------------------------------------------------------- sizing

def _kube_version(text):
    match = re.match(r"v?(\d+)\.(\d+)", str(text))
    return (int(match[1]), int(match[2])) if match else None


def _choose_recipe(requested):
    if requested:
        return requested, "Selected by request."
    return "kserve-llmd-vllm", ("Default: Envoy AI Gateway + KServe + llm-d + vLLM covers four of the six "
                                "projects and adds KV-cache-aware routing across replicas.")


def _fit(model, max_len, concurrency, nodes):
    """Smallest tensor-parallel size that fits on some node, and replicas across all nodes."""
    weights = model["parameters_b"] * 2 * WEIGHT_OVERHEAD
    kv = max_len * concurrency * model["kv_bytes_per_token"] / 1024**3
    need = weights + kv + RUNTIME_RESERVE_GIB

    def fits(node, tp):
        if tp > node["gpu_count"] or node["gpu_count"] % tp:
            return False
        if tp > 1 and node["interconnect"].lower() in {"", "none", "unknown"}:
            return False
        return need / tp <= node["vram_gb"]

    for tp in (1, 2, 4, 8):
        used = [n for n in nodes if fits(n, tp)]
        if used:
            placement = {n["name"]: n["gpu_count"] // tp for n in used}
            return need, tp, placement
    return need, None, {}


def size(inventory, workload, recipe_id=None, model_id=None):
    recipe_id, recipe_reason = _choose_recipe(recipe_id)
    recipe = RECIPES[recipe_id]
    blockers, notes = [], []
    if model_id:
        model_key = next((k for k, m in MODELS.items() if m["model_id"] == model_id), None)
        if model_key is None:
            raise ValueError(f"Unknown model_id: {model_id!r}")
    else:
        model_key = "qwen3-8b" if workload["quality"] == "high" else "qwen3-4b"
    model = MODELS[model_key]

    max_len = workload["context_tokens"] + workload["output_tokens"]
    if max_len > model["max_position_embeddings"]:
        blockers.append(f"context + output ({max_len:,}) exceeds {model['model_id']} limit "
                        f"({model['max_position_embeddings']:,}).")
        max_len = model["max_position_embeddings"]
    if workload["task"].lower() in {"embedding", "embeddings", "rerank", "reranking"}:
        blockers.append(f"{model['model_id']} is a text-generation model, not an {workload['task']} model.")
    allowed = {x.lower() for x in workload["allowed_licenses"]}
    if allowed and model["license"].lower() not in allowed:
        blockers.append(f"Model license {model['license']} is not in allowed_licenses.")

    gpu_nodes = [n for n in inventory["nodes"] if n["gpu_count"] and n["vram_gb"]]
    need, tp, placement = _fit(model, max_len, workload["concurrency"], gpu_nodes)
    if tp is None and not model_id and model_key == "qwen3-8b":
        model = MODELS["qwen3-4b"]
        need, tp, placement = _fit(model, max_len, workload["concurrency"], gpu_nodes)
        notes.append("Qwen3-8B does not fit this hardware; fell back to Qwen3-4B.")
    if not gpu_nodes:
        blockers.append("No GPU nodes in the inventory.")
    elif tp is None:
        blockers.append(f"No node fits the estimated {need:.1f} GiB per replica with TP 1/2/4/8.")

    os_name = inventory["os"].lower()
    if os_name != "unknown" and not any(t in os_name for t in LINUX):
        blockers.append(f"OS {inventory['os']!r} is not a supported Linux for NVIDIA runtime images.")
    kube = _kube_version(inventory["kubernetes_version"])
    if recipe["min_kubernetes"] and kube and kube < tuple(recipe["min_kubernetes"]):
        blockers.append(f"{recipe['stack']} needs Kubernetes "
                        f"v{'.'.join(map(str, recipe['min_kubernetes']))}+, found {inventory['kubernetes_version']}.")
    if kube is None:
        notes.append("Kubernetes version unknown; the preflight step prints it.")
    notes.append(f"Estimated {need:.1f} GiB per replica (weights + KV for {workload['concurrency']} "
                 f"sequences x {max_len:,} tokens + {RUNTIME_RESERVE_GIB:.0f} GiB reserve). Load-test before production.")
    notes.append("Chart and image versions are pinned in production/catalog.py; confirm them against upstream releases.")

    replicas = sum(placement.values()) or 1
    tp = tp or 1
    return {
        "recipe_id": recipe_id,
        "recipe_reason": recipe_reason,
        "stack": recipe["stack"],
        "image": recipe["image"],
        "namespace": recipe["namespace"],
        "model": {k: model[k] for k in ("model_id", "revision", "license")},
        "allocation": {"tensor_parallel": tp, "replicas": replicas, "gpus_per_replica": tp,
                       "placement": placement, "estimated_gib_per_replica": round(need, 1)},
        "engine": {"max_model_len": max_len, "max_num_seqs": workload["concurrency"]},
        "blockers": blockers,
        "notes": notes,
    }


# ---------------------------------------------------------------- YAML

def _q(value):
    return json.dumps(str(value))


def _name(model_id):
    name = re.sub(r"[^a-z0-9-]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return (name or "model")[:50].rstrip("-")


def _kserve_yaml(p):
    m, e, a = p["model"], p["engine"], p["allocation"]
    serve = (f"exec vllm serve /mnt/models --served-model-name {_name(m['model_id'])} --port 8000 "
             f"--tensor-parallel-size {a['tensor_parallel']} \"$@\"")
    return f'''apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: {_name(m["model_id"])}
  namespace: {p["namespace"]}
spec:
  model:
    name: {_q(m["model_id"])}
    uri: {_q(f"hf://{m['model_id']}:{m['revision']}")}
  replicas: {a["replicas"]}
  router:
    gateway:
      refs:
        - group: gateway.networking.k8s.io
          kind: Gateway
          name: model-gateway
          namespace: {p["namespace"]}
    route: {{}}
    scheduler: {{}}
  parallelism:
    tensor: {a["tensor_parallel"]}
  template:
    containers:
      - name: main
        image: {p["image"]}
        imagePullPolicy: IfNotPresent
        command:
          - /bin/bash
          - -c
          - {_q(serve)}
          - --
        args:
          - --max-model-len
          - {_q(e["max_model_len"])}
          - --max-num-seqs
          - {_q(e["max_num_seqs"])}
        envFrom:
          - secretRef:
              name: hf-token-secret
        resources:
          limits:
            nvidia.com/gpu: {_q(a["gpus_per_replica"])}
          requests:
            nvidia.com/gpu: {_q(a["gpus_per_replica"])}
{_probes_yaml(p, 8)}'''


def _probes_yaml(p, indent):
    """Probe tuning from a reliability change (vLLM /health on :8000). Provisioning plans set none."""
    probes = p.get("probes") or {}
    lines = []
    if {"readiness_initial_delay_seconds", "readiness_timeout_seconds"} & set(probes):
        lines += ["readinessProbe:", "  httpGet:", "    path: /health", "    port: 8000"]
        if "readiness_initial_delay_seconds" in probes:
            lines.append(f"  initialDelaySeconds: {probes['readiness_initial_delay_seconds']}")
        if "readiness_timeout_seconds" in probes:
            lines.append(f"  timeoutSeconds: {probes['readiness_timeout_seconds']}")
    if "startup_probe_failure_threshold" in probes:
        lines += ["startupProbe:", "  httpGet:", "    path: /health", "    port: 8000",
                  "  periodSeconds: 10", f"  failureThreshold: {probes['startup_probe_failure_threshold']}"]
    return "".join(" " * indent + line + "\n" for line in lines)


def _dynamo_worker(p, name, kind, replicas, extra_args):
    m, e, a = p["model"], p["engine"], p["allocation"]
    sglang = p["recipe_id"] == "dynamo-sglang"
    flags = (("--model-path", "--context-length", "--max-running-requests", "--tp-size") if sglang
             else ("--model", "--max-model-len", "--max-num-seqs", "--tensor-parallel-size"))
    extra = "".join(f"                - {arg}\n" for arg in extra_args)
    return f'''    - name: {name}
      type: {kind}
      replicas: {replicas}
      podTemplate:
        spec:
          containers:
            - name: main
              image: {p["image"]}
              command: [python3, -m, {"dynamo.sglang" if sglang else "dynamo.vllm"}]
              args:
                - {flags[0]}
                - {_q(m["model_id"])}
                - --revision
                - {_q(m["revision"])}
                - {flags[1]}
                - {_q(e["max_model_len"])}
                - {flags[2]}
                - {_q(e["max_num_seqs"])}
                - {flags[3]}
                - {_q(a["tensor_parallel"])}
{extra}              envFrom:
                - secretRef:
                    name: hf-token-secret
              resources:
                limits:
                  nvidia.com/gpu: {_q(a["gpus_per_replica"])}
                requests:
                  nvidia.com/gpu: {_q(a["gpus_per_replica"])}
'''


# Disaggregated prefill/decode worker flags; set only by an evolution campaign.
# Verify against the pinned Dynamo release's disaggregation guide before rollout.
PD_FLAGS = {"dynamo-vllm": {"prefill": ["--is-prefill-worker"], "decode": []},
            "dynamo-sglang": {"prefill": ["--disaggregation-mode", "prefill", "--disaggregation-transfer-backend", "nixl"],
                              "decode": ["--disaggregation-mode", "decode", "--disaggregation-transfer-backend", "nixl"]}}


def _dynamo_yaml(p):
    m, a = p["model"], p["allocation"]
    sglang = p["recipe_id"] == "dynamo-sglang"
    pd = p.get("disaggregation")
    if pd:
        flags = PD_FLAGS[p["recipe_id"]]
        workers = (_dynamo_worker(p, "prefill", "prefill", pd["prefill_replicas"], flags["prefill"]) +
                   _dynamo_worker(p, "decode", "decode", pd["decode_replicas"], flags["decode"]))
    else:
        workers = _dynamo_worker(p, "worker", "worker", a["replicas"], [])
    return f'''apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: {_name(m["model_id"])}
  namespace: {p["namespace"]}
spec:
  backendFramework: {"sglang" if sglang else "vllm"}
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: {p["image"]}
              command: [python3, -m, dynamo.frontend]
              args: ["--http-port", "8000"]
              envFrom:
                - secretRef:
                    name: hf-token-secret
{workers}'''


def _gatewayclass_yaml():
    return '''apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: eg
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
'''


def _gateway_yaml(p):
    return f'''apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: model-gateway
  namespace: {p["namespace"]}
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


def render_files(p):
    if p["recipe_id"] == "kserve-llmd-vllm":
        return {"k8s/10-gatewayclass.yaml": _gatewayclass_yaml(),
                "k8s/20-gateway.yaml": _gateway_yaml(p),
                "k8s/30-model.yaml": _kserve_yaml(p)}
    return {"k8s/30-model.yaml": _dynamo_yaml(p)}


def validate(files, recipe_id):
    """Parse each YAML file and check it against the pinned upstream CRD schema projection."""
    import yaml
    from jsonschema import Draft202012Validator
    schemas = {"k8s/20-gateway.yaml": "gateway-api-v1.5.0-gateway.schema.json",
               "k8s/30-model.yaml": ("kserve-llmisvc-v0.20.0.schema.json" if recipe_id == "kserve-llmd-vllm"
                                     else "dynamo-dgd-v1.5.0.schema.json")}
    checks = []
    for path, text in files.items():
        try:
            obj = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            checks.append({"file": path, "ok": False, "detail": f"YAML parse failed: {exc}"})
            continue
        if path not in schemas:
            checks.append({"file": path, "ok": True, "detail": "YAML parses."})
            continue
        schema = json.loads((SCHEMAS / schemas[path]).read_text())
        error = next(iter(Draft202012Validator(schema).iter_errors(obj)), None)
        checks.append({"file": path, "ok": error is None,
                       "detail": f"Matches {schemas[path]}." if error is None else
                       f"{schemas[path]}: {'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"})
    return checks


# ---------------------------------------------------------------- commands

GPU_MISSING = ("! kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\\.com/gpu}' "
               "| grep -q '[1-9]'")


def steps(p):
    recipe = RECIPES[p["recipe_id"]]
    ns, name = p["namespace"], _name(p["model"]["model_id"])
    out = [{"title": "Preflight", "why": "Confirm cluster access, Kubernetes version, and schedulable GPUs.",
            "commands": ["kubectl version",
                         "kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.allocatable.nvidia\\.com/gpu'"]}]
    for key in recipe["prereqs"]:
        item = PREREQS[key]
        out.append({"title": f"Provision {item['name']}", "why": item["why"], "commands": item["commands"],
                    "when": GPU_MISSING if key == "gpu-operator" else None})
    for key in recipe["projects"]:
        item = PROJECTS[key]
        if item["commands"]:
            out.append({"title": f"Install {item['name']} {item['version']}", "why": item["role"],
                        "commands": item["commands"]})
    out.append({"title": "Configure namespace and secrets",
                "why": "HF_TOKEN is read from your shell and never written to disk.",
                "commands": [f"kubectl create namespace {ns} --dry-run=client -o yaml | kubectl apply -f -",
                             f'kubectl create secret generic hf-token-secret -n {ns} --from-literal=HF_TOKEN="${{HF_TOKEN:-}}" '
                             "--dry-run=client -o yaml | kubectl apply -f -"]})
    kind = "llmisvc" if p["recipe_id"] == "kserve-llmd-vllm" else "dgd"
    out.append({"title": "Deploy", "why": "Server-side dry run first, then apply and wait for readiness.",
                "commands": ["kubectl apply --dry-run=server -f k8s/",
                             "kubectl apply -f k8s/",
                             f"kubectl wait --for=condition=Ready --timeout=900s {kind}/{name} -n {ns}"]})
    body = json.dumps({"model": name if kind == "llmisvc" else p["model"]["model_id"],
                       "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16})
    if kind == "llmisvc":
        smoke = [f"URL=$(kubectl get llmisvc {name} -n {ns} -o jsonpath='{{.status.url}}')",
                 f"curl -sf \"$URL/v1/chat/completions\" -H 'Content-Type: application/json' -d '{body}'"]
    else:
        smoke = [f"kubectl port-forward -n {ns} svc/{name}-frontend 8000:8000 >/dev/null & PF=$!",
                 "sleep 3",
                 f"curl -sf http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{body}'",
                 "kill $PF"]
    out.append({"title": "Smoke test", "why": "One OpenAI-compatible chat request.", "commands": smoke})
    return out


def rollback(p):
    return ["kubectl delete -f k8s/30-model.yaml"]


def install_script(p, step_list):
    lines = ["#!/usr/bin/env bash",
             f"# {p['stack']} serving {p['model']['model_id']}@{p['model']['revision'][:12]}",
             "# Review PLAN.md first. Rollback: " + "; ".join(rollback(p)),
             "set -euo pipefail", 'cd "$(dirname "$0")"']
    for i, step in enumerate(step_list, 1):
        lines += ["", f"echo '==> {i}. {step['title']}'", f"# {step['why']}"]
        if step.get("when"):
            lines += [f"if {step['when']}; then", *(f"  {c}" for c in step["commands"]), "fi"]
        else:
            lines += step["commands"]
    return "\n".join(lines) + "\n"


def plan_markdown(p, step_list, checks):
    a, m, e = p["allocation"], p["model"], p["engine"]
    lines = [f"# Plan: {p['stack']}", "",
             f"- Model: `{m['model_id']}` @ `{m['revision']}` ({m['license']})",
             f"- Image: `{p['image']}`",
             f"- Topology: {a['replicas']} replica(s) x TP {a['tensor_parallel']} "
             f"({', '.join(f'{k}: {v}' for k, v in a['placement'].items()) or 'no placement'})",
             f"- Engine: max_model_len={e['max_model_len']}, max_num_seqs={e['max_num_seqs']}",
             f"- Decided by: {p['decided_by']}", "", "## Why this stack", "",
             *(f"- Trait: {t}" for t in p["traits"]), *([""] if p["traits"] else []), p["recipe_reason"],
             *(f"- Tradeoff: {t}" for t in p["tradeoffs"]), "", "## Blockers", ""]
    lines += [f"- {b}" for b in p["blockers"]] or ["- None"]
    lines += ["", "## Notes", "", *(f"- {n}" for n in p["notes"]), "", "## Steps", ""]
    for i, step in enumerate(step_list, 1):
        lines.append(f"{i}. **{step['title']}**: {step['why']}")
    lines += ["", "## Validation", "", *(f"- {'ok' if c['ok'] else 'FAIL'} `{c['file']}`: {c['detail']}" for c in checks)]
    lines += ["", "## Rollback", "", *(f"    {c}" for c in rollback(p))]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- entry point

def _decide(req, advisor):
    """Size the stack. Without an explicit recipe, an advisor may pick among all sized candidates."""
    inv, wl, model_id = req["inventory"], req["workload"], req["model_id"]
    if req["recipe_id"] or advisor is None:
        p = size(inv, wl, req["recipe_id"], model_id)
        p["decided_by"] = "request" if req["recipe_id"] else "rules"
        p["tradeoffs"], p["traits"] = [], []
        return p
    candidates = {r: size(inv, wl, r, model_id) for r in RECIPES}
    summary = [{"recipe_id": r, "stack": c["stack"], "best_for": RECIPES[r]["best_for"],
                **{k: c[k] for k in ("model", "allocation", "engine", "blockers")}}
               for r, c in candidates.items()]
    choice = choose(req, summary, advisor)
    p = candidates[choice["recipe_id"]]
    p["recipe_reason"] = choice["explanation"] or (
        _choose_recipe(None)[1] if choice["recipe_id"] == "kserve-llmd-vllm"
        else "Rules: the default stack has blockers; this is the first one without.")
    p["decided_by"], p["tradeoffs"], p["traits"] = choice["decided_by"], choice["tradeoffs"], choice["traits"]
    if choice.get("fallback_reason"):
        p["notes"].append(f"LLM advisor unavailable, used rules ({choice['fallback_reason']}).")
    return p


def build_plan(request, advisor=None) -> dict:
    """``advisor`` (e.g. ``production.advisor.llm_advisor``) picks the stack when none is requested."""
    req = PlanRequest.model_validate(request).model_dump()
    p = _decide(req, advisor)
    files = render_files(p)
    checks = validate(files, p["recipe_id"])
    step_list = steps(p)
    files["install.sh"] = install_script(p, step_list)
    files["PLAN.md"] = plan_markdown(p, step_list, checks)
    recipe = RECIPES[p["recipe_id"]]
    projects = [{"id": k, **{f: v[f] for f in ("name", "repo", "version", "role")},
                 "used": k in recipe["projects"]} for k, v in PROJECTS.items()]
    ready = not p["blockers"] and all(c["ok"] for c in checks)
    plan_id = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:16]
    return {"plan_id": plan_id, "status": "ready" if ready else "blocked", "input": req, "plan": p,
            "projects": projects, "steps": step_list, "rollback": rollback(p),
            "validation": checks, "files": files}
