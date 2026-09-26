"""Deterministic, conservative catalog for production serving plans.

Hardware fit estimates are screening bounds, not a benchmark or a guarantee.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_MODEL_DATA = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "license": "Apache-2.0",
        "parameters_b": 4,
        "max_position_embeddings": 40960,
        "kv_bytes_per_token": 147456,
        "quality_profile": "Smaller general-purpose text-generation candidate with lower memory cost; no task-specific quality claim is made.",
        "sources": [
            "https://huggingface.co/Qwen/Qwen3-4B/commit/1cfa9a7208912126459214e8b04321603b3df60c",
            "https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json",
            "https://github.com/kserve/kserve/blob/v0.20.0/python/storage/README.md",
        ],
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "21073ac5a57f8ac6b159dae129728af51ac707e8",
        "license": "Apache-2.0",
        "parameters_b": 8,
        "max_position_embeddings": 40960,
        # Qwen3-8B has the same 36-layer, 8-KV-head, 128-dimension KV
        # geometry as Qwen3-4B; BF16 K+V is 2 * 2 * 36 * 8 * 128 bytes.
        "kv_bytes_per_token": 147456,
        "quality_profile": "Larger general-purpose text-generation candidate with higher memory cost; no benchmarked quality delta is claimed.",
        "sources": [
            "https://huggingface.co/Qwen/Qwen3-8B/tree/21073ac5a57f8ac6b159dae129728af51ac707e8",
            "https://huggingface.co/Qwen/Qwen3-8B/blob/21073ac5a57f8ac6b159dae129728af51ac707e8/config.json",
        ],
    },
}

_RECIPES = {
    "kserve-llmd-vllm": {
        "stack": "KServe LLMInferenceService + llm-d + vLLM",
        "stack_version": "KServe v0.20.0; llm-d runtime v0.8.0",
        "image": "ghcr.io/llm-d/llm-d-cuda:v0.8.0",
        "model_choices": ["qwen3-4b", "qwen3-8b"],
        "source": "https://github.com/kserve/kserve/blob/v0.20.0/config/llmisvcconfig/config-llm-template.yaml",
        "documentation": [
            {"fact": "KServe v0.20.0 serves LLMInferenceService at serving.kserve.io/v1alpha2; llm-d's reference template uses ghcr.io/llm-d/llm-d-cuda:v0.8.0.", "source": "https://github.com/kserve/kserve/blob/v0.20.0/config/llmisvcconfig/config-llm-template.yaml"},
            {"fact": "KServe Hugging Face model URIs accept hf://org/model:revision; this bundle pins the immutable model commit in the URI and engine arguments.", "source": "https://github.com/kserve/kserve/blob/v0.20.0/python/storage/README.md"},
            {"fact": "KServe's LLM gateway depends on Gateway API and an installed Gateway controller; the bundle emits a Gateway that targets Envoy GatewayClass eg.", "source": "https://github.com/kserve/kserve/blob/v0.20.0/docs/samples/llmisvc/e2e-gpt-oss/README.md"}
        ],
    },
    "dynamo-vllm": {
        "stack": "NVIDIA Dynamo + vLLM",
        "stack_version": "Dynamo v1.5.0 (release commit b83b1d9)",
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "model_choices": ["qwen3-4b", "qwen3-8b"],
        "source": "https://github.com/ai-dynamo/dynamo/blob/v1.5.0/examples/backends/vllm/deploy/agg.yaml",
        "documentation": [
            {"fact": "Dynamo v1.5.0 uses nvidia.com/v1beta1 DynamoGraphDeployment with spec.components list; v1alpha1 spec.services is the deprecated map API.", "source": "https://github.com/ai-dynamo/dynamo/blob/v1.5.0/deploy/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml"},
            {"fact": "The vLLM worker launch module is dynamo.vllm and the model argument is --model; the runtime image uses the Dynamo v1.5.0 tag.", "source": "https://github.com/ai-dynamo/dynamo/blob/v1.5.0/examples/backends/vllm/deploy/agg.yaml"},
            {"fact": "NVIDIA's v1.5.0 Kubernetes quickstart requires Kubernetes v1.30 or later.", "source": "https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart"}
        ],
    },
    "dynamo-sglang": {
        "stack": "NVIDIA Dynamo + SGLang",
        "stack_version": "Dynamo v1.5.0 (release commit b83b1d9)",
        "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.0",
        "model_choices": ["qwen3-4b", "qwen3-8b"],
        "source": "https://github.com/ai-dynamo/dynamo/blob/v1.5.0/examples/backends/sglang/deploy/agg.yaml",
        "documentation": [
            {"fact": "Dynamo v1.5.0 supports the SGLang runtime image and launches workers with module dynamo.sglang and model argument --model-path.", "source": "https://github.com/ai-dynamo/dynamo/blob/v1.5.0/examples/backends/sglang/deploy/agg.yaml"},
            {"fact": "The SGLang worker uses --context-length and --max-running-requests rather than vLLM's --max-model-len and --max-num-seqs flags.", "source": "https://docs.sglang.ai/backend/server_arguments.html"},
            {"fact": "NVIDIA's Dynamo v1.5.0 Kubernetes quickstart requires Kubernetes v1.30 or later.", "source": "https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart"}
        ],
    },
}


def catalog() -> dict[str, Any]:
    """Return the immutable-by-convention recipe and model catalog."""
    return {"recipes": deepcopy(_RECIPES), "models": deepcopy(_MODEL_DATA)}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _gpu_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = []
    for node in inventory.get("nodes", []) or []:
        try:
            count = max(0, int(node.get("gpu_count", 0)))
            vram = float(node.get("vram_gb", 0))
        except (TypeError, ValueError):
            continue
        if count and vram > 0:
            gpus.append({**node, "gpu_count": count, "vram_gb": vram})
    return gpus


def _choose_model(inventory: dict[str, Any], workload: dict[str, Any], recipe: dict[str, Any]) -> str:
    # Select the larger model only when quality is explicitly requested and it fits.
    quality = str(workload.get("quality_tier", "balanced")).lower()
    preferred = "qwen3-8b" if quality in {"high", "quality", "best"} else "qwen3-4b"
    gpus = _gpu_inventory(inventory)
    if preferred == "qwen3-8b" and not any(n["vram_gb"] * n["gpu_count"] >= 28 for n in gpus):
        return "qwen3-4b"
    return preferred if preferred in recipe["model_choices"] else recipe["model_choices"][0]


def select_plan(inventory: dict, workload: dict, recipe_id: str | None = None, model_id: str | None = None) -> dict:
    """Select a stable serving plan and report every unresolved fit assumption.

    The first recipe is the default. A caller may pin one catalog model with
    ``model_id``. Model sizing is a conservative BF16 screening estimate;
    measured load tests remain required before rollout.
    """
    recipe_id = recipe_id or "kserve-llmd-vllm"
    if recipe_id not in _RECIPES:
        raise ValueError(f"Unknown recipe_id: {recipe_id}")
    recipe = _RECIPES[recipe_id]
    model_key_by_id = {m["model_id"]: key for key, m in _MODEL_DATA.items()}
    model_key = model_key_by_id.get(model_id) if model_id else _choose_model(inventory, workload, recipe)
    if model_id and (model_key is None or model_key not in recipe["model_choices"]):
        raise ValueError(f"Model {model_id!r} is not available for recipe {recipe_id!r}")
    model = _MODEL_DATA[model_key]
    blockers: list[str] = []
    assumptions: list[str] = [
        "GPU memory estimate uses BF16 weight size plus 6 GiB runtime reserve and estimated KV cache; validate with the exact engine build and traffic.",
        "One worker replica is planned; request rate and concurrency do not establish throughput without benchmark data.",
        "Container tags are version pinned but not registry digest pinned; verify and record image digests in the target registry before production rollout.",
    ]
    task = str(workload.get("task", "chat")).lower()
    if task in {"embedding", "embeddings", "rerank", "reranking"}:
        blockers.append(f"Catalog model {model['model_id']} is a text-generation model and is not compatible with task={task!r}.")
    context = _positive_int(workload.get("context_tokens"), 4096)
    output = _positive_int(workload.get("output_tokens"), 512)
    requested_context = context + output
    model_max = model["max_position_embeddings"]
    max_model_len = min(requested_context, model_max)
    if requested_context > model_max:
        blockers.append(f"Requested context plus output ({requested_context:,} tokens) exceeds {model['model_id']} limit ({model_max:,}).")
    concurrency = _positive_int(workload.get("concurrency"), 1)
    replicas = 1
    tensor_parallel = 1
    gpus = _gpu_inventory(inventory)
    if not gpus:
        blockers.append("No usable NVIDIA GPU nodes were supplied; GPU fit and placement cannot be validated.")
    else:
        # Only combine GPUs present on one inventory node. The selected tensor
        # parallel factor must evenly partition this model's 8 KV heads.
        weight_gib = model["parameters_b"] * 2 * 1.18
        kv_gib = max_model_len * concurrency * model["kv_bytes_per_token"] / (1024**3)
        runtime_gib = 6.0
        fits: list[tuple[int, float]] = []
        for node in gpus:
            for tp in (1, 2, 4, 8):
                if tp > node["gpu_count"] or (tp > 1 and node["gpu_count"] % tp):
                    continue
                if tp > 1 and str(node.get("interconnect", "")).lower() in {"none", "unknown", ""}:
                    continue
                required_per_gpu = (weight_gib + kv_gib + runtime_gib) / tp
                if required_per_gpu <= node["vram_gb"]:
                    fits.append((tp, required_per_gpu))
                    break
        if fits:
            tensor_parallel = min(tp for tp, _ in fits)
        else:
            max_node_vram = max((n["vram_gb"] * n["gpu_count"] for n in gpus), default=0)
            blockers.append(f"No single inventory node fits the estimated {weight_gib + kv_gib + runtime_gib:.1f} GiB per-replica memory need with a supported tensor-parallel split; largest node aggregate VRAM is {max_node_vram:.1f} GiB.")
        if tensor_parallel > 1:
            assumptions.append(f"Tensor parallelism {tensor_parallel} spans GPUs on one inventoried node; actual fabric bandwidth and engine support must be confirmed.")
    if requested_context > max_model_len:
        assumptions.append("max_model_len is clipped to the model's advertised position limit; requests beyond it must be rejected or truncated upstream.")
    if str(inventory.get("os", "")).lower() and not any(token in str(inventory.get("os", "")).lower() for token in ("linux", "ubuntu", "rhel", "rocky", "debian", "suse", "cos")):
        blockers.append(f"Inventory OS {inventory.get('os')!r} is not recognized as a supported Linux host for these NVIDIA GPU runtime images.")
    if not inventory.get("driver_version"):
        blockers.append("NVIDIA driver_version is missing; host/runtime compatibility cannot be checked.")
    if not inventory.get("cuda_version"):
        blockers.append("CUDA version is missing; host/runtime compatibility cannot be checked.")
    if recipe_id.startswith("dynamo-"):
        try:
            kube_minor = int(str(inventory.get("kubernetes_version", "")).split(".")[1])
            kube_major = int(str(inventory.get("kubernetes_version", "")).split(".")[0].lstrip("v"))
        except (IndexError, ValueError):
            blockers.append("Kubernetes version is missing or malformed; Dynamo's documented v1.30+ prerequisite cannot be checked.")
        else:
            if (kube_major, kube_minor) < (1, 30):
                blockers.append("Dynamo v1.5.0 requires Kubernetes v1.30 or later.")
    licenses = workload.get("allowed_licenses")
    if licenses and model["license"].lower() not in {str(x).lower() for x in licenses}:
        blockers.append(f"Model license {model['license']} is not in workload.allowed_licenses.")
    recipe_copy = deepcopy(recipe)
    alternatives = []
    for candidate_key in recipe["model_choices"]:
        candidate = _MODEL_DATA[candidate_key]
        total_gib = candidate["parameters_b"] * 2 * 1.18 + 6 + max_model_len * concurrency * candidate["kv_bytes_per_token"] / (1024**3)
        best_node = max(gpus, key=lambda n: n["gpu_count"] * n["vram_gb"], default=None)
        alternatives.append({"model_id": candidate["model_id"], "revision": candidate["revision"], "license": candidate["license"], "estimated_total_vram_gib": round(total_gib, 2), "candidate_fit": bool(best_node and total_gib <= best_node["gpu_count"] * best_node["vram_gb"])})
    return {
        "recipe_id": recipe_id,
        "model": {"model_id": model["model_id"], "revision": model["revision"], "license": model["license"]},
        "allocation": {"replicas": replicas, "tensor_parallel": tensor_parallel, "gpus_per_replica": tensor_parallel},
        "config": {"max_model_len": max_model_len, "max_num_seqs": concurrency},
        "blockers": blockers,
        "assumptions": assumptions,
        "alternatives": alternatives,
        "inventory": deepcopy(inventory),
        "workload": deepcopy(workload),
        "versions": {"stack": recipe_copy["stack_version"], "container_image": recipe_copy["image"], "model_revision": model["revision"]},
        "sources": sorted(set([recipe["source"], *model["sources"], "https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart", "https://github.com/kserve/kserve/releases/tag/v0.20.0", "https://huggingface.co/Qwen/Qwen3-4B", "https://huggingface.co/Qwen/Qwen3-8B"])),
        "documented_facts": deepcopy(recipe["documentation"]),
        "quality_profile": model["quality_profile"],
    }
