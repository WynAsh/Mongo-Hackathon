"""Pinned models, the six open-source projects, and the recipes that combine them.

Every version and install command lives here so a reviewer can check them in
one place. Verify them against upstream release notes before a real rollout.
"""
from copy import deepcopy


MODELS = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "license": "Apache-2.0",
        "parameters_b": 4,
        "max_position_embeddings": 40960,
        # BF16 K+V: 2 * 2 bytes * 36 layers * 8 KV heads * 128 dims.
        "kv_bytes_per_token": 147456,
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "21073ac5a57f8ac6b159dae129728af51ac707e8",
        "license": "Apache-2.0",
        "parameters_b": 8,
        "max_position_embeddings": 40960,
        "kv_bytes_per_token": 147456,
    },
}

# Cluster add-ons the six projects depend on. Not part of the six themselves.
PREREQS = {
    "gpu-operator": {
        "name": "NVIDIA GPU Operator",
        "why": "Exposes nvidia.com/gpu to the scheduler. Skip if the preflight already shows allocatable GPUs.",
        "commands": [
            "helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update",
            "helm upgrade --install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace --wait",
        ],
    },
    "cert-manager": {
        "name": "cert-manager",
        "why": "KServe's webhooks need TLS certificates.",
        "commands": [
            "helm upgrade --install cert-manager oci://quay.io/jetstack/charts/cert-manager -n cert-manager --create-namespace --set crds.enabled=true --wait",
        ],
    },
    "gateway-api": {
        "name": "Gateway API + Inference Extension CRDs",
        "why": "Gateway/HTTPRoute for Envoy, InferencePool for the llm-d scheduler.",
        "commands": [
            "kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.0/standard-install.yaml",
            "kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.0/manifests.yaml",
        ],
    },
}

# The six projects. `commands` is empty when the project ships inside a
# container image rather than as a cluster install.
PROJECTS = {
    "envoy-ai-gateway": {
        "name": "Envoy AI Gateway",
        "repo": "https://github.com/envoyproxy/ai-gateway",
        "version": "v0.4.0 (on Envoy Gateway v1.5.0)",
        "role": "OpenAI-compatible ingress and routing in front of the model",
        "commands": [
            "helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version v1.5.0 -n envoy-gateway-system --create-namespace --wait",
            "helm upgrade --install aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm --version v0.4.0 -n envoy-ai-gateway-system --create-namespace",
            "helm upgrade --install aieg oci://docker.io/envoyproxy/ai-gateway-helm --version v0.4.0 -n envoy-ai-gateway-system --create-namespace --wait",
        ],
    },
    "kserve": {
        "name": "KServe",
        "repo": "https://github.com/kserve/kserve",
        "version": "v0.20.0",
        "role": "LLMInferenceService controller that creates the model pods, route, and scheduler",
        "commands": [
            "helm upgrade --install kserve-llmisvc-crd oci://ghcr.io/kserve/charts/kserve-llmisvc-crd --version v0.20.0 -n kserve --create-namespace",
            "helm upgrade --install kserve-llmisvc oci://ghcr.io/kserve/charts/kserve-llmisvc-resources --version v0.20.0 -n kserve --wait",
        ],
    },
    "llm-d": {
        "name": "llm-d",
        "repo": "https://github.com/llm-d/llm-d",
        "version": "v0.8.0",
        "role": "KV-cache-aware scheduler (EPP) and the vLLM runtime image; KServe deploys it",
        "commands": [],
    },
    "vllm": {
        "name": "vLLM",
        "repo": "https://github.com/vllm-project/vllm",
        "version": "bundled in the runtime image",
        "role": "Inference engine",
        "commands": [],
    },
    "dynamo": {
        "name": "NVIDIA Dynamo",
        "repo": "https://github.com/ai-dynamo/dynamo",
        "version": "v1.5.0",
        "role": "Operator, frontend, and router that run the engine workers",
        "commands": [
            "helm upgrade --install dynamo-crds oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds --version 1.5.0 -n default",
            "helm upgrade --install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version 1.5.0 -n dynamo-system --create-namespace --wait",
        ],
    },
    "sglang": {
        "name": "SGLang",
        "repo": "https://github.com/sgl-project/sglang",
        "version": "bundled in the runtime image",
        "role": "Inference engine",
        "commands": [],
    },
}

RECIPES = {
    "kserve-llmd-vllm": {
        "stack": "Envoy AI Gateway + KServe + llm-d + vLLM",
        "best_for": ("Platform teams serving many tenants, teams, or models behind one Kubernetes-native API. "
                     "Envoy AI Gateway adds OpenAI-compatible ingress, auth, rate and token limits; llm-d "
                     "routes each request to the replica with the most matching KV cache. vLLM is the most "
                     "widely supported engine. Most components to operate."),
        "prereqs": ["gpu-operator", "cert-manager", "gateway-api"],
        "projects": ["envoy-ai-gateway", "kserve", "llm-d", "vllm"],
        "image": "ghcr.io/llm-d/llm-d-cuda:v0.8.0",
        "namespace": "kserve-models",
        "min_kubernetes": None,
    },
    "dynamo-vllm": {
        "stack": "NVIDIA Dynamo + vLLM",
        "best_for": ("Teams wanting one operator and fewer moving parts. Dynamo's frontend has a KV-aware router "
                     "across workers and supports disaggregated prefill/decode as load grows. vLLM gives broad "
                     "model support and automatic prefix caching. Good general default for a single application."),
        "prereqs": ["gpu-operator"],
        "projects": ["dynamo", "vllm"],
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "namespace": "dynamo",
        "min_kubernetes": (1, 30),
    },
    "dynamo-sglang": {
        "stack": "NVIDIA Dynamo + SGLang",
        "best_for": ("Agentic and program-like workloads: long multi-turn sessions that keep growing, tool-call "
                     "loops, parallel branches from one context, and constrained JSON/grammar output. SGLang's "
                     "RadixAttention reuses shared and branching prefixes inside each worker and its structured "
                     "decoding is fast; Dynamo's KV-aware router keeps each session on the worker that holds its cache."),
        "prereqs": ["gpu-operator"],
        "projects": ["dynamo", "sglang"],
        "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.0",
        "namespace": "dynamo",
        "min_kubernetes": (1, 30),
    },
}


def catalog() -> dict:
    return deepcopy({"models": MODELS, "prereqs": PREREQS, "projects": PROJECTS, "recipes": RECIPES})
