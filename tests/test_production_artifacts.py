from production.artifacts import render_bundle
from production.catalog import catalog, select_plan


def _inventory(vram=80, count=4):
    return {
        "os": "Ubuntu 24.04",
        "driver_version": "550.90",
        "cuda_version": "12.8",
        "kubernetes_version": "1.32",
        "nodes": [{"name": "gpu-a", "gpu_model": "H100", "gpu_count": count, "vram_gb": vram, "ram_gb": 512, "storage_gb": 2000, "interconnect": "NVLink"}],
    }


def _workload(**overrides):
    return {"task": "chat", "context_tokens": 4096, "output_tokens": 512, "concurrency": 2, "rps": 3, "allowed_licenses": ["Apache-2.0"], **overrides}


def test_catalog_exposes_three_stacks_and_six_model_stack_choices():
    data = catalog()
    assert set(data["recipes"]) == {"kserve-llmd-vllm", "dynamo-vllm", "dynamo-sglang"}
    assert len(data["models"]) == 2
    assert all(len(recipe["model_choices"]) == 2 for recipe in data["recipes"].values())
    assert all(len(model["revision"]) == 40 for model in data["models"].values())


def test_select_plan_pins_model_and_reports_context_capacity_blockers():
    plan = select_plan(_inventory(vram=24, count=1), _workload(quality_tier="high", context_tokens=40000, output_tokens=2000))
    assert plan["model"]["model_id"] == "Qwen/Qwen3-4B"  # 8B preference falls back by per-GPU capacity.
    assert plan["config"]["max_model_len"] == 40960
    assert any("exceeds" in item for item in plan["blockers"])
    assert plan["allocation"] == {"replicas": 1, "tensor_parallel": 1, "gpus_per_replica": 1}


def test_select_plan_rejects_incompatible_task_and_license():
    plan = select_plan(_inventory(), _workload(task="embeddings", allowed_licenses=["MIT"]))
    assert any("not compatible" in item for item in plan["blockers"])
    assert any("license" in item for item in plan["blockers"])


def test_render_bundle_is_deterministic_and_passes_pinned_schema_projections():
    plan = select_plan(_inventory(), _workload(), "dynamo-sglang")
    first = render_bundle(plan)
    second = render_bundle(plan)
    assert first["files"] == second["files"]
    assert first["validation"]["status"] == "offline_validated"
    assert next(c for c in first["validation"]["checks"] if c["name"] == "upstream-crd-schema-projection")["status"] == "passed"
    assert any("server-side dry-run is pending" in item for item in first["validation"]["blockers"])
    assert first["files"]["k8s/deployment.yaml"].startswith("apiVersion: nvidia.com/v1beta1")
    assert "dynamo.sglang" in first["files"]["k8s/deployment.yaml"]
    assert "dynamo.frontend" in first["files"]["k8s/deployment.yaml"]
    assert "--http-port" in first["files"]["k8s/deployment.yaml"]
    assert "--tp-size" in first["files"]["k8s/deployment.yaml"]
    assert plan["model"]["revision"] in first["files"]["k8s/deployment.yaml"]
    assert "${HF_TOKEN}" in first["files"]["commands.txt"]
    assert "HF_TOKEN=\n" in first["files"][".env.example"]
    assert "HF_TOKEN=secret" not in "".join(first["files"].values())


def test_render_bundle_includes_concrete_previous_manifest_rollback():
    plan = select_plan(_inventory(), _workload(), "kserve-llmd-vllm", "Qwen/Qwen3-8B")
    previous = select_plan(_inventory(), _workload(), "kserve-llmd-vllm", "Qwen/Qwen3-4B")
    bundle = render_bundle(plan, previous)
    assert bundle["files"]["k8s/rollback.yaml"].startswith("apiVersion: serving.kserve.io/v1alpha2")
    assert "Qwen/Qwen3-4B" in bundle["files"]["k8s/rollback.yaml"]
    assert "Qwen/Qwen3-8B" in bundle["files"]["CHANGELOG.md"]
    assert "k8s/rollback.yaml" in bundle["files"]["RUNBOOK.md"]
    assert bundle["files"]["k8s/deployment.yaml"].startswith("apiVersion: serving.kserve.io/v1alpha2")
    assert "k8s/gateway.yaml" in bundle["files"]
    assert "parallelism:" in bundle["files"]["k8s/deployment.yaml"]
    assert "--tensor-parallel-size" in bundle["files"]["k8s/deployment.yaml"]
    assert "kubectl wait --for=condition=Ready" in bundle["files"]["commands.txt"]


def test_renderer_fails_schema_validation_for_invalid_upstream_field_type():
    plan = select_plan(_inventory(), _workload(), "dynamo-vllm")
    plan["allocation"]["replicas"] = "many"
    bundle = render_bundle(plan)
    assert bundle["validation"]["status"] == "validation_failed"
    assert next(c for c in bundle["validation"]["checks"] if c["name"] == "upstream-crd-schema-projection")["status"] == "failed"


def test_all_six_stack_model_pairs_validate_offline():
    for recipe_id in catalog()["recipes"]:
        for model_id in ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"):
            plan = select_plan(_inventory(), _workload(), recipe_id, model_id)
            bundle = render_bundle(plan)
            assert bundle["validation"]["status"] == "offline_validated", (recipe_id, model_id, bundle["validation"])
            manifest = bundle["files"]["k8s/deployment.yaml"]
            assert plan["model"]["revision"] in manifest
            if recipe_id == "dynamo-sglang":
                assert "--context-length" in manifest
                assert "--max-running-requests" in manifest
            elif recipe_id == "dynamo-vllm":
                assert "--max-model-len" in manifest
                assert "--max-num-seqs" in manifest
