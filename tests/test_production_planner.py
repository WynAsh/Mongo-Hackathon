import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from production.catalog import RECIPES, PROJECTS
from production.planner import build_plan


def _request(vram=80, count=4, nodes=1, kube="1.32", **workload):
    return {
        "inventory": {"os": "Ubuntu 24.04", "driver_version": "550.90", "cuda_version": "12.8",
                      "kubernetes_version": kube,
                      "nodes": [{"name": f"gpu-{i}", "gpu_model": "H100", "gpu_count": count,
                                 "vram_gb": vram, "interconnect": "NVLink"} for i in range(nodes)]},
        "workload": {"task": "chat", "context_tokens": 4096, "output_tokens": 512, "concurrency": 2,
                     "allowed_licenses": ["Apache-2.0"], **workload},
    }


def test_catalog_covers_six_projects():
    assert set(PROJECTS) == {"envoy-ai-gateway", "kserve", "llm-d", "vllm", "dynamo", "sglang"}
    used = {p for r in RECIPES.values() for p in r["projects"]}
    assert used == set(PROJECTS)


def test_default_plan_uses_every_gpu_and_is_ready():
    result = build_plan(_request(nodes=2))
    plan = result["plan"]
    assert result["status"] == "ready", plan["blockers"]
    assert plan["recipe_id"] == "kserve-llmd-vllm"
    assert plan["allocation"]["tensor_parallel"] == 1
    assert plan["allocation"]["replicas"] == 8
    assert set(result["files"]) == {"k8s/10-gatewayclass.yaml", "k8s/20-gateway.yaml",
                                    "k8s/30-model.yaml", "install.sh", "PLAN.md"}
    assert all(c["ok"] for c in result["validation"])
    titles = [s["title"] for s in result["steps"]]
    assert titles[0] == "Preflight" and titles[-2:] == ["Deploy", "Smoke test"]
    assert any("Envoy AI Gateway" in t for t in titles) and any("KServe" in t for t in titles)


def test_every_recipe_and_model_validates():
    for recipe_id in RECIPES:
        for model_id in ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"):
            result = build_plan({**_request(), "recipe_id": recipe_id, "model_id": model_id})
            assert result["status"] == "ready", (recipe_id, model_id, result["validation"])
            manifest = result["files"]["k8s/30-model.yaml"]
            assert yaml.safe_load(manifest)["spec"]
            assert result["plan"]["model"]["revision"] in manifest
    sglang = build_plan({**_request(), "recipe_id": "dynamo-sglang"})["files"]["k8s/30-model.yaml"]
    assert "dynamo.sglang" in sglang and "--context-length" in sglang and "--tp-size" in sglang


def test_small_gpu_falls_back_and_reports_blockers():
    result = build_plan(_request(vram=24, count=1, quality="high", context_tokens=40000, output_tokens=2000))
    plan = result["plan"]
    assert plan["model"]["model_id"] == "Qwen/Qwen3-4B"
    assert plan["engine"]["max_model_len"] == 40960
    assert any("exceeds" in b for b in plan["blockers"])
    assert result["status"] == "blocked"


def test_large_context_needs_tensor_parallel():
    plan = build_plan(_request(vram=48, count=4, concurrency=16, context_tokens=16000))["plan"]
    assert plan["allocation"]["tensor_parallel"] == 2
    assert plan["allocation"]["replicas"] == 2


def test_dynamo_blocks_old_kubernetes_and_bad_license():
    result = build_plan({**_request(kube="v1.28.3", allowed_licenses=["MIT"]), "recipe_id": "dynamo-vllm"})
    assert any("Kubernetes v1.30+" in b for b in result["plan"]["blockers"])
    assert any("license" in b for b in result["plan"]["blockers"])


def test_install_script_is_ordered_and_keeps_secrets_out():
    script = build_plan(_request())["files"]["install.sh"]
    assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script
    assert script.index("cert-manager") < script.index("kserve-llmisvc") < script.index("kubectl apply -f k8s/")
    assert '"${HF_TOKEN:-}"' in script
    assert "if ! kubectl get nodes" in script  # GPU operator only when GPUs are missing


def test_plan_endpoint_persists_and_downloads(monkeypatch):
    import mongomock
    from production import api
    database = mongomock.MongoClient().db
    monkeypatch.setattr(api.db, "db", lambda: database)
    monkeypatch.setattr(api, "llm_advisor", _advisor("dynamo-vllm"))
    app = FastAPI()
    api.install(app)
    client = TestClient(app)
    body = client.post("/api/production/plan", json=_request()).json()
    assert body["status"] == "ready"
    assert client.get("/api/production/plans").json()[0]["plan_id"] == body["plan_id"]
    download = client.get(f"/api/production/plans/{body['plan_id']}/download")
    assert download.headers["content-type"] == "application/zip"
    assert client.post("/api/production/plan", json={"model_id": "nope"}).status_code == 422


def _advisor(recipe_id):
    return lambda request, candidates: {"recipe_id": recipe_id, "explanation": f"Chose {recipe_id}.", "inferred_traits": ["long sessions"],
                                        "tradeoffs": ["t"]}


def test_advisor_picks_stack_and_explains():
    seen = {}
    def advisor(request, candidates):
        seen["ids"] = [c["recipe_id"] for c in candidates]
        return _advisor("dynamo-sglang")(request, candidates)
    result = build_plan(_request(), advisor=advisor)
    assert seen["ids"] == ["kserve-llmd-vllm", "dynamo-vllm", "dynamo-sglang"]
    assert result["plan"]["recipe_id"] == "dynamo-sglang"
    assert result["plan"]["decided_by"].startswith("llm:")
    assert "Chose dynamo-sglang." in result["files"]["PLAN.md"]


def test_advisor_cannot_pick_blocked_stack_and_failures_fall_back():
    blocked = build_plan(_request(kube="1.28"), advisor=_advisor("dynamo-vllm"))["plan"]
    assert blocked["recipe_id"] == "kserve-llmd-vllm" and blocked["decided_by"] == "rules"
    assert any("LLM advisor unavailable" in n for n in blocked["notes"])
    def broken(request, candidates):
        raise TimeoutError("slow")
    assert build_plan(_request(), advisor=broken)["plan"]["decided_by"] == "rules"


def test_explicit_recipe_skips_advisor():
    def never(request, candidates):
        raise AssertionError("advisor should not run")
    plan = build_plan({**_request(), "recipe_id": "dynamo-vllm"}, advisor=never)["plan"]
    assert plan["decided_by"] == "request"


def test_advisor_sees_stack_strengths_and_traits_reach_plan():
    seen = {}
    def advisor(request, candidates):
        seen["best_for"] = all(c["best_for"] for c in candidates)
        return _advisor("dynamo-sglang")(request, candidates)
    result = build_plan(_request(task="agentic coding"), advisor=advisor)
    assert seen["best_for"]
    assert result["plan"]["traits"] == ["long sessions"]
    assert "Trait: long sessions" in result["files"]["PLAN.md"]
