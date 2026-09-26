from mongomock import MongoClient

from memory.context import ContextCompiler


class NoEmbedding:
    available = False

    def embed(self, texts):
        return None


def test_context_orders_required_memory_and_writes_manifest():
    database = MongoClient().db
    database.policies.insert_one({"_id": "default", "max_error_rate": 0.1})
    database.campaigns.insert_one({"campaign_id": "default", "goal": "lower cost"})
    database.campaign_checkpoints.insert_one({"campaign_id": "default", "revision": 2, "stage": "PROPOSE"})
    database.regimes.insert_one({"vector": [0.1, 0.2, 0.3, 0.0], "arch_key": "candidate", "wins": 2})
    database.lessons.insert_one({"text": "lower cost through fewer replicas", "scope": {"gpu": "t4"}})
    database.experiments.insert_one({"campaign_id": "default", "hypothesis": "lower replica count"})
    database.docs.insert_one({"title": "Scaling", "text": "replica scaling lowers cost"})

    result = ContextCompiler(database, NoEmbedding(), token_budget=1000).compile(
        campaign_id="default", stage="PROPOSE", query="lower cost",
        observation={"p95_ms": 200}, regime_vector=[0.1, 0.2, 0.3, 0.0])

    kinds = [entry["memory_type"] for entry in result["context"]]
    assert kinds == ["policy", "working", "working", "observation", "semantic",
                     "semantic", "episodic", "documentation"]
    assert result["token_estimate"] <= 1000
    assert database.context_manifests.count_documents({"campaign_id": "default"}) == 1


def test_context_respects_budget_and_explains_exclusions():
    database = MongoClient().db
    database.policies.insert_one({"_id": "default", "text": "mandatory policy " * 80})
    result = ContextCompiler(database, NoEmbedding(), token_budget=2).compile(campaign_id="default")
    assert result["token_estimate"] <= 2
    assert result["manifest"]["excluded"][0]["reason"] == "token_budget"


def test_lexical_fallback_ranks_matching_lessons():
    database = MongoClient().db
    database.lessons.insert_many([
        {"text": "batching improves throughput for long prompts"},
        {"text": "replicas reduce cost for bursty traffic"},
    ])
    result = ContextCompiler(database, NoEmbedding()).compile(query="batching throughput")
    lesson = next(x for x in result["context"] if x["memory_type"] == "semantic")
    assert "batching" in lesson["content"]["text"]
