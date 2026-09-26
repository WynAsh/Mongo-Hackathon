"""Stage-aware, bounded memory compilation for agent invocations."""
from __future__ import annotations

import time
import uuid
from typing import Any

from common.contracts import ContextItem, ContextManifest, ExcludedContextItem
from memory.embeddings import OpenRouterEmbedder, approximate_tokens, lexical_score


class ContextCompiler:
    """Compile ordered Mongo records into a bounded context packet.

    Inputs and Mongo documents are treated as plain mappings to keep this
    module compatible with the current contracts and forthcoming campaign
    contracts. Collections are resolved lazily, so mongomock and simple test
    doubles can be used.
    """

    ORDER = ("policy", "campaign", "checkpoint", "observation", "regime_winner",
             "lesson", "experiment", "documentation")

    def __init__(self, database, embedder: OpenRouterEmbedder | None = None,
                 token_budget: int = 12000, clock=time.time):
        self.database = database
        self.embedder = embedder or OpenRouterEmbedder()
        self.token_budget = max(1, int(token_budget))
        self.clock = clock

    def _collection(self, name: str):
        return self.database[name]

    @staticmethod
    def _plain(doc: dict | None) -> dict | None:
        if doc is None:
            return None
        result = {k: v for k, v in doc.items() if k != "_id"}
        if "_id" in doc:
            result.setdefault("id", str(doc["_id"]))
        return result

    def _one(self, collection: str, query: dict, sort=None) -> dict | None:
        return self._plain(self._collection(collection).find_one(query, sort=sort))

    def _many(self, collection: str, query: dict, limit: int = 10, sort=None) -> list[dict]:
        cursor = self._collection(collection).find(query)
        if sort:
            cursor = cursor.sort(sort)
        return [self._plain(x) for x in cursor.limit(limit)]

    def _rank(self, collection: str, query: dict, text_fields: tuple[str, ...],
              query_text: str, limit: int, vector: list[float] | None = None) -> list[dict]:
        coll = self._collection(collection)
        if vector and not getattr(self.embedder, "available", False):
            vector = None
        if vector:
            # Atlas vector search requires the pipeline to start with $vectorSearch.
            # Structured constraints are applied after retrieval to stay compatible
            # with Atlas index definitions that only index the embedding field.
            try:
                pipeline = [
                    {"$vectorSearch": {
                        "index": f"{collection}_text", "path": "embedding",
                        "queryVector": vector, "numCandidates": max(100, limit * 10),
                        "limit": limit * 3,
                        **({"filter": query} if query else {}),
                    }},
                    {"$addFields": {"_retrieval_score": {"$meta": "vectorSearchScore"}}},
                    {"$limit": limit},
                ]
                hits = list(coll.aggregate(pipeline))
                if hits:
                    return [self._plain(h) for h in hits]
            except Exception:
                # Index absent, still building, or mongomock: lexical fallback.
                pass
        docs = list(coll.find(query))
        for doc in docs:
            haystack = " ".join(str(doc.get(field, "")) for field in text_fields)
            doc["_retrieval_score"] = lexical_score(query_text, haystack)
        docs.sort(key=lambda d: (-d.get("_retrieval_score", 0), str(d.get("_id", ""))))
        return [self._plain(d) for d in docs[:limit]]

    def compile(self, *, campaign_id: str = "default", stage: str = "OBSERVE",
                query: str = "", observation: dict | None = None,
                regime_vector: list[float] | None = None,
                scope_filter: dict | None = None,
                token_budget: int | None = None) -> dict[str, Any]:
        budget = max(1, int(token_budget or self.token_budget))
        query_text = query or stage
        query_vector = None
        try:
            vectors = self.embedder.embed([query_text])
            query_vector = vectors[0] if vectors else None
        except Exception:
            query_vector = None

        campaign = self._one("campaigns", {"campaign_id": campaign_id})
        if campaign is None:
            campaign = self._one("campaigns", {"_id": campaign_id})
        checkpoint = self._one("campaign_checkpoints", {"campaign_id": campaign_id},
                               sort=[("revision", -1)])
        policy = self._one("policies", {"campaign_id": campaign_id}) or self._one("policies", {"_id": campaign_id})
        if policy is None:
            policy = self._one("policies", {"scope": "global"})

        winner_records: list[dict] = []
        if regime_vector is not None:
            # Preserve the existing four-dimensional Thompson-sampling memory
            # shape; this retrieval is structured and independent of text embeddings.
            regime_docs = self._many("regimes", {}, limit=100)
            for row in regime_docs:
                vec = row.get("vector", row.get("regime_vector", []))
                if len(vec) != len(regime_vector):
                    continue
                dist = sum((float(a) - float(b)) ** 2 for a, b in zip(vec, regime_vector))
                if row.get("wins", 0) > 0:
                    row["_retrieval_score"] = 1 / (1 + dist)
                    winner_records.append(row)
            winner_records.sort(key=lambda row: -row["_retrieval_score"])
            winner_records = winner_records[:5]

        lesson_filter = {f"scope.{key}": value for key, value in (scope_filter or {}).items()}
        lessons = self._rank("lessons", lesson_filter, ("text", "claim", "summary"), query_text, 5, query_vector)
        experiment_scope = {"campaign_id": campaign_id}
        experiments = self._rank("experiments", experiment_scope if self._collection("experiments").count_documents(experiment_scope) else {},
                                 ("hypothesis", "summary", "decision", "reason"), query_text, 5, query_vector)
        # Documentation is generally workload-agnostic and ingested chunks do
        # not carry a regime scope. Relevance ranking supplies the filter here.
        docs = self._rank("docs", {}, ("title", "section", "text", "content"), query_text, 5, query_vector)

        sections = [
            ("policy", policy), ("campaign", campaign), ("checkpoint", checkpoint),
            ("observation", observation), ("regime_winners", winner_records),
            ("lessons", lessons), ("experiments", experiments), ("documentation", docs),
        ]
        entries, excluded = [], []
        used = 0
        for kind, value in sections:
            values = value if isinstance(value, list) else ([] if value is None else [value])
            for item in values:
                rendered = repr(item)
                cost = approximate_tokens(rendered)
                item_id = str(item.get("_id", item.get("id", item.get("lesson_id", item.get("doc_id", "")))))
                if used + cost <= budget:
                    memory_type = self._memory_type(kind)
                    source_ids = item.get("source_ids", item.get("evidence_ids", []))
                    entry = ContextItem(
                        item_id=item_id or f"{kind}:{len(entries)}", memory_type=memory_type,
                        reason=self._reason(kind), version=str(item.get("version", item.get("revision")))
                        if item.get("version", item.get("revision")) is not None else None,
                        source_ids=[str(x) for x in source_ids], token_estimate=cost, content=item,
                    )
                    entries.append(entry.model_dump())
                    used += cost
                else:
                    excluded.append(ExcludedContextItem(item_id=item_id or f"{kind}:{len(excluded)}",
                                                        reason="token_budget").model_dump())

        revision = int((campaign or {}).get("revision", 0))
        manifest_model = ContextManifest(
            manifest_id=str(uuid.uuid4()), campaign_id=campaign_id, campaign_revision=revision,
            stage=stage, query=query_text, token_budget=budget, included=entries, excluded=excluded,
            created_at=self.clock(),
        )
        manifest = manifest_model.model_dump()
        manifest["total_tokens"] = manifest_model.total_tokens
        self._collection("context_manifests").insert_one(manifest.copy())
        return {"campaign_id": campaign_id, "stage": stage, "context": entries,
                "token_estimate": used, "token_budget": budget, "manifest": manifest}

    @staticmethod
    def _reason(kind: str) -> str:
        return {
            "policy": "mandatory_policy", "campaign": "campaign_state",
            "checkpoint": "latest_checkpoint", "observation": "current_observation",
            "regime_winners": "similar_traffic_regime", "lessons": "scoped_semantic_retrieval",
            "experiments": "supporting_experiment_retrieval", "documentation": "relevant_documentation",
        }[kind]

    @staticmethod
    def _memory_type(kind: str) -> str:
        return {
            "policy": "policy", "campaign": "working", "checkpoint": "working",
            "observation": "observation", "regime_winners": "semantic", "lessons": "semantic",
            "experiments": "episodic", "documentation": "documentation",
        }[kind]
