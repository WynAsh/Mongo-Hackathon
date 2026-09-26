"""Strands-powered, read-only infrastructure architect.

The architect may inspect evidence and return a typed proposal. Workflow code
owns validation, execution, promotion, rollback, and memory writes.
"""
from __future__ import annotations

import json
from typing import Any

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from common import config, db
from common.contracts import (
    Arch,
    ExperimentProposal,
    ExperimentTestPlan,
    Pool,
)
from gateway import metrics


SYSTEM_PROMPT = """You are the Architect for a long-horizon LLM-serving optimizer.
Return one concrete ExperimentProposal. Optimize GPU cost subject to hard p95
latency and error-rate policies. Cite only evidence IDs present in the supplied
context or returned by read-only tools. Never claim a configuration is proven;
the deterministic experiment evaluator makes that decision. Available knobs are
split_threshold_tokens, pool GPU type, and pool replica count. Prefer the
smallest meaningful change and do not repeat a known losing configuration."""


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k != "_id"}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return str(value) if value.__class__.__name__ == "ObjectId" else value


@tool
def get_metric_window(seconds: int = 20) -> dict:
    """Read the latest compact live-serving metric window."""
    live = db.live_arch_doc()
    return metrics.window(seconds, live.get("version") if live else None)


@tool
def get_experiment(experiment_id: str) -> dict:
    """Read one experiment by its stable experiment ID."""
    row = db.db().experiments.find_one(
        {"$or": [{"experiment_id": experiment_id}, {"_id": experiment_id}]}
    )
    return _clean(row or {"error": "experiment not found"})


@tool
def search_lessons(query: str, limit: int = 5) -> list[dict]:
    """Search evidence-linked lessons lexically for offline-safe recall."""
    words = [word.lower() for word in query.split() if len(word) > 2]
    rows = list(db.db().lessons.find().sort("ts", -1).limit(100))
    rows.sort(
        key=lambda row: -sum(
            str(row.get("claim", row.get("text", ""))).lower().count(word)
            for word in words
        )
    )
    return [_clean(row) for row in rows[: max(1, min(limit, 10))]]


@tool
def search_docs(query: str, limit: int = 5) -> list[dict]:
    """Search ingested serving documentation and return provenance."""
    words = [word.lower() for word in query.split() if len(word) > 2]
    rows = list(db.db().docs.find({}, {"embedding": 0}).limit(200))
    rows.sort(
        key=lambda row: -sum(
            str(row.get("text", "")).lower().count(word) for word in words
        )
    )
    return [_clean(row) for row in rows[: max(1, min(limit, 10))]]


@tool
def expand_evidence(evidence_id: str) -> dict:
    """Expand a lesson, summary, experiment, trial, or document by stable ID."""
    for collection, fields in (
        ("lessons", ("lesson_id", "_id")),
        ("summaries", ("summary_id", "_id")),
        ("experiments", ("experiment_id", "_id")),
        ("trials", ("trial_id", "_id")),
        ("docs", ("doc_id", "_id")),
    ):
        row = db.db()[collection].find_one({"$or": [{field: evidence_id} for field in fields]})
        if row:
            return {"collection": collection, "record": _clean(row)}
    return {"error": "evidence not found", "evidence_id": evidence_id}


READ_ONLY_TOOLS = [
    get_metric_window,
    get_experiment,
    search_lessons,
    search_docs,
    expand_evidence,
]


class StrandsArchitect:
    """Create typed proposals with Strands, with a deterministic offline fallback."""

    def __init__(self, model=None):
        self._enabled = bool(config.OPENROUTER_API_KEY or model)
        if model is None and self._enabled:
            model = OpenAIModel(
                client_args={
                    "api_key": config.OPENROUTER_API_KEY,
                    "base_url": config.OPENROUTER_BASE_URL,
                },
                model_id=config.AGENT_MODEL,
                params={"temperature": 0.2, "max_tokens": 1800},
            )
        self.agent = (
            Agent(
                model=model,
                tools=READ_ONLY_TOOLS,
                system_prompt=SYSTEM_PROMPT,
                structured_output_model=ExperimentProposal,
                callback_handler=None,
            )
            if model is not None
            else None
        )

    def propose(self, packet: dict, *, current: Arch, policy: dict) -> ExperimentProposal:
        if self.agent is not None:
            try:
                prompt = (
                    "Propose the next shadow experiment from this bounded context.\n"
                    f"Policy: {json.dumps(_clean(policy), default=str)}\n"
                    f"Current architecture: {current.model_dump_json()}\n"
                    f"Context packet: {json.dumps(_clean(packet), default=str)}"
                )
                proposal = self.agent.structured_output(ExperimentProposal, prompt)
                proposal.candidate.status = "candidate"
                proposal.candidate.created_by = "agent"
                return proposal
            except Exception as exc:  # keep the offline/demo path alive
                print(f"[architect] Strands unavailable ({exc}); deterministic fallback", flush=True)
        return self._fallback(packet, current=current, policy=policy)

    @staticmethod
    def _fallback(packet: dict, *, current: Arch, policy: dict) -> ExperimentProposal:
        observation = {}
        for item in packet.get("context", []):
            if item.get("memory_type") == "observation":
                observation = item.get("content") or {}
                break
        regime = observation.get("regime_vector", [0, 0, 0, 0])
        metrics_ = observation.get("metrics", observation)
        p95 = metrics_.get("p95_ms")
        slo = float(policy.get("slo_p95_ms", config.SLO_P95_MS))
        if p95 is not None and p95 > slo and len(regime) > 3 and regime[3] > 0.05:
            candidate = Arch(
                status="candidate",
                created_by="agent",
                split_threshold_tokens=500,
                pools={
                    "short": Pool(gpu="t4", replicas=2),
                    "long": Pool(gpu="a100", replicas=1),
                },
                reason="isolate long prefills to remove head-of-line blocking",
            )
            hypothesis = "Separating long prefills onto one A100 will restore the p95 SLO."
        elif p95 is not None and p95 < slo * 0.5 and current.usd_hr() > 0.5:
            candidate = Arch(
                status="candidate", created_by="agent",
                pools={"shared": Pool(gpu="t4", replicas=1)},
                reason="test whether a single inexpensive replica retains SLO headroom",
            )
            hypothesis = "The current traffic can meet the SLO with one T4."
        else:
            replicas = min(4, sum(pool.replicas for pool in current.pools.values()) + 1)
            candidate = Arch(
                status="candidate", created_by="agent",
                pools={"shared": Pool(gpu="t4", replicas=replicas)},
                reason="increase inexpensive capacity for the current regime",
            )
            hypothesis = "One additional T4 will reduce queueing enough to meet the SLO."
        evidence = [
            item["item_id"] for item in packet.get("context", [])
            if item.get("memory_type") in {"observation", "semantic", "episodic", "documentation"}
        ][:5]
        return ExperimentProposal(
            campaign_id=packet.get("campaign_id", config.CAMPAIGN_ID),
            hypothesis=hypothesis,
            candidate=candidate,
            evidence_ids=evidence,
            test_plan=ExperimentTestPlan(
                duration_s=config.SHADOW_S,
                repeats=config.TRIAL_REPEATS,
                min_requests=config.MIN_TRIAL_REQUESTS,
            ),
            stop_conditions=[
                f"error rate exceeds {policy.get('max_error_rate', 0.1):.0%}",
                f"p95 latency exceeds {slo:.0f} ms",
            ],
        )
