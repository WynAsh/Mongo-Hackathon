"""Strands-powered, read-only infrastructure architect.

The architect may inspect evidence and return a typed proposal. Workflow code
owns validation, execution, promotion, rollback, and memory writes.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
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

ARCH_RULES = """Architecture invariants are strict:
- When split_threshold_tokens is null, pools MUST contain exactly one key: shared.
- When split_threshold_tokens is an integer, pools MUST contain exactly two keys: short and long.
- The candidate status MUST be candidate and created_by MUST be agent.
- Cite at least one exact evidence item_id from the supplied context or a read-only tool result.
Return only a proposal that satisfies these invariants."""


class _ProposalDraft(BaseModel):
    """Permissive transport shape; domain validation happens after normalization.

    Providers occasionally emit a coherent pool layout with a stale split flag.
    Keeping that syntactic mismatch outside ``Arch`` lets us repair the flag once,
    while every safety and policy rule still runs against the strict contracts.
    """

    proposal_id: str | None = None
    campaign_id: str = "default"
    hypothesis: str = Field(min_length=1)
    candidate: dict[str, Any]
    evidence_ids: list[str] = Field(default_factory=list)
    expected_effect: dict[str, Any] = Field(default_factory=dict)
    test_plan: dict[str, Any] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    rollback_plan: str = "Restore the previous live architecture"


def _validate_draft(draft: _ProposalDraft) -> ExperimentProposal:
    """Normalize only redundant metadata, then apply the strict public contract."""
    data = draft.model_dump()
    candidate = dict(data["candidate"])
    pools = dict(candidate.get("pools") or {})
    pool_names = set(pools)
    if pool_names == {"shared"}:
        candidate["split_threshold_tokens"] = None
    elif pool_names == {"short", "long"}:
        candidate["split_threshold_tokens"] = (
            candidate.get("split_threshold_tokens") or 500
        )
    candidate["status"] = "candidate"
    candidate["created_by"] = "agent"
    data["candidate"] = candidate
    return ExperimentProposal.model_validate(data)


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

_DEFAULT_MODEL = object()


class StrandsArchitect:
    """Create typed proposals with Strands, with a deterministic offline fallback."""

    def __init__(self, model=_DEFAULT_MODEL):
        # ``None`` is an explicit offline mode, useful for deterministic demos
        # and tests. Omitting the argument selects the configured OpenRouter model.
        if model is _DEFAULT_MODEL:
            model = None
            if config.OPENROUTER_API_KEY:
                model = OpenAIModel(
                    client_args={
                        "api_key": config.OPENROUTER_API_KEY,
                        "base_url": config.OPENROUTER_BASE_URL,
                    },
                    model_id=config.AGENT_MODEL,
                    params={"temperature": 0.2, "max_tokens": 1800},
                )
        self._enabled = model is not None
        self.last_run = {"source": "offline", "model": None, "attempts": 0, "error": None}
        self.model = model

    def _new_agent(self) -> Agent:
        """Create an invocation-scoped agent so chat history never accumulates."""
        return Agent(
            model=self.model,
            tools=READ_ONLY_TOOLS,
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )

    def propose(self, packet: dict, *, current: Arch, policy: dict) -> ExperimentProposal:
        if self.model is not None:
            base_prompt = (
                "Propose the next shadow experiment from this bounded context.\n"
                f"{ARCH_RULES}\n"
                f"Policy: {json.dumps(_clean(policy), default=str)}\n"
                f"Current architecture: {current.model_dump_json()}\n"
                f"Context packet: {json.dumps(_clean(packet), default=str)}"
            )
            error = None
            for attempt in range(1, 3):
                try:
                    prompt = base_prompt if error is None else (
                        f"{base_prompt}\nYour prior output failed validation: {error}. "
                        "Repair the proposal and obey every architecture invariant.")
                    result = self._new_agent()(
                        prompt,
                        structured_output_model=_ProposalDraft,
                    )
                    draft = result.structured_output
                    if draft is None:
                        raise ValueError("model returned no structured proposal")
                    proposal = _validate_draft(draft)
                    self.last_run = {"source": "openrouter", "model": config.AGENT_MODEL,
                                     "attempts": attempt, "error": None}
                    return proposal
                except Exception as exc:  # structured repair gets one bounded retry
                    error = str(exc)
            self.last_run = {"source": "fallback", "model": config.AGENT_MODEL,
                             "attempts": 2, "error": error}
            print(f"[architect] structured proposal failed ({error}); deterministic fallback", flush=True)
        else:
            self.last_run = {"source": "fallback", "model": None, "attempts": 0,
                             "error": "OPENROUTER_API_KEY is not configured"}
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
