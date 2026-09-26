"""Deterministic, evidence-linked compaction helpers."""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from common import config


class CuratedMemory(BaseModel):
    claim: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class MemoryCurator:
    """Create memory records from completed experiments without touching workflow state."""

    def __init__(self, structured_hook: Callable[[dict], dict] | None = None):
        self.structured_hook = structured_hook or self._openrouter_hook()

    @staticmethod
    def _openrouter_hook():
        """Build a separate stateless Strands curator when credentials exist."""
        if not config.OPENROUTER_API_KEY:
            return None
        try:
            from strands import Agent
            from strands.models.openai import OpenAIModel

            model = OpenAIModel(
                client_args={"api_key": config.OPENROUTER_API_KEY,
                             "base_url": config.OPENROUTER_BASE_URL},
                model_id=config.AGENT_MODEL,
                params={"temperature": 0, "max_tokens": 500},
            )
            def invoke(payload: dict) -> dict:
                # Invocation-scoped agent: the durable evidence record is the
                # only memory, so compaction never accumulates chat history.
                agent = Agent(
                    model=model,
                    system_prompt=(
                        "Compact a terminal infrastructure experiment into one scoped claim and "
                        "a short summary. Do not invent evidence or identifiers."
                    ),
                    callback_handler=None,
                )
                result = agent(repr(payload), structured_output_model=CuratedMemory)
                if result.structured_output is None:
                    raise ValueError("memory curator returned no structured output")
                return result.structured_output.model_dump()

            return invoke
        except Exception:
            return None

    def curate(self, experiment: dict, related_experiments: list[dict] | None = None) -> dict:
        related = related_experiments or [experiment]
        experiment_id = str(experiment.get("experiment_id", experiment.get("_id", "unknown")))
        evidence_ids = [str(row.get("experiment_id", row.get("_id", "unknown"))) for row in related]
        evidence_ids = list(dict.fromkeys(evidence_ids))
        claim = self._claim(experiment)
        lesson = {
            "schema_version": 1,
            "lesson_id": f"lesson:{experiment_id}",
            "claim": claim,
            "text": claim,
            "scope": experiment.get("scope", {k: experiment[k] for k in ("model", "gpu", "engine", "regime") if k in experiment}),
            "evidence_ids": evidence_ids,
            "confidence": float(experiment.get("confidence", 0.5)),
            "contradictions": [],
            "supersedes": experiment.get("supersedes") or [],
        }
        summary = {
            "schema_version": 1,
            "summary_id": f"experiment-summary:{experiment_id}",
            "level": "experiment",
            "text": self._summary(experiment),
            "source_ids": evidence_ids,
            "source_hash": str(experiment.get("source_hash", "")),
            "version": int(experiment.get("summary_version", 1)),
            "scope": lesson["scope"],
        }
        result = {"lesson": lesson, "summary": summary}
        if self.structured_hook:
            try:
                structured = self.structured_hook({"experiment": experiment, "related": related,
                                                   "deterministic": result})
                # Treat model output as an optional refinement. Never permit it
                # to remove evidence linkage or replace the deterministic record.
                if isinstance(structured, dict):
                    if structured.get("claim"):
                        result["lesson"]["claim"] = str(structured["claim"])
                        result["lesson"]["text"] = str(structured["claim"])
                    if structured.get("summary"):
                        result["summary"]["text"] = str(structured["summary"])
            except Exception:
                pass
        return result

    @staticmethod
    def _claim(exp: dict) -> str:
        decision = str(exp.get("decision", "completed"))
        hypothesis = exp.get("hypothesis") or exp.get("proposal") or "Configuration tested"
        reason = exp.get("reason") or exp.get("summary") or ""
        return f"{hypothesis} — outcome: {decision}. {reason}".strip()

    @staticmethod
    def _summary(exp: dict) -> str:
        fields = ("hypothesis", "decision", "candidate_key", "baseline_p95_ms", "candidate_p95_ms",
                  "cost_change_pct", "reason")
        parts = [f"{field.replace('_', ' ')}: {exp[field]}" for field in fields if exp.get(field) is not None]
        return "; ".join(parts) or MemoryCurator._claim(exp)
