"""LLM advisor: picks one of the rule-sized stacks and explains the choice.

The rules size every stack first; the model only chooses among those
candidates. Any invalid answer or failure falls back to the rules' default.
"""
import json
from typing import Literal
from pydantic import BaseModel, Field
from common import config

SYSTEM_PROMPT = (
    "You are an inference infrastructure engineer choosing a Kubernetes serving stack for NVIDIA GPUs. "
    "You receive a workload (its task field is a free-text use case), a hardware inventory, and candidate "
    "stacks already sized by deterministic rules. Sizing is often identical across candidates, so decide "
    "on workload fit. Step 1: infer 3-5 concrete serving traits from the use case, e.g. session length "
    "and growth, prefix reuse within or across users, branching or parallel calls, tool calls or "
    "structured output, number of tenants or teams, need for central auth and quotas, operations capacity. "
    "Step 2: pick the candidate whose best_for matches those traits most closely; replica count alone is "
    "not a reason, since every candidate routes across replicas. Never pick a candidate with blockers "
    "when one without blockers exists. Explain in 2-4 plain sentences that connect the traits to the "
    "choice and cite numbers from the candidates. Do not invent benchmarks, versions, or capabilities "
    "beyond the input. No Markdown."
)


class StackChoice(BaseModel):
    inferred_traits: list[str] = Field(default_factory=list, max_length=6)
    recipe_id: Literal["kserve-llmd-vllm", "dynamo-vllm", "dynamo-sglang"]
    explanation: str = Field(max_length=1500)
    tradeoffs: list[str] = Field(default_factory=list, max_length=4)


def llm_advisor(request, candidates):
    """Return a StackChoice from the configured model, or raise."""
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    from strands import Agent
    from strands.models.openai import OpenAIModel
    model = OpenAIModel(client_args={"api_key": config.OPENROUTER_API_KEY,
                                     "base_url": config.OPENROUTER_BASE_URL,
                                     "timeout": 60, "max_retries": 1},
                        model_id=config.AGENT_MODEL,
                        params={"temperature": 0.1, "max_tokens": 1200})
    agent = Agent(model=model, callback_handler=None, system_prompt=SYSTEM_PROMPT)
    prompt = json.dumps({"workload": request["workload"], "inventory": request["inventory"],
                         "candidates": candidates}, default=str)
    result = agent(prompt, structured_output_model=StackChoice).structured_output
    if result is None:
        raise ValueError("empty structured output")
    return result


def choose(request, candidates, advisor=llm_advisor):
    """Validate the advisor's pick against the candidates; fall back to the first on any problem."""
    default = next((c["recipe_id"] for c in candidates if not c["blockers"]), candidates[0]["recipe_id"])
    try:
        choice = StackChoice.model_validate(advisor(request, candidates))
        clean = {c["recipe_id"] for c in candidates if not c["blockers"]}
        if clean and choice.recipe_id not in clean:
            raise ValueError(f"picked {choice.recipe_id} which has blockers")
        return {"recipe_id": choice.recipe_id, "explanation": choice.explanation,
                "tradeoffs": choice.tradeoffs, "traits": choice.inferred_traits, "decided_by": f"llm:{config.AGENT_MODEL}"}
    except Exception as exc:
        return {"recipe_id": default, "explanation": None, "tradeoffs": [], "traits": [],
                "decided_by": "rules", "fallback_reason": f"{type(exc).__name__}: {exc}"[:300]}
