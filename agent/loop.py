"""Durable architect entrypoint retained for ``scripts.run_all`` compatibility."""
from __future__ import annotations

import time

from agent.workflow import DurableOptimizationWorkflow
from common import config, db


def cycle(state: dict | None = None) -> bool:
    """Advance one persisted stage; ``state`` remains accepted by legacy callers."""
    workflow = (state or {}).get("workflow") if state is not None else None
    if workflow is None:
        workflow = DurableOptimizationWorkflow()
        if state is not None:
            state["workflow"] = workflow
    return workflow.tick()


def main():
    workflow = DurableOptimizationWorkflow()
    campaign = workflow.bootstrap()
    resumed = campaign.get("revision", 0) > 0
    db.log_event(
        "resume" if resumed else "start",
        f"Long-horizon architect {'resumed' if resumed else 'online'} at "
        f"{campaign['stage']} r{campaign['revision']}. "
        f"LLM: {config.AGENT_MODEL if config.OPENROUTER_API_KEY else 'offline heuristic'}",
        campaign_id=config.CAMPAIGN_ID,
    )
    while True:
        try:
            progressed = workflow.tick()
            time.sleep(0.25 if progressed else config.CYCLE_S)
        except Exception as exc:
            print(f"[agent] cycle failed: {exc}", flush=True)
            time.sleep(config.CYCLE_S)


if __name__ == "__main__":
    main()
