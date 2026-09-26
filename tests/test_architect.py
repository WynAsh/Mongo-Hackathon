from agent.architect import StrandsArchitect, _ProposalDraft, _validate_draft
from common.contracts import Arch, Pool


def test_offline_architect_returns_typed_split_proposal():
    architect = StrandsArchitect(model=None)
    current = Arch(status="live", version=1, pools={"shared": Pool(gpu="t4", replicas=2)})
    packet = {
        "campaign_id": "c",
        "context": [{
            "item_id": "observation",
            "memory_type": "observation",
            "content": {"p95_ms": 6000, "regime_vector": [0.3, 0.2, 0.9, 0.3]},
        }],
    }
    proposal = architect.propose(packet, current=current, policy={"slo_p95_ms": 4000})
    assert proposal.candidate.split_threshold_tokens == 500
    assert proposal.candidate.created_by == "agent"


def test_offline_architect_scales_down_when_overprovisioned():
    architect = StrandsArchitect(model=None)
    current = Arch(status="live", version=1, pools={"shared": Pool(gpu="t4", replicas=2)})
    packet = {
        "campaign_id": "c",
        "context": [{
            "item_id": "observation",
            "memory_type": "observation",
            "content": {"p95_ms": 500, "regime_vector": [0.1, 0.1, 0.1, 0]},
        }],
    }
    proposal = architect.propose(packet, current=current, policy={"slo_p95_ms": 4000})
    assert proposal.candidate.pools["shared"].replicas == 1


def test_model_draft_repairs_stale_split_flag_for_shared_pool():
    proposal = _validate_draft(_ProposalDraft(
        campaign_id="c",
        hypothesis="One shared T4 should preserve SLO headroom.",
        candidate={
            "split_threshold_tokens": 500,
            "pools": {"shared": {"gpu": "t4", "replicas": 1}},
            "reason": "reduce idle capacity",
        },
        evidence_ids=["metric-window-1"],
    ))
    assert proposal.candidate.split_threshold_tokens is None
    assert proposal.candidate.status == "candidate"
    assert proposal.candidate.created_by == "agent"


def test_model_draft_repairs_missing_split_flag_for_named_pools():
    proposal = _validate_draft(_ProposalDraft(
        campaign_id="c",
        hypothesis="Separate long prefills.",
        candidate={
            "pools": {
                "short": {"gpu": "t4", "replicas": 2},
                "long": {"gpu": "a100", "replicas": 1},
            },
            "reason": "isolate prefills",
        },
        evidence_ids=["metric-window-1"],
    ))
    assert proposal.candidate.split_threshold_tokens == 500
