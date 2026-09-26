from agent.architect import StrandsArchitect
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
