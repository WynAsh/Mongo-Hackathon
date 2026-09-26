from __future__ import annotations

import mongomock
import pytest

from agent.state import (
    BudgetExceeded,
    CampaignStore,
    LeaseLost,
    RevisionConflict,
    ensure_indexes,
)


@pytest.fixture
def store():
    client = mongomock.MongoClient()
    now = [1000.0]
    return CampaignStore(client.db, clock=lambda: now[0]), now


def test_bootstrap_is_idempotent_and_preserves_existing_campaign(store):
    state, _ = store
    first = state.bootstrap("campaign-a", goal="first goal", max_experiments=4)
    second = state.bootstrap("campaign-a", goal="replacement", max_experiments=99)

    assert first["_id"] == second["_id"] == "campaign-a"
    assert second["goal"] == "first goal"
    assert second["max_experiments"] == 4
    assert second["revision"] == 0
    assert second["stage"] == "OBSERVE"


def test_lease_acquire_renew_expire_and_release(store):
    state, now = store
    state.bootstrap("c")
    lease = state.acquire_lease("c", "worker-a", ttl_s=10)
    assert lease["lease"]["owner"] == "worker-a"
    assert state.acquire_lease("c", "worker-b", ttl_s=10) is None

    renewed = state.renew_lease("c", "worker-a", lease_token=lease["lease"]["token"], ttl_s=20)
    assert renewed["lease"]["until"] == 1020
    assert state.release_lease("c", "worker-a", lease_token=lease["lease"]["token"]) is True
    next_lease = state.acquire_lease("c", "worker-b", ttl_s=10)
    assert next_lease is not None

    now[0] += 11
    with pytest.raises(LeaseLost):
        state.renew_lease("c", "worker-b", ttl_s=5)
    reacquired = state.acquire_lease("c", "worker-b", ttl_s=5)
    renewed = state.renew_lease("c", "worker-b", lease_token=reacquired["lease"]["token"], ttl_s=5)
    assert renewed["lease"]["until"] == 1016


def test_renew_requires_live_lease_and_release_requires_owner(store):
    state, _ = store
    state.bootstrap("c")
    state.acquire_lease("c", "worker-a", ttl_s=1)
    assert state.release_lease("c", "worker-b") is False
    with pytest.raises(LeaseLost):
        state.renew_lease("c", "worker-b")


def test_transition_checks_revision_and_live_lease_and_writes_immutable_checkpoint(store):
    state, _ = store
    state.bootstrap("c")
    lease = state.acquire_lease("c", "worker-a")["lease"]
    changed = state.transition(
        "c", expected_revision=0, stage="PROPOSE", owner="worker-a",
        lease_token=lease["token"],
        updates={"active_experiment_id": "exp-1"},
        checkpoint={"goal": "lower cost", "open": ["exp-1"]},
    )
    assert changed["revision"] == 1
    assert changed["stage"] == "PROPOSE"
    assert state.checkpoint_history("c")[0]["checkpoint"]["open"] == ["exp-1"]

    with pytest.raises(RevisionConflict):
        state.transition("c", expected_revision=0, stage="OBSERVE", owner="worker-a", lease_token=lease["token"])
    with pytest.raises(ValueError):
        state.transition("c", expected_revision=1, stage="OBSERVE", lease_token=lease["token"], updates={"revision": 8})


def test_transition_rejects_another_or_expired_lease(store):
    state, now = store
    state.bootstrap("c")
    state.acquire_lease("c", "worker-a", ttl_s=2)
    with pytest.raises(LeaseLost):
        state.transition("c", expected_revision=0, stage="PROPOSE", owner="worker-b")
    now[0] += 3
    with pytest.raises(LeaseLost):
        state.transition("c", expected_revision=0, stage="PROPOSE", owner="worker-a")


def test_budget_debits_are_idempotent_and_cannot_exceed_limit(store):
    state, _ = store
    state.bootstrap("c", max_experiments=2)
    first, debited = state.debit_budget("c", "exp-1")
    duplicate, duplicate_debited = state.debit_budget("c", "exp-1")
    second, second_debited = state.debit_budget("c", "exp-2")

    assert debited is True and duplicate_debited is False and second_debited is True
    assert first["experiments_spent"] == duplicate["experiments_spent"] == 1
    assert second["experiments_spent"] == 2
    with pytest.raises(BudgetExceeded):
        state.debit_budget("c", "exp-3")


def test_indexes_are_idempotent(store):
    state, _ = store
    ensure_indexes(state.database)
    ensure_indexes(state.database)
    assert state.database.campaign_checkpoints.index_information()
