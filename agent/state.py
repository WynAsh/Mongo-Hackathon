"""Durable campaign state and checkpoint operations.

The workflow owns transitions; this module supplies atomic persistence primitives.
Campaign documents intentionally use plain BSON-compatible dictionaries so the
state layer can evolve independently from the agent's public Pydantic contracts.
"""
from __future__ import annotations

import copy
import time
from typing import Any
from uuid import uuid4

from pymongo import ReturnDocument

from common import config, db as db_module


class CampaignStateError(RuntimeError):
    """Base class for invalid durable campaign operations."""


class RevisionConflict(CampaignStateError):
    """The campaign changed after the caller read its revision."""


class LeaseLost(CampaignStateError):
    """The caller does not hold the campaign's active lease."""


class BudgetExceeded(CampaignStateError):
    """The requested debit would exceed the campaign experiment budget."""


class CampaignStore:
    """PyMongo-compatible persistence for one or more optimization campaigns."""

    def __init__(self, database=None, *, clock=time.time):
        self.database = database if database is not None else db_module.db()
        self.clock = clock
        self.campaigns = self.database["campaigns"]
        self.checkpoints = self.database["campaign_checkpoints"]

    def bootstrap(
        self,
        campaign_id: str | None = None,
        *,
        goal: str = "Optimize serving cost while meeting latency and error objectives",
        max_experiments: int = 20,
        policy: dict[str, Any] | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a campaign once, or return its existing durable state."""
        if max_experiments < 0:
            raise ValueError("max_experiments must be non-negative")
        campaign_id = campaign_id or getattr(config, "CAMPAIGN_ID", "default")
        now = float(self.clock())
        defaults = {
            "campaign_id": campaign_id,
            "goal": goal,
            "stage": "OBSERVE",
            "incumbent_version": None,
            "active_experiment_id": None,
            "max_experiments": int(max_experiments),
            "experiments_spent": 0,
            # Bounded by max_experiments, this set makes budget debits idempotent
            # without an unbounded event list or multi-document transaction.
            "budget_debit_keys": [],
            "revision": 0,
            "checkpoint": copy.deepcopy(initial_state or {}),
            "policy": copy.deepcopy(policy or {}),
            "lease": None,
            "created_at": now,
            "updated_at": now,
            "schema_version": 1,
        }
        self.campaigns.update_one(
            {"_id": campaign_id}, {"$setOnInsert": defaults}, upsert=True
        )
        return self.get(campaign_id)

    def get(self, campaign_id: str | None = None) -> dict[str, Any] | None:
        campaign_id = campaign_id or getattr(config, "CAMPAIGN_ID", "default")
        return self.campaigns.find_one({"_id": campaign_id})

    def acquire_lease(
        self,
        campaign_id: str,
        owner: str,
        *,
        ttl_s: float = 60.0,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Acquire an absent/expired lease, or renew one already held by owner."""
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        now = float(self.clock() if now is None else now)
        lease = {"owner": owner, "token": uuid4().hex, "until": now + ttl_s}
        result = self.campaigns.find_one_and_update(
            {
                "_id": campaign_id,
                "$or": [
                    {"lease": None},
                    {"lease": {"$exists": False}},
                    {"lease.until": {"$lte": now}},
                    {"lease.owner": owner},
                ],
            },
            {"$set": {"lease": lease, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return result

    def renew_lease(
        self,
        campaign_id: str,
        owner: str,
        *,
        lease_token: str | None = None,
        ttl_s: float = 60.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        now = float(self.clock() if now is None else now)
        query = {"_id": campaign_id, "lease.owner": owner, "lease.until": {"$gt": now}}
        if lease_token is not None:
            query["lease.token"] = lease_token
        result = self.campaigns.find_one_and_update(
            query,
            {"$set": {"lease.until": now + ttl_s, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        if result is None:
            raise LeaseLost(f"active lease for campaign {campaign_id!r} is not held by {owner!r}")
        return result

    def release_lease(
        self, campaign_id: str, owner: str, *, lease_token: str | None = None,
        now: float | None = None
    ) -> bool:
        now = float(self.clock() if now is None else now)
        query = {"_id": campaign_id, "lease.owner": owner}
        if lease_token is not None:
            query["lease.token"] = lease_token
        result = self.campaigns.update_one(
            query,
            {"$set": {"lease": None, "updated_at": now}},
        )
        return result.modified_count == 1

    def transition(
        self,
        campaign_id: str,
        *,
        expected_revision: int,
        stage: str,
        updates: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        owner: str | None = None,
        lease_token: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """CAS a workflow transition and persist an immutable revision checkpoint.

        The caller supplies the revision it read. If ``owner`` is given, the same
        atomic update also verifies that its lease remains live.
        """
        now = float(self.clock() if now is None else now)
        query: dict[str, Any] = {"_id": campaign_id, "revision": expected_revision}
        if owner is not None:
            query.update({"lease.owner": owner, "lease.until": {"$gt": now}})
            if lease_token is not None:
                query["lease.token"] = lease_token
        changes = copy.deepcopy(updates or {})
        forbidden = {"_id", "revision", "stage", "created_at", "schema_version", "lease", "checkpoint"}
        if forbidden.intersection(changes):
            raise ValueError(f"updates cannot override reserved fields: {sorted(forbidden.intersection(changes))}")
        new_revision = expected_revision + 1
        changes.update({"stage": stage, "revision": new_revision, "updated_at": now})
        if checkpoint is not None:
            changes["checkpoint"] = copy.deepcopy(checkpoint)
        result = self.campaigns.find_one_and_update(
            query, {"$set": changes}, return_document=ReturnDocument.AFTER
        )
        if result is None:
            current = self.get(campaign_id)
            if current is None or current.get("revision") != expected_revision:
                raise RevisionConflict(f"campaign {campaign_id!r} is not at revision {expected_revision}")
            raise LeaseLost(f"active lease for campaign {campaign_id!r} is not held by {owner!r}")
        if checkpoint is not None:
            snapshot = {
                "campaign_id": campaign_id,
                "revision": new_revision,
                "stage": stage,
                "checkpoint": copy.deepcopy(checkpoint),
                "created_at": now,
                "schema_version": 1,
            }
            # A revision is written once. Repeating the same transition cannot
            # overwrite its historical checkpoint.
            self.checkpoints.update_one(
                {"campaign_id": campaign_id, "revision": new_revision},
                {"$setOnInsert": snapshot}, upsert=True,
            )
        return result

    def debit_budget(
        self,
        campaign_id: str,
        debit_key: str,
        *,
        amount: int = 1,
        now: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Debit experiment budget once per key; return (campaign, newly_debited)."""
        if not debit_key:
            raise ValueError("debit_key must be non-empty")
        if amount <= 0:
            raise ValueError("amount must be positive")
        now = float(self.clock() if now is None else now)
        current = self.get(campaign_id)
        if current is None:
            raise CampaignStateError(f"campaign {campaign_id!r} does not exist")
        if debit_key in current.get("budget_debit_keys", []):
            return current, False
        # The configured cap is immutable during normal workflow operation. A
        # literal bound avoids relying on server-side expression support, which
        # keeps this operation compatible with mongomock and older MongoDBs.
        remaining_cap = current.get("max_experiments", 0) - amount
        result = self.campaigns.find_one_and_update(
            {
                "_id": campaign_id,
                "budget_debit_keys": {"$ne": debit_key},
                "experiments_spent": {"$lte": remaining_cap},
            },
            {
                "$inc": {"experiments_spent": amount},
                "$addToSet": {"budget_debit_keys": debit_key},
                "$set": {"updated_at": now},
            },
            return_document=ReturnDocument.AFTER,
        )
        if result is not None:
            return result, True
        current = self.get(campaign_id)
        if current is None:
            raise CampaignStateError(f"campaign {campaign_id!r} does not exist")
        if debit_key in current.get("budget_debit_keys", []):
            return current, False
        if current.get("experiments_spent", 0) + amount > current.get("max_experiments", 0):
            raise BudgetExceeded(f"campaign {campaign_id!r} has insufficient experiment budget")
        raise CampaignStateError("budget debit failed due to a concurrent campaign update")

    def checkpoint_history(self, campaign_id: str) -> list[dict[str, Any]]:
        return list(self.checkpoints.find({"campaign_id": campaign_id}).sort("revision", 1))


def ensure_indexes(database=None) -> None:
    """Create the state layer's indexes; safe to call repeatedly."""
    d = database if database is not None else db_module.db()
    d.campaigns.create_index("campaign_id", unique=True, sparse=True)
    d.campaign_checkpoints.create_index(
        [("campaign_id", 1), ("revision", 1)], unique=True
    )
