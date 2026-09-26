"""Performance and reliability API. Nothing here applies resources or runs commands."""
import io
import json
import time
import zipfile
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from common import db
from production.evidence import fixtures, normalize_evidence
from production.harness import ProductionDB, compile_context, manifest_view, record_lesson, store
from production.operations import analyze, change_id_for, llm_ops_advisor, verify_change

router = APIRouter(prefix="/api/production/ops")

# Which fixtures each page offers: exports to diagnose, then exports observed after a change.
FIXTURES = {
    "performance": {"diagnose": ["overload", "underutilized"],
                    "verify": ["overload-recovered", "overload-persists", "underutilized-recovered"]},
    "reliability": {"diagnose": ["oom", "overload", "readiness", "slow-startup", "gateway"],
                    "verify": ["oom-recovered", "overload-recovered", "overload-persists", "readiness-recovered",
                               "slow-startup-recovered", "gateway-recovered"]},
}


class EvidenceRequest(BaseModel):
    base_id: str = Field(min_length=1, max_length=80)
    payload: Any


class AnalyzeRequest(BaseModel):
    base_id: str = Field(min_length=1, max_length=80)
    evidence_id: str = Field(min_length=1, max_length=80)
    slo_p95_ms: float | None = Field(default=None, gt=0)
    max_error_rate: float | None = Field(default=None, ge=0, le=1)


class VerifyRequest(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=80)


def _plain(row):
    return {k: v for k, v in row.items() if k != "_id"} if row else None


def _base(base_id):
    d = db.db()
    row = d.production_changes.find_one({"change_id": base_id}) or d.production_plans.find_one({"plan_id": base_id})
    if not row:
        raise HTTPException(404, f"No plan or change {base_id!r}")
    return _plain(row)


def _evidence(evidence_id, attached_to):
    row = db.db().production_evidence.find_one({"evidence_id": evidence_id})
    if not row:
        raise HTTPException(404, "Evidence not found")
    if attached_to not in row.get("base_ids", []):
        raise HTTPException(422, f"Evidence was not imported for {attached_to}")
    return _plain(row)


@router.get("/fixtures")
def get_fixtures(role: Literal["performance", "reliability"]):
    all_fixtures = fixtures()
    return {stage: {name: all_fixtures[name] for name in names} for stage, names in FIXTURES[role].items()}


@router.post("/evidence", status_code=201)
def import_evidence(request: EvidenceRequest):
    """Normalize and redact one export, and attach it to the plan or change it was observed on."""
    _base(request.base_id)
    payload = request.payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"format": "logs", "logs": payload}
    record = normalize_evidence(payload)
    db.db().production_evidence.update_one(
        {"evidence_id": record["evidence_id"]},
        {"$setOnInsert": {**record, "created_at": time.time()}, "$addToSet": {"base_ids": request.base_id}},
        upsert=True)
    return record


@router.post("/{role}/analyze")
def run_analysis(role: Literal["performance", "reliability"], request: AnalyzeRequest):
    base = _base(request.base_id)
    evidence = _evidence(request.evidence_id, request.base_id)
    targets = {"slo_p95_ms": request.slo_p95_ms, "max_error_rate": request.max_error_rate}
    d = db.db()
    existing = d.production_changes.find_one({"change_id": change_id_for(base, evidence, role, targets)})
    if existing:  # same base + evidence + targets: return the recorded decision, never a duplicate
        return _with_verifications(_plain(existing))
    packet = compile_context(d, campaign_id=f"ops-{base['plan_id']}", lineage=base["plan_id"],
                             query=f"{role} {' '.join(e.get('type', '') for e in evidence.get('events', []) if isinstance(e, dict))}")
    avoid = [json.loads(l["scope"]["patch"]) for l in ProductionDB(d).lessons.find(
        {"scope.lineage": base["plan_id"], "scope.patch": {"$exists": True}, "confirmed": 0, "superseded_by": None})]
    row = analyze(base, evidence, role, targets, advisor=llm_ops_advisor, avoid=avoid)
    row["memory"] = manifest_view(packet)
    row["created_at"] = time.time()
    d.production_changes.insert_one(dict(row))
    _ops_checkpoint(base["plan_id"], f"{row['role_name']}: {row['diagnosis'].get('fault') or 'no fault'} -> {row['status']}",
                    row["change_id"])
    return _with_verifications(row)


def _ops_checkpoint(lineage, summary, evidence_id):
    """Each analysis or verification is a CAS transition on the lineage's durable operations campaign."""
    s = store(db.db())
    cid = f"ops-{lineage}"
    campaign = s.bootstrap(cid, goal="Keep this deployment healthy and efficient", max_experiments=1000,
                           initial_state={"summary": "Operations started", "decisions": []})
    cp = campaign.get("checkpoint") or {}
    s.transition(cid, expected_revision=campaign["revision"], stage="CHECKPOINT", checkpoint={
        "summary": summary, "decisions": (cp.get("decisions", []) + [summary])[-12:],
        "evidence_ids": [evidence_id], "updated_at": time.time()})


def _with_verifications(row):
    row["verifications"] = list(db.db().production_verifications.find(
        {"change_id": row["change_id"]}, {"_id": 0}).sort("created_at", -1))
    return row


@router.get("/changes")
def list_changes(role: Literal["performance", "reliability"] | None = None):
    query = {"role": role} if role else {}
    fields = {"_id": 0, "files": 0, "input": 0, "steps": 0, "diff": 0}
    return list(db.db().production_changes.find(query, fields).sort("created_at", -1).limit(30))


@router.get("/changes/{change_id}")
def get_change(change_id: str):
    row = db.db().production_changes.find_one({"change_id": change_id})
    if not row:
        raise HTTPException(404, "Change not found")
    return _with_verifications(_plain(row))


@router.post("/changes/{change_id}/verify")
def verify(change_id: str, request: VerifyRequest):
    """Record a new, immutable verification; earlier verifications and the proposal are never rewritten."""
    change = get_change(change_id)
    evidence = _evidence(request.evidence_id, change_id)
    try:
        result = verify_change(change, evidence)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["created_at"] = time.time()
    db.db().production_verifications.update_one({"verification_id": result["verification_id"]},
                                                {"$setOnInsert": result}, upsert=True)
    if result["outcome"] in {"confirmed", "contradicted"}:
        d = change["diagnosis"]
        record_lesson(db.db(), record={
            "experiment_id": result["verification_id"], "hypothesis": f"{change['role_name']}: {d['diagnosis']}",
            "decision": result["outcome"], "reason": "; ".join(result["reasons"]),
            "confidence": 0.9 if result["provenance"] == "imported" else 0.6},
            scope={"lineage": change["plan_id"], "fault": d["fault"], "patch": json.dumps(d["patch"], sort_keys=True),
                   "provenance": result["provenance"]},
            confirmed=result["outcome"] == "confirmed")
        _ops_checkpoint(change["plan_id"], f"verified {change_id}: {result['outcome']}", result["verification_id"])
    return get_change(change_id)


@router.get("/changes/{change_id}/download")
def download(change_id: str):
    row = get_change(change_id)
    if not row["changes"]:
        raise HTTPException(404, "This analysis proposed no change bundle")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        files = {**row["files"], "change.diff": row["diff"],
                 "validation.json": json.dumps(row["validation"], indent=2)}
        for path, text in sorted(files.items()):
            info = zipfile.ZipInfo(f"{change_id}/{path}")
            info.external_attr = (0o755 if path.endswith(".sh") else 0o644) << 16
            zf.writestr(info, text)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{change_id}.zip"'})


def install(app):
    app.include_router(router)


# ---------------------------------------------------------------- long-horizon evolution campaigns
import threading

from production import evolution

_workers: dict[str, threading.Thread] = {}


class CampaignRequest(BaseModel):
    base_id: str = Field(min_length=1, max_length=80)
    slo_p95_ms: float | None = Field(default=None, gt=0)
    max_error_rate: float = Field(default=0.05, ge=0, le=1)
    budget: int = Field(default=12, ge=1, le=50)
    timeline: list[dict] | None = None


def _run(campaign_id):
    campaign = evolution.EvolutionCampaign(db.db(), campaign_id)
    try:
        evolution.run_until_idle(campaign)
    except Exception:  # noqa: BLE001  (recorded on the campaign as last_error)
        pass


def _start(campaign_id):
    worker = _workers.get(campaign_id)
    if worker is None or not worker.is_alive():
        _workers[campaign_id] = threading.Thread(target=_run, args=(campaign_id,), daemon=True)
        _workers[campaign_id].start()


@router.get("/campaigns/preview")
def preview_campaign(base_id: str):
    base = _base(base_id)
    return {"timeline": evolution.default_timeline(base), "slo_p95_ms": evolution.suggested_slo_ms(base)}


@router.post("/campaigns", status_code=201)
def create_campaign(request: CampaignRequest):
    base = _base(request.base_id)
    timeline = request.timeline or evolution.default_timeline(base)
    policy = {"slo_p95_ms": request.slo_p95_ms or evolution.suggested_slo_ms(base),
              "max_error_rate": request.max_error_rate}
    campaign_id = evolution.EvolutionCampaign.campaign_id_for(base, timeline, policy)
    evolution.EvolutionCampaign(db.db(), campaign_id).create(base, timeline, policy, request.budget)
    _start(campaign_id)
    return get_campaign(campaign_id)


@router.post("/campaigns/{campaign_id}/resume")
def resume_campaign(campaign_id: str):
    get_campaign(campaign_id)
    _start(campaign_id)
    return get_campaign(campaign_id)


@router.get("/campaigns")
def list_campaigns(lineage: str | None = None):
    pdb = ProductionDB(db.db())
    query = {"campaign_id": {"$regex": "^evo-"}, **({"lineage": lineage} if lineage else {})}
    rows = pdb.campaigns.find(query, {"_id": 0, "campaign_id": 1, "stage": 1, "lineage": 1, "base_id": 1,
                                      "checkpoint.summary": 1, "checkpoint.revision": 1, "updated_at": 1})
    return list(rows.sort("updated_at", -1).limit(20))


@router.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str):
    pdb = ProductionDB(db.db())
    c = pdb.campaigns.find_one({"_id": campaign_id}, {"_id": 0, "budget_debit_keys": 0, "lease.token": 0})
    if not c:
        raise HTTPException(404, "Campaign not found")
    windows = list(pdb.metric_windows.find({"campaign_id": campaign_id}, {"_id": 0}).sort("ts", 1))
    experiments = list(pdb.experiments.find({"campaign_id": campaign_id}, {"_id": 0}).sort("created_at", 1))
    for exp in experiments:
        m = pdb.context_manifests.find_one({"manifest_id": exp.get("manifest_id")}, {"_id": 0, "included.content": 0})
        exp["context"] = m and {"tokens": sum(i["token_estimate"] for i in m["included"]), "budget": m["token_budget"],
                                "included": [{k: i[k] for k in ("item_id", "memory_type", "reason", "token_estimate")}
                                             for i in m["included"]], "excluded": len(m["excluded"])}
        if exp.get("lesson_id"):
            lesson = pdb.lessons.find_one({"lesson_id": exp["lesson_id"]}, {"_id": 0, "claim": 1, "confirmed": 1, "supersedes": 1})
            exp["lesson"] = lesson
    revisions = list(pdb.revisions.find({"campaign_id": campaign_id}, {"_id": 0, "files": 0, "plan": 0}).sort("revision", 1))
    events = list(pdb.events.find({"campaign_id": campaign_id}, {"_id": 0}).sort("ts", -1).limit(80))
    c["running"] = bool(_workers.get(campaign_id) and _workers[campaign_id].is_alive())
    return {"campaign": c, "windows": windows, "experiments": experiments, "revisions": revisions, "events": events}


@router.get("/campaigns/{campaign_id}/revisions/{revision}/download")
def download_revision(campaign_id: str, revision: int):
    row = ProductionDB(db.db()).revisions.find_one({"campaign_id": campaign_id, "revision": revision})
    if not row:
        raise HTTPException(404, "Revision not found")
    buf = io.BytesIO()
    name = f"{campaign_id}-rev{revision}"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, text in sorted({**row["files"], "revision.diff": row.get("diff", "")}.items()):
            info = zipfile.ZipInfo(f"{name}/{path}")
            info.external_attr = (0o755 if path.endswith(".sh") else 0o644) << 16
            zf.writestr(info, text)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})
