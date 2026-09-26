"""Additive planning APIs; no endpoint applies generated resources."""
import asyncio
import threading
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from common import db
from production.contracts import TaskRequest
from production.service import PlanningService, plain

router = APIRouter(prefix="/api/production")


@lru_cache(maxsize=1)
def service():
    return PlanningService(db.db())


async def bounded_json(request: Request):
    import json
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 2_000_000:
            raise HTTPException(413, "Import exceeds 2 MB")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except Exception:
        raise HTTPException(400, "Expected JSON body")


@router.get("/catalog")
def get_catalog():
    from production.catalog import catalog
    return catalog()


@router.get("/inventory/discovery")
def discovery():
    from production.inventory import DISCOVERY_SCRIPT
    return {"filename": "discover-hardware.sh", "content": DISCOVERY_SCRIPT,
            "execution": "not_executed"}


@router.post("/inventory/import")
async def inventory(request: Request):
    from production.inventory import parse_inventory
    payload = await bounded_json(request)
    try:
        return parse_inventory(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))


@router.get("/tasks")
def tasks():
    return [plain(t) for t in service().db.production_tasks.find({}, {"context_manifest.included": 0})
            .sort("created_at", -1).limit(50)]


@router.post("/tasks", status_code=201)
def create_task(request: TaskRequest):
    try:
        return service().create(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/tasks/{task_id}")
def task(task_id: str):
    result = service().get(task_id)
    if not result:
        raise HTTPException(404, "Task not found")
    return result


@router.post("/evidence", status_code=201)
async def evidence(request: Request):
    payload = await bounded_json(request)
    try:
        return await asyncio.to_thread(service().import_evidence, payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc))


@router.get("/fixtures")
def fixtures():
    from production.evidence import fixtures
    return fixtures()


@router.get("/bundles/{bundle_id}")
def bundle(bundle_id: str):
    result = plain(service().db.production_bundles.find_one({"bundle_id": bundle_id}))
    if not result:
        raise HTTPException(404, "Bundle not found")
    return result


@router.get("/bundles/{bundle_id}/download")
def download(bundle_id: str):
    try:
        content = service().bundle_zip(bundle_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return Response(content, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{bundle_id}.zip"'})


@router.post("/tasks/{task_id}/verify")
async def verify_task(task_id: str, request: Request):
    payload = await bounded_json(request)
    try:
        evidence_id = payload["evidence_id"]
        if payload.get("applied_bundle_id"):
            record = await asyncio.to_thread(
                service().attest_applied_bundle, evidence_id, payload["applied_bundle_id"])
            evidence_id = record["evidence_id"]
        return await asyncio.to_thread(service().verification, task_id, evidence_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc))


def worker(stop):
    while not stop.wait(0.5):
        try:
            service().tick()
        except Exception as exc:
            print(f"[production-worker] {type(exc).__name__}: {exc}", flush=True)
            stop.wait(3)


def install(app):
    app.include_router(router)
    @app.on_event("startup")
    def start():
        app.state.production_stop = threading.Event()
        threading.Thread(target=worker, args=(app.state.production_stop,), daemon=True).start()
    @app.on_event("shutdown")
    def shutdown():
        app.state.production_stop.set()
