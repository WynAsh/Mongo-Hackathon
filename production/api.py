"""Planning API. Nothing here applies resources or runs commands."""
import io
import zipfile
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from common import db
from production.catalog import catalog
from production.contracts import PlanRequest
from production.inventory import DISCOVERY_SCRIPT, parse_inventory
from production.advisor import llm_advisor
from production.planner import build_plan

router = APIRouter(prefix="/api/production")


@router.get("/catalog")
def get_catalog():
    return catalog()


@router.get("/inventory/discovery")
def discovery():
    return {"filename": "discover-hardware.sh", "content": DISCOVERY_SCRIPT}


@router.post("/inventory/import")
async def import_inventory(request: Request):
    try:
        return parse_inventory(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))


@router.post("/plan")
def plan(request: PlanRequest):
    try:
        result = build_plan(request.model_dump(), advisor=llm_advisor)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    db.db().production_plans.replace_one({"plan_id": result["plan_id"]}, result, upsert=True)
    return result


@router.get("/plans")
def plans():
    return list(db.db().production_plans.find({}, {"_id": 0, "files": 0}).sort("_id", -1).limit(20))


@router.get("/plans/{plan_id}/download")
def download(plan_id: str):
    row = db.db().production_plans.find_one({"plan_id": plan_id})
    if not row:
        raise HTTPException(404, "Plan not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, text in sorted(row["files"].items()):
            info = zipfile.ZipInfo(f"{plan_id}/{path}")
            info.external_attr = (0o755 if path.endswith(".sh") else 0o644) << 16
            zf.writestr(info, text)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{plan_id}.zip"'})


def install(app):
    app.include_router(router)
