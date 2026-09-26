"""Durable production planning service. No deployment or shell execution ports."""
import copy
import io
import json
import threading
import time
import uuid
import zipfile
from pymongo import ReturnDocument
from production.contracts import TaskRequest
from production.reasoning import ProductionReasoner, compile_context, digest

STAGES = ["ASSESS", "CONTEXT", "REASON", "RENDER", "VALIDATE", "PUBLISH", "LEARN", "COMPLETE"]
PATCH_LIMITS = {"max_num_seqs": (1, 4096), "max_model_len": (128, 1048576),
                "startup_probe_failure_threshold": (1, 120),
                "readiness_initial_delay_seconds": (0, 600),
                "readiness_timeout_seconds": (1, 60), "upstream_timeout_seconds": (1, 600),
                "retry_max_attempts": (0, 0)}


def apply_patch_to_plan(plan, patch):
    """Apply only bounded renderer-supported fields to an immutable copy."""
    candidate = copy.deepcopy(plan)
    for group, fields in patch.items():
        if group not in {"config", "allocation"} or not isinstance(fields, dict):
            raise ValueError("unsupported plan patch group")
        for name, value in fields.items():
            limits = PATCH_LIMITS.get(name) if group == "config" else {"replicas": (1, 128)}.get(name)
            if limits is None or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"unsupported plan patch field: {group}.{name}")
            if not limits[0] <= value <= limits[1]:
                raise ValueError(f"patch value outside bounds: {name}")
            candidate.setdefault(group, {})[name] = value
    return candidate


def plain(row):
    return {k: v for k, v in row.items() if k != "_id"} if row else None


class PlanningService:
    def __init__(self, database, reasoner=None):
        self.db = database
        self.reasoner = reasoner or ProductionReasoner()
        self.owner = uuid.uuid4().hex
        self.db.production_tasks.create_index("task_id", unique=True)
        self.db.production_tasks.create_index("idempotency", unique=True)
        self.db.production_bundles.create_index("bundle_id", unique=True)
        self.db.production_evidence.create_index("evidence_id", unique=True)
        self.db.production_checkpoints.create_index([("task_id", 1), ("revision", 1)], unique=True)
        self.db.production_lessons.create_index("lesson_id", unique=True)
        self.db.production_contexts.create_index("manifest_id", unique=True)
        self.db.production_verifications.create_index("verification_id", unique=True)
        self.db.production_tasks.create_index([("status", 1), ("priority", 1), ("next_run_at", 1)])

    def create(self, request):
        request = TaskRequest.model_validate(request)
        data = request.model_dump()
        env = self.db.production_environments.find_one({"_id": request.environment_id})
        if not env:
            self.db.production_environments.update_one({"_id": request.environment_id},
                {"$setOnInsert": {"revision": 0}}, upsert=True)
            env = self.db.production_environments.find_one({"_id": request.environment_id})
        if request.kind != "provision":
            base = self.get(request.base_task_id)
            if not base or base["environment_id"] != request.environment_id or not base.get("plan"):
                raise ValueError("a base task with a plan from this environment is required")
            if base.get("published_revision") != env["revision"]:
                raise ValueError("base plan is stale; select the current environment plan")
            evidence = self.db.production_evidence.find_one({"evidence_id": request.evidence_id})
            if not evidence or evidence["environment_id"] != request.environment_id:
                raise ValueError("evidence from this environment is required")
            if evidence.get("plan_revision") != env["revision"]:
                raise ValueError("evidence refers to a different plan revision")
            data["inventory"] = base["input"]["inventory"]
            data["workload"] = base["input"]["workload"]
        key = digest({"environment": request.environment_id,
                      "key": request.idempotency_key or data})
        existing = self.db.production_tasks.find_one({"idempotency": key})
        if existing:
            return plain(existing)
        task_id = "task-" + uuid.uuid4().hex
        row = {"task_id": task_id, "schema_version": 1, "environment_id": request.environment_id,
               "kind": request.kind, "input": data, "input_hash": digest(data),
               "base_revision": env["revision"], "idempotency": key, "stage": "ASSESS",
               "status": "queued", "revision": 0, "attempts": 0,
               "priority": {"repair": 0, "provision": 1, "optimize": 2}[request.kind],
               "created_at": time.time(), "updated_at": time.time(), "next_run_at": 0}
        self.db.production_tasks.update_one({"idempotency": key}, {"$setOnInsert": row}, upsert=True)
        return plain(self.db.production_tasks.find_one({"idempotency": key}))

    def get(self, task_id):
        return plain(self.db.production_tasks.find_one({"task_id": task_id}))

    def import_evidence(self, payload):
        from production.evidence import normalize_evidence
        record = normalize_evidence(payload)
        self.db.production_evidence.update_one({"evidence_id": record["evidence_id"]},
                                               {"$setOnInsert": record}, upsert=True)
        return plain(record)

    def tick(self):
        now = time.time()
        task = self.db.production_tasks.find_one_and_update(
            {"stage": {"$ne": "COMPLETE"}, "status": {"$nin": ["failed", "cancelled", "stale"]},
             "next_run_at": {"$lte": now},
             "$or": [{"lease_until": {"$lt": now}}, {"lease_until": {"$exists": False}}]},
            {"$set": {"lease_owner": self.owner, "lease_until": now + 90, "status": "running"}},
            sort=[("priority", 1), ("created_at", 1)], return_document=ReturnDocument.AFTER)
        if not task:
            return False
        if task.get("checkpoint"):
            saved = task["checkpoint"]
            self.db.production_checkpoints.update_one(
                {"task_id": task["task_id"], "revision": saved["revision"]},
                {"$setOnInsert": saved}, upsert=True)
        stop = threading.Event()
        def renew():
            while not stop.wait(20):
                self.db.production_tasks.update_one(
                    {"task_id": task["task_id"], "lease_owner": self.owner},
                    {"$set": {"lease_until": time.time() + 90}})
        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            updates = self.advance(plain(task))
            stage = updates.pop("stage", STAGES[min(STAGES.index(task["stage"]) + 1, len(STAGES)-1)])
            rev = task["revision"] + 1
            checkpoint = {"task_id": task["task_id"], "revision": rev, "stage": stage,
                          "input_hash": task["input_hash"], "output_hash": digest(updates),
                          "created_at": time.time()}
            result = self.db.production_tasks.update_one(
                {"task_id": task["task_id"], "revision": task["revision"],
                 "lease_owner": self.owner, "lease_until": {"$gt": time.time()}},
                {"$set": {**updates, "stage": stage, "revision": rev, "attempts": 0,
                          "checkpoint": checkpoint,
                          "updated_at": time.time()}, "$unset": {"error": ""}})
            if result.modified_count:
                self.db.production_checkpoints.update_one(
                    {"task_id": task["task_id"], "revision": rev}, {"$setOnInsert": checkpoint}, upsert=True)
        except Exception as exc:
            attempts = task.get("attempts", 0) + 1
            self.db.production_tasks.update_one({"task_id": task["task_id"], "lease_owner": self.owner},
                {"$set": {"attempts": attempts, "error": str(exc)[:1000],
                          "status": "failed" if attempts >= 3 else "retrying",
                          "next_run_at": time.time() + 2 ** attempts}})
        finally:
            stop.set()
            self.db.production_tasks.update_one({"task_id": task["task_id"], "lease_owner": self.owner},
                {"$unset": {"lease_owner": "", "lease_until": ""}})
        return True

    def advance(self, task):
        from production.catalog import select_plan
        from production.artifacts import render_bundle
        from production.evidence import diagnose
        inp, stage = task["input"], task["stage"]
        evidence = plain(self.db.production_evidence.find_one({"evidence_id": inp.get("evidence_id")}))
        if stage == "ASSESS":
            if task["kind"] == "provision":
                plan = select_plan(inp["inventory"], inp["workload"], inp.get("recipe_id"))
                plan.setdefault("inventory", inp["inventory"])
                plan.setdefault("workload", inp["workload"])
                return {"plan": plan}
            base = self.get(inp["base_task_id"])
            plan = copy.deepcopy(base["plan"])
            diagnosis = diagnose(plan, evidence, task["kind"])
            if diagnosis.get("status") == "remediation_proposed":
                plan = apply_patch_to_plan(plan, diagnosis.get("patch", {}))
            return {"plan": plan, "diagnosis": diagnosis, "previous_plan": base["plan"]}
        if stage == "CONTEXT":
            from production.catalog import catalog
            context_plan = {**task["plan"], "candidate_catalog": catalog(),
                            "diagnosis": task.get("diagnosis")}
            manifest = compile_context(self.db, task, context_plan, evidence)
            self.db.production_contexts.update_one({"manifest_id": manifest["manifest_id"]},
                                                   {"$setOnInsert": manifest}, upsert=True)
            return {"context_manifest": manifest}
        if stage == "REASON":
            role = {"provision": "Provisioning Architect", "repair": "Reliability Engineer",
                    "optimize": "Performance Engineer"}[task["kind"]]
            fallback = {"rationale": task.get("diagnosis", {}).get("diagnosis") or
                        "Selected from compatible catalog recipes using task and estimated GPU memory fit.",
                        "evidence_ids": [x["item_id"] for x in task["context_manifest"]["included"]],
                        "uncertainties": ["Runtime performance and task quality are unverified."]}
            reasoning = self.reasoner.decide(role, task["context_manifest"], fallback)
            plan = task["plan"]
            try:
                if task["kind"] == "provision" and reasoning.get("source") == "openrouter":
                    plan = select_plan(inp["inventory"], inp["workload"],
                                       inp.get("recipe_id") or reasoning.get("recipe_id") or plan["recipe_id"],
                                       model_id=reasoning.get("model_id") or plan["model"]["model_id"])
                    reasoning["patch_application"] = (
                        "Catalog model and recipe selections accepted; free-form model patch ignored")
                elif reasoning.get("patch") and task.get("diagnosis", {}).get("status") == "remediation_proposed":
                    plan = apply_patch_to_plan(plan, reasoning["patch"])
                    reasoning["patch_application"] = "Bounded patch accepted for deterministic artifact validation"
                else:
                    reasoning["patch_application"] = "No configuration patch applied"
            except (ValueError, KeyError, TypeError) as exc:
                plan = task["plan"]
                reasoning["patch_application"] = f"Model change rejected: {exc}; retained validated catalog/detector recommendation"
            return {"reasoning": reasoning, "plan": plan}
        if stage == "RENDER":
            bundle = render_bundle(task["plan"], task.get("previous_plan"))
            bundle["task_id"] = task["task_id"]
            bundle["environment_id"] = task["environment_id"]
            bundle_id = "bundle-" + digest(bundle)
            bundle["bundle_id"] = bundle_id
            self.db.production_bundles.update_one({"bundle_id": bundle_id}, {"$setOnInsert": bundle}, upsert=True)
            return {"bundle_id": bundle_id, "validation": bundle["validation"]}
        if stage == "VALIDATE":
            env = self.db.production_environments.find_one({"_id": task["environment_id"]})
            if env["revision"] != task["base_revision"]:
                return {"stage": "COMPLETE", "status": "stale",
                        "error": "Environment plan changed; recreate against the current revision."}
            return {"status": task["validation"]["status"]}
        if stage == "PUBLISH":
            # CAS publishes a proposed plan, never an applied infrastructure state.
            env = self.db.production_environments.find_one({"_id": task["environment_id"]})
            if env.get("last_task_id") == task["task_id"]:
                return {"published_revision": env["revision"]}
            if task.get("diagnosis", {}).get("status") == "awaiting_evidence":
                return {"status": "awaiting_evidence"}
            if task["validation"]["status"] != "offline_validated":
                return {"status": task["validation"]["status"]}
            result = self.db.production_environments.find_one_and_update(
                {"_id": task["environment_id"], "revision": task["base_revision"]},
                {"$inc": {"revision": 1}, "$set": {"last_task_id": task["task_id"],
                  "bundle_id": task["bundle_id"], "deployment_status": "not_applied"}},
                return_document=ReturnDocument.AFTER)
            if not result:
                return {"stage": "COMPLETE", "status": "stale"}
            return {"published_revision": result["revision"]}
        if stage == "LEARN":
            manifest = task["context_manifest"]
            outcome = {"validation": task["validation"], "diagnosis": task.get("diagnosis"),
                       "bundle_id": task["bundle_id"], "runtime_verified": False}
            lesson_id = "lesson-" + task["task_id"]
            curator_manifest = compile_context(self.db, task, {**task["plan"], "terminal_outcome": outcome}, evidence)
            curated = self.reasoner.decide("Memory Curator", curator_manifest, {
                "rationale": f"Generated {task['kind']} bundle; validation: {task['validation']['status']}. Runtime unverified.",
                "evidence_ids": ["plan"]})
            lesson = {**manifest["scope"], "lesson_id": lesson_id, "task_id": task["task_id"],
                      "summary": curated["rationale"], "outcome": outcome,
                      "source_ids": [task["bundle_id"]], "source_hash": digest(outcome),
                      "version": 1, "created_at": time.time()}
            self.db.production_lessons.update_one({"lesson_id": lesson_id}, {"$setOnInsert": lesson}, upsert=True)
            status = task["validation"]["status"]
            if task.get("diagnosis", {}).get("status") == "awaiting_evidence":
                status = "awaiting_evidence"
            elif task["kind"] != "provision" and status == "offline_validated":
                status = "remediation_proposed"
            return {"lesson_id": lesson_id, "status": status, "runtime_verified": False}
        return {}

    def verification(self, task_id, evidence_id):
        from production.evidence import verify
        task = self.get(task_id)
        evidence = plain(self.db.production_evidence.find_one({"evidence_id": evidence_id}))
        if not task or not evidence or evidence.get("environment_id") != task["environment_id"]:
            raise ValueError("task and evidence must belong to the same environment")
        if evidence.get("applied_bundle_id") != task.get("bundle_id"):
            raise ValueError("verification evidence must identify the externally applied bundle")
        if evidence.get("plan_revision") != task.get("published_revision"):
            raise ValueError("verification evidence must match the published plan revision")
        conditions = task.get("diagnosis", {}).get("conditions") or {
            "window_samples": 3,
            "coverage_window_seconds": 60,
            "plan_revision": task.get("published_revision"),
            "metrics": {"ready": {"eq": 1}, "error_rate": {"lte": 0.05},
                        "latency_p95_ms": {"lte": task["input"]["workload"]["slo_p95_ms"]}}}
        conditions = copy.deepcopy(conditions)
        conditions.setdefault("environment_id", task["environment_id"])
        conditions.setdefault("plan_revision", task.get("published_revision"))
        conditions.setdefault("coverage_window_seconds", 60)
        result = verify(conditions, evidence)
        result.update(task_id=task_id, evidence_id=evidence_id,
                      provenance=evidence.get("provenance"), created_at=time.time())
        result["verification_id"] = digest({"task": task_id, "evidence": evidence_id})
        self.db.production_verifications.update_one({"verification_id": result["verification_id"]},
                                                   {"$setOnInsert": result}, upsert=True)
        self.db.production_tasks.update_one({"task_id": task_id}, {"$set": {"verification": result,
            "runtime_verified": result.get("outcome") == "confirmed" and evidence.get("provenance") == "imported"}})
        # Verification is a new immutable lesson, never a rewrite of planning evidence.
        original = self.db.production_lessons.find_one({"lesson_id": task.get("lesson_id")})
        if original:
            scope = task["context_manifest"]["scope"]
            lesson = {**scope, "provenance": evidence.get("provenance"),
                      "lesson_id": "verification-" + result["verification_id"],
                      "summary": f"{result['outcome']}: imported recovery predicates for {task['bundle_id']}",
                      "outcome": result, "source_ids": [evidence_id, task["bundle_id"]],
                      "source_hash": digest(result), "version": 1, "created_at": time.time(),
                      "supersedes": [original["lesson_id"]] if result.get("outcome") == "confirmed" else [],
                      "contradictions": [original["lesson_id"]] if result.get("outcome") == "contradicted" else []}
            self.db.production_lessons.update_one({"lesson_id": lesson["lesson_id"]},
                                                  {"$setOnInsert": lesson}, upsert=True)
        return result

    def attest_applied_bundle(self, evidence_id, bundle_id):
        """Create immutable evidence enriched by an explicit operator attestation."""
        base = plain(self.db.production_evidence.find_one({"evidence_id": evidence_id}))
        if not base:
            raise ValueError("evidence not found")
        record = copy.deepcopy(base)
        record.pop("evidence_id", None)
        record["applied_bundle_id"] = str(bundle_id)
        record["attested_at"] = time.time()
        record["attestation_source_id"] = evidence_id
        record["evidence_id"] = digest(record)
        self.db.production_evidence.update_one(
            {"evidence_id": record["evidence_id"]}, {"$setOnInsert": record}, upsert=True)
        return record

    def bundle_zip(self, bundle_id):
        bundle = self.db.production_bundles.find_one({"bundle_id": bundle_id})
        if not bundle:
            raise ValueError("bundle not found")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, content in sorted(bundle["files"].items()):
                if path.startswith(("/", "\\")) or ".." in path.split("/"):
                    raise ValueError("unsafe artifact path")
                archive.writestr(path, content)
            archive.writestr("validation-report.json", json.dumps(bundle["validation"], indent=2))
        return output.getvalue()
