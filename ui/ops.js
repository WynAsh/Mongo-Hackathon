// Shared UI for the Performance and Reliability pages. Each page sets window.OPS = {role, verb, ...}.
const OPS = window.OPS;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const list = (items, cls = "") => items && items.length ? `<ul class="issues ${cls}">${items.map(x => `<li>${esc(typeof x === "string" ? x : JSON.stringify(x))}</li>`).join("")}</ul>` : "";
let FIXTURES = {diagnose: {}, verify: {}};

const STATUS = {
  offline_validated: ["ready", "Offline-validated change"],
  validation_failed: ["blocked", "Validation failed"],
  blocked: ["blocked", "Blocked"],
  awaiting_evidence: ["gray", "Awaiting evidence"],
  remediation_proposed: ["ready", "Change proposed"],
  handoff: ["warn", "Hand off to Reliability"],
};
const FAULTS = {
  overload: "Queue saturation / overload", underutilized: "Idle GPU capacity", oom: "OOM / KV-cache pressure",
  readiness_timeout: "Readiness timeouts / crash loop", slow_startup: "Slow startup", gateway_failure: "Gateway timeouts / backend failures",
};

async function api(path, body) {
  const res = await fetch("/api/production" + path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

function pill([cls, text]) { return `<span class="pill ${cls}">${esc(text)}</span>`; }
function provenancePill(p) { return p === "fixture" ? pill(["sim", "Fixture evidence · simulated"]) : pill(["gray", "Imported evidence"]); }
function topology(p) { return `${p.allocation.replicas}×TP${p.allocation.tensor_parallel} · seqs ${p.engine.max_num_seqs}`; }

// ------------------------------------------------------------ bases and history

async function loadBases(select) {
  const [plans, changes] = await Promise.all([api("/plans"), api("/ops/changes")]);
  const base = $("base"), keep = select || base.value;
  base.innerHTML = "";
  if (!plans.length) base.add(new Option("No plans yet: build one on the Plan page", ""));
  plans.forEach(r => {
    base.add(new Option(`Plan ${r.plan_id.slice(0, 8)} · ${r.plan.stack} · ${r.plan.model.model_id.split("/").pop()} · ${topology(r.plan)}`, r.plan_id));
    changes.filter(c => c.plan_id === r.plan_id && c.changes.length && c.status === "offline_validated").forEach(c =>
      base.add(new Option(`   ↳ ${c.role_name} ${c.change_id} · ${c.changes.map(x => `${x.field}=${x.after}`).join(", ")}`, c.change_id)));
  });
  if (keep && [...base.options].some(o => o.value === keep)) base.value = keep;
  showBase(plans, changes);
  base.onchange = () => showBase(plans, changes);
  renderHistory(changes.filter(c => c.role === OPS.role));
}

function showBase(plans, changes) {
  const id = $("base").value, row = plans.find(p => p.plan_id === id) || changes.find(c => c.change_id === id);
  $("baseInfo").innerHTML = row ? `${esc(row.plan.stack)} · ${esc(row.plan.model.model_id)} · ${esc(topology(row.plan))}` +
    (row.plan.probes ? ` · probes ${esc(JSON.stringify(row.plan.probes))}` : "") +
    (row.plan.gateway_policy ? ` · gateway ${esc(JSON.stringify(row.plan.gateway_policy))}` : "") : "";
}

function renderHistory(changes) {
  $("history").innerHTML = changes.length ? changes.map(c => `<button type="button" class="hist" data-id="${esc(c.change_id)}">
      <b>${esc(FAULTS[c.diagnosis.fault] || "No supported fault")}</b>
      <span>${esc(c.change_id)} · ${esc((STATUS[c.status] || ["", c.status])[1])} · ${c.provenance === "fixture" ? "fixture" : "imported"}</span></button>`).join("")
    : `<div class="muted">None yet.</div>`;
  document.querySelectorAll(".hist").forEach(b => b.onclick = async () => render(await api(`/ops/changes/${b.dataset.id}`)));
}

// ------------------------------------------------------------ result

function diffHtml(diff) {
  return diff.split("\n").map(l => {
    const cls = l.startsWith("+++") || l.startsWith("---") ? "" : l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : l.startsWith("@@") ? "hunk" : "";
    return cls ? `<span class="${cls}">${esc(l)}</span>` : esc(l);
  }).join("\n");
}

function verificationHtml(v) {
  const label = v.outcome === "confirmed" ? (v.operationally_verified ? ["ready", "Operationally verified"] : ["sim", "Confirmed · simulated"])
    : v.outcome === "contradicted" ? ["blocked", "Contradicted"] : ["gray", "Awaiting evidence"];
  return `<div class="verif">${pill(label)} ${provenancePill(v.provenance)} <span class="kv">${esc(v.verification_id)} · evidence ${esc(v.evidence_id.slice(0, 12))} · ${new Date(v.created_at * 1000).toLocaleString()}</span>
    ${list(v.reasons)}${list(v.missing_evidence, "bad")}</div>`;
}

function render(c) {
  const d = c.diagnosis, x = c.explanation || {}, files = Object.keys(c.files);
  const ok = c.validation.filter(v => v.ok).length;
  $("result").innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
      <h1>${esc(FAULTS[d.fault] || "No supported fault detected")}</h1>
      <div>${pill(STATUS[c.status] || ["gray", c.status])} ${provenancePill(c.provenance)}
      ${c.changes.length ? `<a class="btn" style="text-decoration:none;margin-left:6px" href="/api/production/ops/changes/${esc(c.change_id)}/download">Download change bundle</a>` : ""}</div>
    </div>
    <div class="muted">${esc(c.change_id)} · base ${esc(c.base_id)} · ${esc(c.plan.stack)} · ${esc(topology(c.plan))}</div>
    <div class="why"><span class="pill ${String(x.decided_by).startsWith("llm") ? "ready" : ""}">${esc(OPS.name)} · ${esc(x.decided_by || "rules")}</span>
      <p>${esc(d.diagnosis)}</p>${x.summary ? `<p>${esc(x.summary)}</p>` : ""}
      ${x.hypotheses && x.hypotheses.length ? `<h2>Competing explanations</h2>${list(x.hypotheses)}` : ""}
      ${x.next_checks && x.next_checks.length ? `<h2>Checks that would confirm or refute this</h2>${list(x.next_checks)}` : ""}
      ${x.fallback_reason ? `<div class="kv">LLM unavailable, rules only: ${esc(x.fallback_reason)}</div>` : ""}</div>
    ${d.memory_note ? `<div class="why"><span class="pill sim">Memory changed the remedy</span><p>${esc(d.memory_note)}</p></div>` : ""}
    ${c.memory ? `<details class="why"><summary><span class="pill gray">Memory · ${c.memory.included.length} items · ${c.memory.tokens}/${c.memory.budget} tokens</span></summary>
      <table>${c.memory.included.map(i => `<tr><td class="kv">${esc(i.memory_type)}</td><td>${esc(i.detail || i.item_id)}</td><td class="kv">${i.tokens}</td></tr>`).join("")}</table></details>` : ""}
    ${c.status === "handoff" ? `<p><a href="/reliability">Open the Reliability page →</a></p>` : ""}
    ${c.blockers.length ? `<h2>Blockers</h2>${list(c.blockers, "bad")}` : ""}
    ${d.missing_evidence.length ? `<h2>Missing evidence</h2>${list(d.missing_evidence, "bad")}` : ""}
    ${c.changes.length ? `<h2>Proposed change</h2><table><tr><th>Field</th><th>Before</th><th>After</th></tr>
      ${c.changes.map(r => `<tr><td><code>${esc(r.field)}</code></td><td>${esc(r.before ?? "unset")}</td><td><b>${esc(r.after)}</b></td></tr>`).join("")}</table>` : ""}
    ${d.tradeoffs.length ? `<h2>Tradeoffs</h2>${list(d.tradeoffs)}` : ""}
    ${d.alternatives.filter(a => Object.keys(a.config || {}).length).length ? `<h2>Alternatives</h2>${list(d.alternatives.filter(a => Object.keys(a.config || {}).length).map(a => `${JSON.stringify(a.config)} when ${a.when}`))}` : ""}
    ${Object.keys(d.conditions || {}).length ? `<h2>Recovery conditions</h2><pre class="file">${esc(JSON.stringify(d.conditions, null, 2))}</pre>` : ""}
    ${c.diff ? `<h2>Diff against base</h2><pre class="file diff">${diffHtml(c.diff)}</pre>` : ""}
    ${c.changes.length ? `<h2>Files</h2><div class="tabs">${files.map(f => `<button type="button" class="tab" data-f="${esc(f)}">${esc(f)}</button>`).join("")}</div><pre class="file" id="file"></pre>
    <h2>Validation (${ok}/${c.validation.length})</h2><table>${c.validation.map(v => `<tr><td>${v.ok ? "✓" : "✗"}</td><td><code>${esc(v.file)}</code></td><td>${esc(v.detail)}</td></tr>`).join("")}</table>` : ""}
    ${c.changes.length && c.status === "offline_validated" ? `<div class="section"><h2>3. Verify after you apply it</h2>
      <div class="muted">Import telemetry observed after applying this bundle. Each check is recorded; earlier results are never overwritten.</div>
      <label for="vfixture">Fixture</label><select id="vfixture"><option value="">Paste my own export</option>${Object.keys(FIXTURES.verify).map(n => `<option>${esc(n)}</option>`).join("")}</select>
      <textarea id="vpayload" rows="6" style="font:11px ui-monospace,monospace;margin-top:6px"></textarea>
      <button type="button" class="btn secondary" id="verify">Import and verify</button><div class="error" id="verror"></div>
      <div id="verifications" style="margin-top:10px">${(c.verifications || []).map(verificationHtml).join("") || `<div class="muted">No verification yet: status stays proposed, not resolved.</div>`}</div></div>` : ""}`;
  if (c.changes.length) {
    const show = f => { $("file").textContent = c.files[f]; document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.f === f)); };
    document.querySelectorAll(".tab").forEach(t => t.onclick = () => show(t.dataset.f));
    show(files.includes("CHANGE.md") ? "CHANGE.md" : files[0]);
  }
  if ($("verify")) {
    $("vfixture").onchange = e => $("vpayload").value = e.target.value ? JSON.stringify(FIXTURES.verify[e.target.value], null, 2) : "";
    $("verify").onclick = async () => {
      $("verror").textContent = "";
      try {
        const ev = await api("/ops/evidence", {base_id: c.change_id, payload: $("vpayload").value});
        render(await api(`/ops/changes/${c.change_id}/verify`, {evidence_id: ev.evidence_id}));
      } catch (err) { $("verror").textContent = err.message; }
    };
  }
  history.replaceState(null, "", `?change=${encodeURIComponent(c.change_id)}`);
}

// ------------------------------------------------------------ form

$("fixture").onchange = e => $("payload").value = e.target.value ? JSON.stringify(FIXTURES.diagnose[e.target.value], null, 2) : "";

$("form").onsubmit = async e => {
  e.preventDefault();
  $("error").textContent = "";
  const btn = e.target.querySelector("[type=submit]"), label = btn.textContent;
  if (!$("base").value) { $("error").textContent = "Build a plan on the Plan page first."; return; }
  if (!$("payload").value.trim()) { $("error").textContent = "Choose a fixture or paste an export."; return; }
  btn.disabled = true; btn.textContent = `${OPS.name} is analyzing…`;
  try {
    const ev = await api("/ops/evidence", {base_id: $("base").value, payload: $("payload").value});
    const change = await api(`/ops/${OPS.role}/analyze`, {base_id: $("base").value, evidence_id: ev.evidence_id,
      slo_p95_ms: +$("slo").value || null, max_error_rate: $("err").value === "" ? null : +$("err").value});
    render(change);
    await loadBases();
  } catch (err) { $("error").textContent = err.message; }
  finally { btn.disabled = false; btn.textContent = label; }
};

(async () => {
  FIXTURES = await api(`/ops/fixtures?role=${OPS.role}`);
  Object.keys(FIXTURES.diagnose).forEach(n => $("fixture").add(new Option(n, n)));
  await loadBases();
  const id = new URLSearchParams(location.search).get("change");
  if (id) try { render(await api(`/ops/changes/${encodeURIComponent(id)}`)); } catch (_) { /* stale link */ }
})().catch(err => $("error").textContent = err.message);
