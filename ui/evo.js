// Long-horizon evolution campaign on the Performance page. Uses $, esc, api, pill from ops.js.
const STAGES = ["OBSERVE", "BUILD_CONTEXT", "PROPOSE", "VALIDATE", "PREPARE_REPLAY", "RUN_TRIALS", "EVALUATE",
  "PROMOTE", "REJECT", "VERIFY_LIVE", "LEARN", "CHECKPOINT"];
const DECISION = {promoted: ["ready", "Promoted"], terminal: ["gray", "Done"], rejected: ["blocked", "Rejected"],
  reject: ["blocked", "Rejected"], rolled_back: ["blocked", "Rolled back"], promote: ["ready", "Promoted"]};
let evoTimer = null, evoId = null;

function outcome(e) {
  if (e.outcome_won) return ["ready", "Promoted · revision " + (e.revision ?? "")];
  if (e.status === "terminal") return e.holdout && !e.holdout.passed ? ["blocked", "Rolled back"] : ["blocked", "Rejected by gates"];
  return DECISION[e.status] || ["gray", e.status.replace(/_/g, " ")];
}

function trialLine(t) {
  const p95 = side => (t[side] || []).map(x => `${x.p95_ms}`).join(" / ");
  return `incumbent p95 ${p95("incumbent")} ms · candidate p95 ${p95("candidate")} ms`;
}

function renderEvo(s) {
  const c = s.campaign, cp = c.checkpoint || {}, tl = c.timeline || [];
  const idx = cp.phase_index ?? 0;
  const byPhase = i => s.windows.filter(w => w.phase_index === i);
  $("evoTimeline").innerHTML = `
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px">
      ${pill(c.stage === "COMPLETE" ? ["ready", "Complete"] : c.running ? ["sim", "Running · " + c.stage] : ["warn", "Paused · " + c.stage])}
      <span class="kv">${esc(c.campaign_id)} · checkpoint revision ${c.revision} · ${c.experiments_spent}/${c.max_experiments} experiments · SLO ${c.policy.slo_p95_ms} ms</span>
      ${!c.running && c.stage !== "COMPLETE" ? `<button type="button" class="btn" id="evoResume">Resume from checkpoint</button>` : ""}
    </div>
    ${c.last_error ? `<div class="error">Last error: ${esc(c.last_error)}</div>` : ""}
    <div class="stages">${STAGES.map(x => `<span class="${x === c.stage ? "on" : ""}">${x}</span>`).join("")}</div>
    <div class="phases">${tl.map((p, i) => {
      const w = byPhase(i), last = w[w.length - 1];
      return `<div class="phase ${i === idx && c.stage !== "COMPLETE" ? "now" : i < idx ? "done" : ""}">
        <h3>${i + 1}. ${esc(p.name)}</h3><span class="muted">${esc(p.story)}</span>
        <span class="kv">${p.rps} rps · prompts ${p.short_tokens.join("–")}${p.long_share ? ` / ${Math.round(p.long_share * 100)}% ${p.long_tokens.join("–")}` : ""} · reuse ${Math.round(p.prefix_reuse * 100)}%</span>
        ${w.map(x => `<span class="kv">obs ${x.attempt + 1}: p95 <b>${x.metrics.p95_ms}</b> ms on ${x.metrics.gpus} GPUs → ${esc(x.trigger || "within SLO")}</span>`).join("")}
        ${last ? "" : `<span class="kv">not reached</span>`}</div>`;
    }).join("")}</div>
    <h2>Architecture revisions</h2>
    <div class="revs">${s.revisions.map(r => `<div class="rev ${esc(r.status || "")}"><b>Rev ${r.revision}${r.phase ? " · " + esc(r.phase) : " · provisioned"}</b>${esc(r.summary)}
      ${r.status === "rolled_back" ? " · rolled back" : ""} · <a href="/api/production/ops/campaigns/${esc(c.campaign_id)}/revisions/${r.revision}/download">bundle</a></div>`).join(" → ")}</div>`;
  const exps = [...s.experiments].reverse();
  $("evoBody").innerHTML = `<h2>Experiments (newest first)</h2>` + (exps.map(e => `
    <div class="exp"><header><b>Phase ${e.phase_index + 1} · ${esc(tl[e.phase_index]?.name || "")} · ${esc(e.move)}${e.recalled ? " · recalled from memory" : ""}</b>
      <span>${pill(outcome(e))} ${e.migration ? pill(["warn", "Stack migration · operator review"]) : ""} ${pill(["gray", (e.architect_run || {}).source || "rules"])}</span></header>
      <p style="margin:6px 0">${esc(e.hypothesis)}</p>
      <div class="kv">${esc(e.incumbent_summary)} → <b>${esc(e.candidate_summary)}</b></div>
      ${e.context ? `<div class="kv">memory: ${e.context.included.length} items (${[...new Set(e.context.included.map(i => i.memory_type))].join(", ")}) · ${e.context.tokens}/${e.context.budget} tokens · ${e.context.excluded} excluded</div>` : ""}
      ${e.trials ? `<div class="kv">${trialLine(e.trials)}</div>` : ""}
      ${e.evaluation ? `<table style="margin-top:6px">${e.evaluation.gates.map(g => `<tr><td>${g.passed ? "✓" : "✗"}</td><td><code>${esc(g.name)}</code></td><td>${esc(g.detail)}</td></tr>`).join("")}</table>` : ""}
      ${(e.validation_reasons || []).length ? `<ul class="issues bad">${e.validation_reasons.map(r => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
      ${e.holdout ? `<div class="kv">hold-out replay: p95 ${e.holdout.p95_ms} ms · ${e.holdout.passed ? "passed" : "failed → rolled back"}</div>` : ""}
      ${e.lesson ? `<div class="kv">lesson${e.lesson.supersedes && e.lesson.supersedes.length ? " (supersedes an older one)" : ""}: ${esc(e.lesson.claim.slice(0, 260))}</div>` : ""}
      <details><summary class="kv">Menu the architect chose from</summary><ul class="issues">${(e.options || []).map(o => `<li><b>${esc(o.move)}</b> ${esc(o.summary)}: ${esc(o.why)}</li>`).join("")}</ul></details>
    </div>`).join("") || `<div class="muted">No experiments yet.</div>`) +
    `<h2>Event log</h2><table>${s.events.slice(0, 30).map(ev => `<tr><td class="kv">${esc(ev.kind)}</td><td>${esc(ev.msg)}</td></tr>`).join("")}</table>`;
  if ($("evoResume")) $("evoResume").onclick = async () => renderEvo(await api(`/ops/campaigns/${c.campaign_id}/resume`, {}));
  clearTimeout(evoTimer);
  if (c.running || (c.stage !== "COMPLETE" && !c.last_error)) evoTimer = setTimeout(() => loadEvo(c.campaign_id), 2500);
}

async function loadEvo(id) {
  evoId = id;
  try { renderEvo(await api(`/ops/campaigns/${encodeURIComponent(id)}`)); } catch (err) { $("evoError").textContent = err.message; }
}

async function refreshEvoBase() {
  const base = $("base").value;
  if (!base) return;
  try {
    const [pre, past] = await Promise.all([api(`/ops/campaigns/preview?base_id=${encodeURIComponent(base)}`),
                                           api(`/ops/campaigns`)]);
    $("evoSlo").value = pre.slo_p95_ms;
    $("evoPast").innerHTML = `<option value="">—</option>` + past.map(c => `<option value="${esc(c.campaign_id)}">${esc(c.campaign_id)} · ${esc(c.stage)} · ${esc((c.checkpoint || {}).summary || "")}</option>`).join("");
  } catch (err) { $("evoError").textContent = err.message; }
}

$("evoPast").onchange = e => e.target.value && loadEvo(e.target.value);
$("evoStart").onclick = async () => {
  $("evoError").textContent = "";
  if (!$("base").value) { $("evoError").textContent = "Build a plan on the Plan page first."; return; }
  try {
    renderEvo(await api("/ops/campaigns", {base_id: $("base").value, slo_p95_ms: +$("evoSlo").value || null,
                                           budget: +$("evoBudget").value || 12}));
  } catch (err) { $("evoError").textContent = err.message; }
};
setTimeout(() => { refreshEvoBase(); $("base").addEventListener("change", refreshEvoBase); }, 800);
