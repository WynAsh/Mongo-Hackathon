"""DEV C. The architect agent: observe -> recall -> propose -> shadow-test -> promote -> learn.

python -m agent.loop
"""
from __future__ import annotations

import time

from agent import proposer
from common import config, db
from common.contracts import Arch, Experiment, describe_regime, score
from gateway import metrics
from infra import shadow
from memory import store

TRUST_CONFIRMATIONS = 2   # a lesson confirmed this many times is applied fast (single shadow check)


def better(a, b) -> bool:
    """Is result a meaningfully better than result b?"""
    sa, sb = score(a, config.SLO_P95_MS), score(b, config.SLO_P95_MS)
    if sa[0] != sb[0]:
        return sa[0] < sb[0]
    if sa[0] == 0:                       # both meet SLO: cheaper wins
        return a.usd_hr < b.usd_hr
    return a.p95_ms < b.p95_ms * (1 - config.PROMOTE_MARGIN)


def trigger_for(arch: Arch, obs: dict) -> str | None:
    if obs["n"] < 15 or obs["p95_ms"] is None or not obs.get("stable"):
        return None   # too little data, or traffic is mid-change: don't learn from a mixed window
    if obs["p95_ms"] > config.SLO_P95_MS or obs["errors"] > obs["n"] * 0.1:
        return "slo_breach"
    if obs["p95_ms"] < config.SLO_P95_MS * 0.5 and arch.usd_hr() > 0.5:
        return "overprovisioned"
    return None


def cycle(state: dict):
    doc = db.live_arch_doc()
    if not doc:
        return
    arch = Arch(**{k: v for k, v in doc.items() if k != "_id"})
    obs = metrics.window(config.WINDOW_S, arch.version)
    trig = trigger_for(arch, obs)
    if not trig:
        return
    vec = obs["regime_vector"]
    key = (trig, tuple(store._bucket(vec)), arch.key())
    if state.get("last_noop") == key:        # already concluded nothing beats the incumbent here
        return
    db.log_event("observe", f"{'SLO breach' if trig == 'slo_breach' else 'Over-provisioned'}: "
                 f"p95 {obs['p95_ms']/1000:.1f}s (SLO {config.SLO_P95_MS/1000:.0f}s) at "
                 f"{describe_regime(vec)}, running {arch.summary()}", trigger=trig, obs={k: obs[k] for k in ("rps", "p50_ms", "p95_ms", "n")})

    mem = store.recall(vec, exclude_key=arch.key())
    candidates: list[Arch] = []
    trusted = [w for w in mem["winners"] if w["wins"] >= TRUST_CONFIRMATIONS and w["wins"] > w["losses"]]
    if mem["winners"]:
        top = mem["winners"][0]
        db.log_event("recall", f"Memory: {len(mem['winners'])} past winner(s) for similar traffic. "
                     f"Bandit picked '{store.arch_from_memory(top).summary()}' "
                     f"(won {top['wins']}, lost {top['losses']}).", lessons=mem["lessons"])
        candidates.append(store.arch_from_memory(top))
    else:
        db.log_event("recall", "Memory: nothing learned yet for this kind of traffic.")

    if not trusted:  # only pay for LLM exploration when memory isn't confident
        losers = [e["arch"] and Arch(**{k: v for k, v in e["arch"].items() if k != "_id"}).summary()
                  for e in db.db().experiments.find({"won": False}).sort("ts", -1).limit(20)
                  if sum((a - b) ** 2 for a, b in zip(e["regime_vector"], vec)) < 0.1]
        diag, props, rejected = proposer.propose(arch, obs, trig, mem, sorted(set(losers)))
        db.log_event("propose", f"Diagnosis: {diag}", candidates=[
            {"summary": p.summary(), "reason": p.reason} for p in props], rejected=rejected)
        for p in props:
            if p.key() not in {c.key() for c in candidates}:
                candidates.append(p)
    candidates = candidates[:3]
    if not candidates:
        state["last_noop"] = key
        return

    profile = metrics.profile_from_window(obs)
    db.log_event("test", f"Shadow-testing {len(candidates)} candidate(s) + incumbent on replayed traffic "
                 f"({profile.rps:.1f} rps, {config.SHADOW_S}s)...",
                 candidates=[c.summary() for c in candidates])
    results = shadow.run_many([arch] + candidates, profile)
    inc_res, cand_res = results[0], results[1:]
    lines = [f"incumbent: p95 {inc_res.p95_ms/1000:.1f}s ${inc_res.usd_hr}/hr"]
    exps = []
    for c, r in zip(candidates, cand_res):
        won = better(r, inc_res)
        lines.append(f"{c.summary()}: p95 {r.p95_ms/1000:.1f}s {'WIN' if won else 'lose'}")
        exps.append(Experiment(regime_vector=vec, arch=c, result=r, incumbent_key=arch.key(),
                               incumbent_p95_ms=inc_res.p95_ms, won=won))
    db.log_event("results", " | ".join(lines), results=[r.model_dump() for r in results])

    winners = sorted([(c, r) for c, r in zip(candidates, cand_res) if better(r, inc_res)],
                     key=lambda cr: score(cr[1], config.SLO_P95_MS))
    for e in exps:   # only the chosen winner counts as a win in memory
        e.won = bool(winners) and e.arch.key() == winners[0][0].key()
    store.record(exps)

    if not winners:
        db.log_event("hold", "Nothing beat the incumbent. Keeping the current setup.")
        state["last_noop"] = key
        return
    best, res = winners[0]
    best.reason = (f"{best.reason} | shadow p95 {inc_res.p95_ms/1000:.1f}s -> {res.p95_ms/1000:.1f}s, "
                   f"${inc_res.usd_hr} -> ${res.usd_hr}/hr")
    v = db.promote(best.model_dump())
    db.log_event("promote", f"Promoted v{v}: {best.summary()}. {best.reason}", version=v)
    state.pop("last_noop", None)
    time.sleep(config.WINDOW_S)   # let a fresh window build up on the new version


def main():
    db.log_event("start", f"Architect agent online. SLO p95 <= {config.SLO_P95_MS/1000:.0f}s. "
                 f"LLM: {config.AGENT_MODEL if config.OPENROUTER_API_KEY else 'heuristic (no key)'}")
    state: dict = {}
    while True:
        try:
            cycle(state)
        except Exception as e:  # noqa: BLE001
            db.log_event("error", f"agent cycle failed: {e}")
        time.sleep(config.CYCLE_S)


if __name__ == "__main__":
    main()
