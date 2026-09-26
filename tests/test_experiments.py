import mongomock

from agent.experiments import ExperimentRunner, make_replay_plan
from common.contracts import (
    Arch, Campaign, ExperimentProposal, ObservationSnapshot, Pool, ReplayPlan,
    TrafficProfile, TrialRole,
)


def test_replay_generation_is_reproducible_and_hashed():
    profile = TrafficProfile(rps=4, prompt_tokens=[30, 400, 1200], output_tokens=12)
    first = make_replay_plan(profile, 10, seed=71, campaign_id="c", experiment_id="e")
    second = make_replay_plan(profile, 10, seed=71, campaign_id="c", experiment_id="e")
    assert first.content_hash == second.content_hash
    assert [event.model_dump() for event in first.events] == [event.model_dump() for event in second.events]
    assert first.replay_id == second.replay_id


def test_runner_uses_identical_events_serial_order_and_reuses_trials():
    database = mongomock.MongoClient().db
    trace = []

    def executor(architecture, events, slot):
        trace.append((architecture.key(), events))
        return [{"event_index": i, "latency_ms": float(100 + i), "error": None} for i in range(len(events))]

    runner = ExperimentRunner(database, executor=executor, seed=1)
    incumbent = Arch(version=3, status="live", pools={"shared": Pool(gpu="t4", replicas=2)})
    candidate = Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)})
    replay = make_replay_plan(TrafficProfile(rps=4, prompt_tokens=[50], output_tokens=4), 2,
                              seed=8, experiment_id="exp")
    inc, cand = runner.run_repeated_trials("exp", incumbent, candidate, replay, 3, warmup_requests=1)
    assert [trial.repeat for trial in inc] == [0, 1, 2]
    assert [trial.execution_order for trial in inc] == [0, 1, 0]
    assert [trial.execution_order for trial in cand] == [1, 0, 1]
    assert all(trial.n == len(replay.events) - 1 and trial.warmup_excluded == 1 for trial in inc + cand)
    assert all(trial.outcomes[0].latency_ms == 100 for trial in inc + cand)
    assert len(trace) == 6
    assert all(args == trace[0][1] for _, args in trace)
    # Trial identifiers make a restart/retry idempotent.
    runner.run_trial_sync("exp", candidate, replay, 0, 1, role=TrialRole.CANDIDATE, warmup_requests=1)
    assert len(trace) == 6
    assert database.trials.count_documents({}) == 6


def test_prepare_replay_persists_one_plan():
    database = mongomock.MongoClient().db
    runner = ExperimentRunner(database, seed=9)
    campaign = Campaign(campaign_id="c")
    proposal = ExperimentProposal(proposal_id="e", campaign_id="c", hypothesis="test",
                                 candidate=Arch(status="candidate", pools={"shared": Pool(gpu="t4", replicas=1)}))
    observation = ObservationSnapshot(architecture=Arch(status="live", pools={"shared": Pool(gpu="t4", replicas=1)}),
                                      regime_vector=[0.1, 0.2, 0.3, 0.0], metrics={"rps": 2})
    first = runner.prepare_replay_sync(campaign, proposal, observation)
    second = runner.prepare_replay_sync(campaign, proposal, observation)
    assert first.content_hash == second.content_hash
    assert database.replay_plans.count_documents({}) == 1

