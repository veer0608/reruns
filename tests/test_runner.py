from __future__ import annotations

import json

from evals.checkpoint import Checkpoint
from evals.runner import Summary, main, report, run_solver, run_trial
from evals.runner import load_verdicts as runner_load, regrade as runner_regrade
from reruns.grade import Verdict


def verdict(task_id, trial, passed):
    return Verdict(task_id=task_id, trial=trial, state_ok=passed, calls_ok=passed)


def summary(verdicts, k=5, tasks=2, complete=True):
    return Summary(solver="model", model="fake", k=k, tasks=tasks,
                   verdicts=verdicts, complete=complete)


def test_pass_hat_k_is_not_pass_at_1():
    """The number the project exists for. Ten trials, six passes, but only one
    of the two tasks passed all five -- 0.6 against 0.5, and the gap widens
    fast as tasks get flakier."""
    verdicts = [verdict("a", i, True) for i in range(1, 6)]
    verdicts += [verdict("b", i, i == 1) for i in range(1, 6)]
    got = summary(verdicts)
    assert got.pass_at_1 == 0.6
    assert got.pass_hat_k == 0.5


def test_one_failure_in_five_costs_the_whole_task():
    verdicts = [verdict("a", i, i != 3) for i in range(1, 6)]
    got = summary(verdicts, tasks=1)
    assert got.pass_at_1 == 0.8
    assert got.pass_hat_k == 0.0


def test_a_task_with_fewer_than_k_trials_is_not_counted():
    """A half-measured task is not a task. Counting it would let an early stop
    inflate pass^k by dropping the trials that would have failed."""
    verdicts = [verdict("a", i, True) for i in range(1, 6)]
    verdicts += [verdict("b", 1, True)]
    got = summary(verdicts)
    assert got.pass_hat_k == 1.0
    assert round(got.pass_at_1, 4) == round(6 / 6, 4)


def test_violations_are_counted_by_rule():
    from reruns.policy import Violation

    bad = verdict("a", 1, False)
    bad.violations = [Violation(7, "amount_stated_first", "x"),
                      Violation(7, "amount_stated_first", "y")]
    got = summary([bad, verdict("a", 2, True)], k=2, tasks=1)
    assert got.violation_rate == 0.5
    assert got.by_rule["7 amount_stated_first"] == 2


def test_the_oracle_scores_one_and_the_mute_solver_zero(domain):
    empty = Checkpoint.load(None)
    for solver, wanted in (("oracle", 1.0), ("mute", 0.0)):
        got = run_solver(domain, solver, k=1, tasks=domain.tasks, client=None,
                         scripted=True, checkpoint=empty, max_turns=12, quiet=True)
        assert got.pass_at_1 == wanted, solver


def test_every_task_is_failed_by_the_mute_solver_for_a_stated_reason(domain):
    """A task the mute solver fails for no recorded reason is a task whose
    expectations are empty -- it would pass anything."""
    empty = Checkpoint.load(None)
    got = run_solver(domain, "mute", k=1, tasks=domain.tasks, client=None,
                     scripted=True, checkpoint=empty, max_turns=12, quiet=True)
    for one in got.verdicts:
        assert one.reasons, one.task_id


def test_a_partial_run_prints_no_percentage():
    text = report(summary([verdict("a", 1, True)], complete=False))
    assert "PARTIAL" in text
    assert "%" not in text
    assert "pass@1" not in text


def test_the_daily_cap_stops_the_run_where_it_stands(domain, monkeypatch):
    import evals.runner as runner

    seen = []

    def capped(domain_, task, trial, solver, **kwargs):
        seen.append((task.id, trial))
        one = verdict(task.id, trial, True)
        if len(seen) == 3:
            one.error = "quota: 429 tokens per day"
        return one

    monkeypatch.setattr(runner, "run_trial", capped)
    got = run_solver(domain, "model", k=2, tasks=domain.tasks[:4], client=None,
                     scripted=True, checkpoint=Checkpoint.load(None),
                     max_turns=12, quiet=True)
    assert not got.complete
    assert len(seen) == 3


def test_a_checkpoint_is_resumed_rather_than_re_paid_for(domain, tmp_path, monkeypatch):
    import evals.runner as runner

    path = tmp_path / "run.json"
    calls = []

    def counted(domain_, task, trial, solver, **kwargs):
        calls.append((task.id, trial))
        return verdict(task.id, trial, True)

    monkeypatch.setattr(runner, "run_trial", counted)
    tasks = domain.tasks[:3]
    first = run_solver(domain, "model", k=2, tasks=tasks, client=None, scripted=True,
                       checkpoint=Checkpoint.load(path), max_turns=12, quiet=True)
    assert len(calls) == 6 and first.pass_at_1 == 1.0

    calls.clear()
    resumed = run_solver(domain, "model", k=2, tasks=tasks, client=None, scripted=True,
                         checkpoint=Checkpoint.load(path), max_turns=12, quiet=True)
    assert calls == []
    assert len(resumed.verdicts) == 6


def test_a_checkpoint_written_mid_run_is_readable(domain, tmp_path):
    path = tmp_path / "run.json"
    checkpoint = Checkpoint.load(path)
    checkpoint.add(verdict("a", 1, True))
    checkpoint.add(verdict("a", 2, False))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw["verdicts"]) == 2
    assert Checkpoint.load(path).has("a", 2)
    assert not list(path.parent.glob("*.tmp"))


def test_each_trial_gets_a_world_of_its_own(domain):
    """Run the same mutating task twice and the second must not inherit the
    first's refund -- the refund id would move from rf_2 to rf_3 and the task
    would fail for a reason that has nothing to do with the agent."""
    task = domain.task("refund_kettle")
    first = run_trial(domain, task, 1, "oracle")
    second = run_trial(domain, task, 2, "oracle")
    assert first.passed and second.passed


def test_check_passes_on_the_shipped_domain(capsys):
    assert main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "oracle   pass@1 1.000" in out
    assert "mute     pass@1 0.000" in out


def test_the_runner_says_so_when_there_is_no_key(monkeypatch, capsys):
    import evals.runner as runner

    monkeypatch.setattr(runner, "build_client", lambda *a, **k: None)
    assert main(["--solvers", "model", "--k", "1"]) == 2
    assert "no API key" in capsys.readouterr().out


def test_a_run_file_is_written_when_asked(domain, tmp_path, capsys):
    out = tmp_path / "oracle.json"
    assert main(["--solvers", "oracle", "--quiet", "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["summaries"][0]["pass_at_1"] == 1.0
    assert len(written["summaries"][0]["verdicts"]) == len(domain.tasks)


def test_out_may_not_overwrite_the_checkpoint(tmp_path, capsys):
    """They are different file formats. Pointing both at one path destroyed a
    60 trial checkpoint on the first real run of this project."""
    same = tmp_path / "run.json"
    assert main(["--solvers", "oracle", "--checkpoint", str(same), "--out", str(same)]) == 2
    assert "must be different files" in capsys.readouterr().out


def test_a_dead_trial_is_not_cached(domain, tmp_path, monkeypatch):
    """A quota-killed trial cached as a failure hands tomorrow's resume a
    result that never happened, which is the abandonment rule defeated from
    inside."""
    import evals.runner as runner

    path = tmp_path / "run.json"

    def dies(domain_, task, trial, solver, **kwargs):
        one = verdict(task.id, trial, False)
        one.error = "quota: 429 tokens per day"
        return one

    monkeypatch.setattr(runner, "run_trial", dies)
    got = run_solver(domain, "model", k=1, tasks=domain.tasks[:1], client=None,
                     scripted=True, checkpoint=Checkpoint.load(path),
                     max_turns=12, quiet=True)
    assert not got.complete
    assert got.verdicts == []
    assert not Checkpoint.load(path).has(domain.tasks[0].id, 1)


def test_a_dead_trial_in_an_old_checkpoint_is_ignored(domain, tmp_path):
    path = tmp_path / "run.json"
    checkpoint = Checkpoint.load(path)
    clean = verdict("a", 1, True)
    dead = verdict("a", 2, False)
    dead.error = "quota: 429"
    checkpoint.verdicts[("a", 1)] = clean
    checkpoint.verdicts[("a", 2)] = dead
    checkpoint.save()

    reloaded = Checkpoint.load(path)
    assert reloaded.has("a", 1)
    assert not reloaded.has("a", 2), "a trial that never ran must be re-run"


def test_a_dropped_connection_is_retried_before_it_counts(domain, monkeypatch):
    import evals.runner as runner

    attempts = []

    def flaky(domain_, task, trial, solver, **kwargs):
        attempts.append(trial)
        one = verdict(task.id, trial, len(attempts) > 1)
        if len(attempts) == 1:
            one.error = "llm: RemoteDisconnected"
        return one

    monkeypatch.setattr(runner, "run_trial", flaky)
    got = run_solver(domain, "model", k=1, tasks=domain.tasks[:1], client=None,
                     scripted=True, checkpoint=Checkpoint.load(None),
                     max_turns=12, quiet=True)
    assert len(attempts) == 2
    assert got.complete and got.pass_at_1 == 1.0


def test_regrading_a_transcript_reproduces_its_verdict(domain):
    """The guarantee that makes --regrade trustworthy. Replay must land on the
    same answer, or re-scoring from disk is inventing results."""
    for task in domain.tasks:
        original = run_trial(domain, task, 1, "oracle")
        again = runner_regrade(domain, original)
        assert again.as_dict() == original.as_dict(), task.id


def test_regrading_picks_up_a_policy_fix_without_a_model(domain):
    """A trial graded under a broken rule is re-scored from its own transcript."""
    task = domain.task("refund_original_payment")
    original = run_trial(domain, task, 1, "oracle")
    assert original.passed

    from reruns.grade import grade as grade_fn
    stale = Verdict.from_dict(original.as_dict())
    from reruns.policy import Violation
    stale.violations = [Violation(9, "store_credit_default", "wrongly flagged")]
    assert not stale.passed

    assert runner_regrade(domain, stale).passed


def test_load_verdicts_reads_a_run_file_as_well_as_a_checkpoint(domain, tmp_path):
    out = tmp_path / "run.json"
    assert main(["--solvers", "oracle", "--quiet", "--out", str(out)]) == 0
    assert len(runner_load(out)) == len(domain.tasks)


def test_dry_run_spends_nothing_and_says_what_a_run_would(domain, tmp_path, capsys):
    """Checking that a resume picks up where the last one stopped should not
    cost a day's allowance to find out."""
    path = tmp_path / "run.json"
    checkpoint = Checkpoint.load(path)
    banked = verdict("refund_kettle", 1, True)
    banked.prompt_tokens = 10_000
    checkpoint.add(banked)

    assert main(["--dry-run", "--k", "2", "--checkpoint", str(path),
                 "--tasks", "refund_kettle,status_question"]) == 0
    out = capsys.readouterr().out
    assert "1 trials already banked, 3 to run" in out
    assert "refund_kettle" in out and "trials 2" in out
    assert "status_question" in out and "trials 1,2" in out


def test_dry_run_with_no_checkpoint_runs_everything(domain, capsys):
    assert main(["--dry-run", "--k", "5"]) == 0
    out = capsys.readouterr().out
    assert f"0 trials already banked, {len(domain.tasks) * 5} to run" in out


def test_regrade_reports_a_violation_change_that_flips_nothing(domain, tmp_path, capsys):
    """A rule that adds a violation to a trial already failing on state flips no
    verdict. Reporting only flips hid exactly that: the strict reading of rule 7
    found the mechanism behind three failures and the summary said nothing
    changed."""
    out = tmp_path / "oracle.json"
    assert main(["--solvers", "oracle", "--quiet", "--out", str(out)]) == 0
    capsys.readouterr()

    saved = runner_load(out)
    stale = [Verdict.from_dict(v.as_dict()) for v in saved]
    from reruns.policy import Violation

    stale[0].violations = [Violation(3, "delivered_items_only", "invented")]
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"verdicts": [v.as_dict() for v in stale]}), encoding="utf-8")

    assert main(["--regrade", str(path), "--k", "1"]) == 0
    printed = capsys.readouterr().out
    assert "-delivered_items_only" in printed
    assert "nothing changed" not in printed


def test_regrade_says_so_when_truly_nothing_moved(domain, tmp_path, capsys):
    out = tmp_path / "oracle.json"
    assert main(["--solvers", "oracle", "--quiet", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["--regrade", str(out), "--k", "1"]) == 0
    assert "nothing changed" in capsys.readouterr().out


def test_strict_rule_7_is_a_regrade_reading_not_a_default(domain, tmp_path, capsys):
    """The flag must not change what a live run records, or a measurement half
    graded each way becomes two measurements."""
    out = tmp_path / "oracle.json"
    assert main(["--solvers", "oracle", "--quiet", "--out", str(out)]) == 0
    plain = [v.as_dict() for v in runner_load(out)]

    from reruns.dataset import Domain as D
    strict = [runner_regrade(D.load(), Verdict.from_dict(v), True).as_dict() for v in plain]
    assert [v["violations"] for v in plain] == [v["violations"] for v in strict], \
        "the oracle announces and waits, so neither reading should fault it"


def test_a_checkpoint_from_another_harness_is_refused(tmp_path, capsys):
    """The guard that replaces remembering.

    The closing turn, the escalation rule and the customer's stop condition
    each changed what the agent experiences, and each time the only thing
    between a corrupted measurement and a clean one was somebody noticing.
    """
    from reruns.agent import HARNESS_VERSION

    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "meta": {"harness": HARNESS_VERSION - 1},
        "verdicts": [verdict("a", 1, True).as_dict()],
    }), encoding="utf-8")

    assert main(["--dry-run", "--k", "1", "--checkpoint", str(path)]) == 2
    printed = capsys.readouterr().out
    assert "two measurements" in printed
    assert f"harness {HARNESS_VERSION - 1}" in printed


def test_a_checkpoint_with_no_marker_is_treated_as_older(tmp_path, capsys):
    """Unknown must not mean compatible. A file without the marker predates it,
    and reading unknown as safe is the one reading that lets trials from two
    harnesses into one number."""
    path = tmp_path / "unmarked.json"
    path.write_text(json.dumps({
        "meta": {"k": 5},
        "verdicts": [verdict("a", 1, True).as_dict()],
    }), encoding="utf-8")

    assert main(["--dry-run", "--k", "1", "--checkpoint", str(path)]) == 2
    assert "older than" in capsys.readouterr().out


def test_a_checkpoint_this_harness_wrote_is_accepted(domain, tmp_path, capsys):
    path = tmp_path / "current.json"
    Checkpoint.load(path).add(verdict("refund_kettle", 1, True))
    assert main(["--dry-run", "--k", "1", "--checkpoint", str(path)]) == 0
    assert "1 trials already banked" in capsys.readouterr().out
