"""The eval. Runs every task k times and reports what survives repetition.

    python -m evals.runner --solvers oracle,mute          # free, no key
    python -m evals.runner --check                        # what CI runs
    python -m evals.runner --solvers model --k 5 --checkpoint runs/x.json

Two numbers come out of it.

    pass@1    the share of all task-trials that passed. What everyone reports.
    pass^k    the share of TASKS that passed on all k trials.

They answer different questions. pass@1 asks whether the agent can do this;
pass^k asks whether it can be relied on to. The second is the one a person
deciding whether to put an agent in front of customers actually needs, and it
is almost always the lower of the two -- an agent at pass@1 0.6 with
independent trials would sit near 0.08 at pass^5.

**A run that does not finish gets no score.** If the daily token allowance runs
out mid-run, the trials that never happened are indistinguishable from trials
that failed, and a percentage computed over what completed is a wrong number
wearing the costume of a right one. The run stops, the checkpoint is kept, and
the summary refuses to print. Resume it tomorrow with the same --checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from reruns import agent
from reruns.dataset import DEFAULT_DOMAIN, Domain, Task, validate
from reruns.grade import Verdict, grade
from reruns.llm import build_client
from reruns.tools import Toolbox, Trace

from .checkpoint import Checkpoint

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Summary:
    solver: str
    model: str | None
    k: int
    tasks: int
    verdicts: list[Verdict]
    complete: bool

    @property
    def pass_at_1(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.passed for v in self.verdicts) / len(self.verdicts)

    @property
    def pass_hat_k(self) -> float:
        by_task: dict[str, list[Verdict]] = {}
        for verdict in self.verdicts:
            by_task.setdefault(verdict.task_id, []).append(verdict)
        if not by_task:
            return 0.0
        whole = [
            trials for trials in by_task.values() if len(trials) == self.k
        ]
        if not whole:
            return 0.0
        return sum(all(v.passed for v in trials) for trials in whole) / len(whole)

    @property
    def state_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.state_ok and v.calls_ok for v in self.verdicts) / len(self.verdicts)

    @property
    def violation_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(not v.policy_ok for v in self.verdicts) / len(self.verdicts)

    @property
    def by_rule(self) -> Counter:
        counts: Counter = Counter()
        for verdict in self.verdicts:
            for violation in verdict.violations:
                counts[f"{violation.rule} {violation.name}"] += 1
        return counts

    def as_dict(self) -> dict:
        return {
            "solver": self.solver,
            "model": self.model,
            "k": self.k,
            "tasks": self.tasks,
            "complete": self.complete,
            "pass_at_1": round(self.pass_at_1, 4),
            "pass_hat_k": round(self.pass_hat_k, 4),
            "state_and_calls_rate": round(self.state_rate, 4),
            "violation_rate": round(self.violation_rate, 4),
            "violations_by_rule": dict(self.by_rule),
            "prompt_tokens": sum(v.prompt_tokens for v in self.verdicts),
            "completion_tokens": sum(v.completion_tokens for v in self.verdicts),
            "cost_usd": _total_cost(self.verdicts),
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


def _total_cost(verdicts: list[Verdict]) -> float | None:
    priced = [v.cost_usd for v in verdicts if v.cost_usd is not None]
    return round(sum(priced), 6) if priced else None


def run_trial(
    domain: Domain,
    task: Task,
    trial: int,
    solver: str,
    client=None,
    scripted: bool = True,
    max_turns: int = agent.MAX_TURNS,
    final_turn: bool = True,
) -> Verdict:
    """One task, one trial, in a world nobody else has touched."""
    store = domain.store()
    before = store.snapshot()
    trace = Trace()
    toolbox = Toolbox(store, trace)
    user = agent.build_user(task, client, scripted=scripted)
    try:
        episode = agent.run(solver, task, domain.policy, toolbox, user, client,
                            max_turns, final_turn=final_turn)
    except Exception as exc:  # noqa: BLE001 - a crashed trial is a failed trial, not a dead run
        episode = agent.Episode(trace=trace, error=f"{type(exc).__name__}: {exc}")
    after = store.snapshot()
    store.close()

    verdict = grade(task, trial, trace, before, after, domain.seed.now, error=episode.error)
    verdict.prompt_tokens = episode.prompt_tokens
    verdict.completion_tokens = episode.completion_tokens
    verdict.cost_usd = episode.cost_usd
    return verdict


def load_verdicts(path: str | Path) -> list[Verdict]:
    """Verdicts out of either file this project writes.

    A checkpoint keeps them at the top level; a run file nests them under each
    summary. Reading both means a run file is still usable as a resume source,
    which mattered the day `--out` was pointed at the checkpoint and overwrote
    it.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("verdicts")
    if entries is None:
        entries = [v for summary in raw.get("summaries", []) for v in summary.get("verdicts", [])]
    return [Verdict.from_dict(entry) for entry in entries]


def regrade(domain: Domain, verdict: Verdict, strict_rule_7: bool = False) -> Verdict:
    """Score a saved transcript again, without calling a model.

    The store is deterministic, so replaying a transcript's calls into a fresh
    world reproduces exactly the state that trial ended in. That makes a
    grading fix free: when a policy rule turns out to be wrong -- as rule 9
    was, matching a regex against generated prose and scoring three correct
    trials as violations -- every finished trial can be re-scored from disk
    instead of re-bought from the provider.
    """
    task = domain.task(verdict.task_id)
    store = domain.store()
    before = store.snapshot()
    trace = Trace()
    box = Toolbox(store, trace)
    for event in verdict.transcript:
        kind = event.get("kind")
        if kind == "assistant":
            trace.say(event.get("text", ""))
        elif kind == "user":
            trace.hear(event.get("text", ""))
        elif kind == "call":
            box.invoke(event.get("name", ""), event.get("arguments") or {})
    after = store.snapshot()
    store.close()

    fresh = grade(task, verdict.trial, trace, before, after, domain.seed.now,
                  error=verdict.error, strict_rule_7=strict_rule_7)
    fresh.prompt_tokens = verdict.prompt_tokens
    fresh.completion_tokens = verdict.completion_tokens
    fresh.cost_usd = verdict.cost_usd
    return fresh


def run_solver(
    domain: Domain,
    solver: str,
    *,
    k: int,
    tasks: tuple[Task, ...],
    client=None,
    scripted: bool,
    checkpoint: Checkpoint,
    max_turns: int,
    quiet: bool,
    final_turn: bool = True,
) -> Summary:
    verdicts: list[Verdict] = []
    complete = True
    for task in tasks:
        for trial in range(1, k + 1):
            cached = checkpoint.get(task.id, trial) if solver == "model" else None
            if cached is not None:
                verdicts.append(cached)
                continue
            verdict = run_trial(
                domain, task, trial, solver,
                client=client, scripted=scripted, max_turns=max_turns,
                final_turn=final_turn,
            )
            # A trial that died is not a measurement. A dropped connection
            # scored as a failed trial is the same error this project refuses
            # to make at the run level, one layer down -- so retry it once, and
            # if the second attempt dies too, stop rather than record it.
            if verdict.error and not verdict.error.startswith("quota"):
                if not quiet:
                    print(f"  {task.id:<28} trial {trial}/{k}  retrying after {verdict.error[:40]}")
                verdict = run_trial(
                    domain, task, trial, solver,
                    client=client, scripted=scripted, max_turns=max_turns,
                    final_turn=final_turn,
                )
            verdicts.append(verdict)
            # Only a real measurement is cached. Caching a quota-killed trial
            # would hand tomorrow's resume a failure that never happened, which
            # is exactly the contamination the abandonment rule exists to stop.
            if solver == "model" and not verdict.error:
                checkpoint.add(verdict)
            if not quiet:
                mark = "pass" if verdict.passed else "FAIL"
                note = "" if verdict.passed else "  " + "; ".join(
                    verdict.reasons + [v.name for v in verdict.violations]
                )[:110]
                print(f"  {task.id:<28} trial {trial}/{k}  {mark}{note}")
            if verdict.error:
                reason = ("daily allowance exhausted"
                          if verdict.error.startswith("quota")
                          else f"unrecoverable: {verdict.error[:60]}")
                print(f"\n  {reason} during {task.id} trial {trial}.")
                # Dropped, not kept as a failure. It never produced a reading.
                verdicts.pop()
                complete = False
                break
        if not complete:
            break

    return Summary(
        solver=solver,
        model=getattr(client, "model", None) if solver == "model" else None,
        k=k,
        tasks=len(tasks),
        verdicts=verdicts,
        complete=complete,
    )


def report(summary: Summary) -> str:
    if not summary.complete:
        return (
            f"\n{summary.solver}: PARTIAL RUN, no score.\n"
            f"  {len(summary.verdicts)} of {summary.tasks * summary.k} trials completed "
            f"before the daily allowance ran out.\n"
            "  The trials that never ran cannot be told apart from trials that failed,\n"
            "  so there is no honest percentage to print. Resume with the same "
            "--checkpoint.\n"
        )
    lines = [
        f"\n{summary.solver}"
        + (f" ({summary.model})" if summary.model else "")
        + f"  {summary.tasks} tasks x {summary.k} trials",
        f"  pass@1            {summary.pass_at_1:.3f}",
        f"  pass^{summary.k}            {summary.pass_hat_k:.3f}",
        f"  state and calls   {summary.state_rate:.3f}",
        f"  policy violated   {summary.violation_rate:.3f} of trials",
    ]
    if summary.by_rule:
        lines.append("  rules broken:")
        for rule, count in summary.by_rule.most_common():
            lines.append(f"    {count:>3}x  rule {rule}")
    cost = _total_cost(summary.verdicts)
    if cost is not None:
        lines.append(f"  cost              ${cost:.4f}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.runner", description=__doc__)
    parser.add_argument("--domain", default=str(DEFAULT_DOMAIN))
    parser.add_argument("--solvers", default="model",
                        help="comma separated: model, oracle, mute")
    parser.add_argument("--k", type=int, default=5, help="trials per task")
    parser.add_argument("--tasks", default=None, help="comma separated task ids")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--scripted-user", action="store_true",
                        help="replay the fixed script instead of simulating the customer")
    parser.add_argument("--max-turns", type=int, default=agent.MAX_TURNS)
    parser.add_argument("--no-final-turn", action="store_true",
                        help="legacy: end the episode the moment the customer stops, "
                             "even mid-action. Only for extending a run measured "
                             "before the final turn existed.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None, help="write the run JSON here")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the trials a run would execute, and stop")
    parser.add_argument("--regrade", default=None, metavar="PATH",
                        help="re-score saved trials from their transcripts, no model calls")
    parser.add_argument("--strict-rule-7", action="store_true",
                        help="regrade only: read rule 7 as written, so announcing and "
                             "refunding in one message counts as the customer never "
                             "having seen the number before it was final")
    parser.add_argument("--check", action="store_true",
                        help="validate the task file, then assert oracle 1.0 and mute 0.0")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.out and args.checkpoint and Path(args.out) == Path(args.checkpoint):
        # These are different file formats. Writing the run file over the
        # checkpoint at the end of a partial run destroys the resume data and
        # the next run silently re-pays for every finished trial. Ask me how I
        # know.
        print("--out and --checkpoint must be different files: "
              "the run file would overwrite the checkpoint.")
        return 2

    domain = Domain.load(args.domain)
    problems = validate(domain)
    if problems:
        print("task file is not valid:")
        for problem in problems:
            print(f"  {problem}")
        return 2

    if args.check:
        return _check(domain)

    if args.regrade:
        return _regrade(domain, args)

    solvers = [name.strip() for name in args.solvers.split(",") if name.strip()]
    tasks = domain.select(args.tasks)

    if args.dry_run:
        return _dry_run(domain, tasks, args)

    client = None
    if "model" in solvers:
        client = build_client(args.provider, args.model)
        if client is None:
            print("no API key found, so the model solver cannot run. "
                  "Try --solvers oracle,mute.")
            return 2

    checkpoint = Checkpoint.load(args.checkpoint)
    checkpoint.meta = {
        "domain": domain.name,
        "k": args.k,
        "model": getattr(client, "model", None),
        "scripted_user": bool(args.scripted_user),
        "final_turn": not args.no_final_turn,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    summaries = []
    exit_code = 0
    for solver in solvers:
        if not args.quiet:
            print(f"\n{solver}:")
        summary = run_solver(
            domain, solver,
            k=1 if solver in {"oracle", "mute"} else args.k,
            tasks=tasks,
            client=client,
            scripted=args.scripted_user or solver != "model",
            checkpoint=checkpoint,
            max_turns=args.max_turns,
            quiet=args.quiet,
            final_turn=not args.no_final_turn,
        )
        summaries.append(summary)
        print(report(summary))
        if not summary.complete:
            exit_code = 3

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "meta": checkpoint.meta,
                    "summaries": [s.as_dict() for s in summaries],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return exit_code


def _dry_run(domain: Domain, tasks, args) -> int:
    """What a run would cost, before it costs it.

    A resume is only worth starting if it picks up where the last one stopped.
    Finding out that it did not, by watching a day's allowance go on trials
    that were already paid for, is an expensive way to learn it.
    """
    checkpoint = Checkpoint.load(args.checkpoint)
    todo, cached = [], []
    for task in tasks:
        for trial in range(1, args.k + 1):
            (cached if checkpoint.has(task.id, trial) else todo).append((task.id, trial))

    print(f"checkpoint: {args.checkpoint or 'none'}")
    print(f"  {len(cached)} trials already banked, {len(todo)} to run")
    print()
    by_task: dict[str, list[int]] = {}
    for task_id, trial in todo:
        by_task.setdefault(task_id, []).append(trial)
    for task_id, trials in by_task.items():
        runs = ",".join(str(t) for t in trials)
        print(f"  {task_id:<30} trials {runs}")
    if not todo:
        print("  nothing to run; every trial is in the checkpoint")

    if cached:
        spent = [checkpoint.get(t, n) for t, n in cached]
        tokens = sum(v.prompt_tokens + v.completion_tokens for v in spent if v)
        per = tokens / len(spent) if spent else 0
        print()
        print(f"  banked trials cost {tokens:,} tokens, {per:,.0f} each")
        print(f"  so {len(todo)} more is roughly {per * len(todo):,.0f} tokens")
    return 0


def _regrade(domain: Domain, args) -> int:
    """Re-score finished trials from disk. Costs nothing, changes no transcript."""
    before = load_verdicts(args.regrade)
    after = [regrade(domain, verdict, args.strict_rule_7) for verdict in before]
    # Pass/fail flips are not the only thing a regrade can change. A rule that
    # adds a violation to a trial already failing on state flips nothing, and
    # reporting only flips hid exactly that: the strict reading of rule 7 found
    # the mechanism behind three failures and this printed "no verdict changed".
    moved = [
        (old, new) for old, new in zip(before, after)
        if old.passed != new.passed
        or {v.name for v in old.violations} != {v.name for v in new.violations}
    ]
    tasks = len({v.task_id for v in after})
    summary = Summary(solver="model", model=None, k=args.k, tasks=tasks,
                      verdicts=after, complete=False)

    reading = " under rule 7 as written" if args.strict_rule_7 else ""
    print(f"re-scored {len(after)} trials from {args.regrade}{reading}")
    for old, new in moved:
        was = "pass" if old.passed else "FAIL"
        now = "pass" if new.passed else "FAIL"
        verdict = f"{was} -> {now}" if old.passed != new.passed else f"{now}, same"
        gained = {v.name for v in new.violations} - {v.name for v in old.violations}
        lost = {v.name for v in old.violations} - {v.name for v in new.violations}
        change = ", ".join(
            [f"+{name}" for name in sorted(gained)] + [f"-{name}" for name in sorted(lost)]
        )
        why = change or "; ".join(new.reasons) or "clean"
        print(f"  {old.task_id:<28} trial {old.trial}  {verdict:<12} {why[:58]}")
    if not moved:
        print("  nothing changed: no verdict flipped and no violation moved")

    violations = Counter(f"{v.rule} {v.name}" for one in after for v in one.violations)
    if violations:
        print("\n  violations under this reading:")
        for rule, count in violations.most_common():
            print(f"    {count:>3}x  rule {rule}")

    complete = [
        task for task in {v.task_id for v in after}
        if len([v for v in after if v.task_id == task]) == args.k
    ]
    print(f"\n{len(complete)} of {len(domain.tasks)} tasks have all {args.k} trials.")
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"meta": {"regraded_from": str(args.regrade)},
                        "verdicts": [v.as_dict() for v in after]}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return 0


def _check(domain: Domain) -> int:
    """The harness self-check. No key, no cost, and it fails loudly."""
    checkpoint = Checkpoint.load(None)
    ok = True
    for solver, wanted in (("oracle", 1.0), ("mute", 0.0)):
        summary = run_solver(
            domain, solver, k=1, tasks=domain.tasks, client=None, scripted=True,
            checkpoint=checkpoint, max_turns=agent.MAX_TURNS, quiet=True,
        )
        got = summary.pass_at_1
        verdict = "ok" if abs(got - wanted) < 1e-9 else "BROKEN"
        print(f"{solver:<8} pass@1 {got:.3f}  (must be {wanted:.1f})  {verdict}")
        if verdict == "BROKEN":
            ok = False
            for bad in summary.verdicts:
                if (bad.passed) != (wanted == 1.0):
                    detail = "; ".join(bad.reasons + [v.name for v in bad.violations])
                    print(f"    {bad.task_id}: {detail or 'passed when it should not have'}")
    print(f"\n{len(domain.tasks)} tasks validated.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
