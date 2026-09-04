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
) -> Verdict:
    """One task, one trial, in a world nobody else has touched."""
    store = domain.store()
    before = store.snapshot()
    trace = Trace()
    toolbox = Toolbox(store, trace)
    user = agent.build_user(task, client, scripted=scripted)
    try:
        episode = agent.run(solver, task, domain.policy, toolbox, user, client, max_turns)
    except Exception as exc:  # noqa: BLE001 - a crashed trial is a failed trial, not a dead run
        episode = agent.Episode(trace=trace, error=f"{type(exc).__name__}: {exc}")
    after = store.snapshot()
    store.close()

    verdict = grade(task, trial, trace, before, after, domain.seed.now, error=episode.error)
    verdict.prompt_tokens = episode.prompt_tokens
    verdict.completion_tokens = episode.completion_tokens
    verdict.cost_usd = episode.cost_usd
    return verdict


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
            )
            verdicts.append(verdict)
            if solver == "model":
                checkpoint.add(verdict)
            if not quiet:
                mark = "pass" if verdict.passed else "FAIL"
                note = "" if verdict.passed else "  " + "; ".join(
                    verdict.reasons + [v.name for v in verdict.violations]
                )[:110]
                print(f"  {task.id:<28} trial {trial}/{k}  {mark}{note}")
            if verdict.error and verdict.error.startswith("quota"):
                print(f"\n  daily allowance exhausted during {task.id} trial {trial}.")
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None, help="write the run JSON here")
    parser.add_argument("--check", action="store_true",
                        help="validate the task file, then assert oracle 1.0 and mute 0.0")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    domain = Domain.load(args.domain)
    problems = validate(domain)
    if problems:
        print("task file is not valid:")
        for problem in problems:
            print(f"  {problem}")
        return 2

    if args.check:
        return _check(domain)

    solvers = [name.strip() for name in args.solvers.split(",") if name.strip()]
    tasks = domain.select(args.tasks)
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
