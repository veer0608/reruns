"""Three verdicts on one conversation, kept apart on purpose.

**state** -- did the world end up the way the task says it should, and did
nothing else move. Both halves matter. An agent that refunds the right cable
and also the chair has reached the expected state for the cable, and it is not
a success.

**calls** -- did the required tools get used and the forbidden ones stay
unused. This is what stops a do-nothing agent from scoring on the six tasks
whose correct outcome is that nothing changed.

**policy** -- did it get there by an allowed route (see `policy.py`).

A task passes only when all three hold. They are also reported separately,
because the gap between "correct state" and "correct state and conduct" is one
of the two numbers this project exists to show. The other is what happens when
you run the same task again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Task
from .policy import Violation, check as check_policy
from .store import Snapshot, diff
from .tools import Trace


@dataclass
class Verdict:
    task_id: str
    trial: int
    state_ok: bool
    calls_ok: bool
    violations: list[Violation] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    #: The whole conversation, kept on the verdict and written to the run file.
    #: A failing trial that cannot be read is a failing trial nobody can act
    #: on, and re-running one to see what happened costs a second budget.
    transcript: list[dict] = field(default_factory=list)

    @property
    def policy_ok(self) -> bool:
        return not self.violations

    @property
    def passed(self) -> bool:
        return self.state_ok and self.calls_ok and self.policy_ok and self.error is None

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "passed": self.passed,
            "state_ok": self.state_ok,
            "calls_ok": self.calls_ok,
            "policy_ok": self.policy_ok,
            "violations": [v.as_dict() for v in self.violations],
            "reasons": self.reasons,
            "turns": self.turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "transcript": self.transcript,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Verdict":
        return cls(
            task_id=raw["task_id"],
            trial=raw["trial"],
            state_ok=raw["state_ok"],
            calls_ok=raw["calls_ok"],
            violations=[
                Violation(v["rule"], v["name"], v["detail"]) for v in raw.get("violations", ())
            ],
            reasons=list(raw.get("reasons", ())),
            turns=raw.get("turns", 0),
            prompt_tokens=raw.get("prompt_tokens", 0),
            completion_tokens=raw.get("completion_tokens", 0),
            cost_usd=raw.get("cost_usd"),
            error=raw.get("error"),
            transcript=list(raw.get("transcript", ())),
        )


def check_state(task: Task, before: Snapshot, after: Snapshot) -> list[str]:
    """Empty list means the world is exactly as the task specified."""
    problems: list[str] = []
    actual = diff(before, after)

    for table, rows in actual.items():
        for pk, columns in rows.items():
            expected_row = task.expect.get(table, {}).get(pk)
            for column, (old, new) in columns.items():
                if expected_row is None or column not in expected_row:
                    problems.append(
                        f"unexpected change {table}.{pk}.{column}: {old!r} -> {new!r}"
                    )

    for table, rows in task.expect.items():
        for pk, columns in rows.items():
            row = after.get(table, {}).get(pk)
            if row is None:
                problems.append(f"missing {table}.{pk}")
                continue
            for column, wanted in columns.items():
                if row.get(column) != wanted:
                    problems.append(
                        f"{table}.{pk}.{column} is {row.get(column)!r}, expected {wanted!r}"
                    )
    return sorted(problems)


def check_calls(task: Task, trace: Trace) -> list[str]:
    """Required calls count attempts; forbidden calls count only what landed.

    The asymmetry is deliberate. "You must have looked them up" is satisfied by
    looking them up, even when the lookup comes back empty -- `unknown_email`
    is a task whose correct first move is a `find_customer` that fails, and
    requiring a successful one there requires the impossible. "You must not
    have refunded this" is about the world, so it asks whether a refund
    actually happened.

    Requiring only the attempt does not let a broken agent through: a task that
    expects a refund still has to end with the refund in the database, and a
    call that failed put nothing there.
    """
    tried = {call.name for call in trace.calls}
    landed = {call.name for call in trace.calls if call.ok}
    problems = [f"never called {name}" for name in task.must_call if name not in tried]
    problems += [f"called {name}" for name in task.forbid_call if name in landed]
    return sorted(problems)


def grade(
    task: Task,
    trial: int,
    trace: Trace,
    before: Snapshot,
    after: Snapshot,
    now: str,
    *,
    error: str | None = None,
) -> Verdict:
    state_problems = check_state(task, before, after)
    call_problems = check_calls(task, trace)
    return Verdict(
        task_id=task.id,
        trial=trial,
        state_ok=not state_problems,
        calls_ok=not call_problems,
        violations=check_policy(trace, before, now, asked_for_card=task.asks_for_card),
        reasons=state_problems + call_problems,
        turns=len([e for e in trace.events if e.get("kind") == "assistant"]),
        error=error,
        transcript=list(trace.events),
    )
