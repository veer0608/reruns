"""Loading a domain: its policy, its seed, and its tasks.

A domain is a directory with three files -- `policy.md`, `seed.json`,
`tasks.json`. Nothing in the package hardcodes "retail", so a second domain is
a directory and not a code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .store import Seed, Store
from .tools import TOOL_NAMES

DEFAULT_DOMAIN = Path(__file__).resolve().parent.parent / "domains" / "retail"


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    goal: str
    scripted_user: tuple[str, ...]
    solution: tuple[dict, ...]
    must_call: tuple[str, ...]
    forbid_call: tuple[str, ...]
    expect: dict = field(default_factory=dict)
    #: Whether this customer asks to be refunded to their original payment
    #: method. Declared here rather than inferred from what the simulated
    #: customer says, because a regex over generated prose got it wrong three
    #: times in sixty trials and each miss was scored against the agent.
    asks_for_card: bool = False

    @property
    def read_only(self) -> bool:
        """A task whose correct outcome is that nothing in the world moved."""
        return not self.expect

    @classmethod
    def from_dict(cls, raw: dict) -> "Task":
        return cls(
            id=raw["id"],
            title=raw.get("title", raw["id"]),
            goal=raw["goal"],
            scripted_user=tuple(raw.get("scripted_user", ())),
            solution=tuple(raw.get("solution", ())),
            must_call=tuple(raw.get("must_call", ())),
            forbid_call=tuple(raw.get("forbid_call", ())),
            expect=raw.get("expect", {}) or {},
            asks_for_card=bool(raw.get("asks_for_card", False)),
        )


@dataclass(frozen=True)
class Domain:
    name: str
    root: Path
    policy: str
    seed: Seed
    tasks: tuple[Task, ...]

    @classmethod
    def load(cls, root: str | Path = DEFAULT_DOMAIN) -> "Domain":
        root = Path(root)
        raw = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
        return cls(
            name=root.name,
            root=root,
            policy=(root / "policy.md").read_text(encoding="utf-8"),
            seed=Seed.load(root / "seed.json"),
            tasks=tuple(Task.from_dict(entry) for entry in raw),
        )

    def store(self) -> Store:
        """A fresh world. Never share one between trials."""
        return Store(self.seed)

    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"no task {task_id!r} in domain {self.name}")

    def select(self, ids: str | None) -> tuple[Task, ...]:
        if not ids:
            return self.tasks
        wanted = [part.strip() for part in ids.split(",") if part.strip()]
        return tuple(self.task(one) for one in wanted)


def validate(domain: Domain) -> list[str]:
    """Problems in the task file that would make a number meaningless.

    Run by the tests and by `--check`. A task that names a tool which does not
    exist, or expects a table the schema does not have, does not fail loudly at
    run time -- it fails as an agent that could not do it.
    """
    problems: list[str] = []
    seen: set[str] = set()
    snapshot = domain.store().snapshot()
    for task in domain.tasks:
        if task.id in seen:
            problems.append(f"{task.id}: duplicate task id")
        seen.add(task.id)
        if not task.scripted_user:
            problems.append(f"{task.id}: no scripted_user, so it cannot run without a key")
        for name in (*task.must_call, *task.forbid_call):
            if name not in TOOL_NAMES:
                problems.append(f"{task.id}: unknown tool {name!r}")
        for name in task.must_call:
            if name in task.forbid_call:
                problems.append(f"{task.id}: {name} is both required and forbidden")
        for step in task.solution:
            call = step.get("call")
            if call and call.get("name") not in TOOL_NAMES:
                problems.append(f"{task.id}: solution calls unknown tool {call.get('name')!r}")
        methods = {row.get("method") for row in task.expect.get("refunds", {}).values()}
        if ("original_payment" in methods) != task.asks_for_card:
            problems.append(
                f"{task.id}: expects {methods or 'no refund'} but asks_for_card="
                f"{task.asks_for_card}; rule 9 would contradict the expected state"
            )
        for table, rows in task.expect.items():
            if table not in snapshot:
                problems.append(f"{task.id}: expects unknown table {table!r}")
                continue
            for pk, columns in rows.items():
                known = snapshot[table].get(pk)
                for column in columns:
                    if known is not None and column not in known:
                        problems.append(f"{task.id}: expects unknown column {table}.{column}")
    return problems
