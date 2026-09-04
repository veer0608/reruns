"""Trial-level resume.

pass^5 costs five times what pass@1 costs, and a free tier's daily token budget
is gone with no warning in any response header. Without this file, hitting the
cap on trial 4 of task 12 throws away the eleven tasks already paid for.

The unit is one trial, not one task. A task half measured is not a task, but
its finished trials are still real and are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from reruns.grade import Verdict


@dataclass
class Checkpoint:
    path: Path | None
    verdicts: dict[tuple[str, int], Verdict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "Checkpoint":
        if path is None:
            return cls(path=None)
        target = Path(path)
        if not target.is_file():
            return cls(path=target)
        raw = json.loads(target.read_text(encoding="utf-8"))
        verdicts = {}
        for entry in raw.get("verdicts", []):
            verdict = Verdict.from_dict(entry)
            # A trial that died on a quota or a dropped connection is not a
            # reading, and a resume that reuses it records a failure that never
            # happened. Skipping it here means the resume re-runs it, which is
            # the only honest thing to do with a trial nobody ever observed.
            if verdict.error:
                continue
            verdicts[(verdict.task_id, verdict.trial)] = verdict
        return cls(path=target, verdicts=verdicts, meta=raw.get("meta", {}))

    def has(self, task_id: str, trial: int) -> bool:
        return (task_id, trial) in self.verdicts

    def get(self, task_id: str, trial: int) -> Verdict | None:
        return self.verdicts.get((task_id, trial))

    def add(self, verdict: Verdict) -> None:
        self.verdicts[(verdict.task_id, verdict.trial)] = verdict
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "verdicts": [v.as_dict() for v in self.verdicts.values()],
        }
        # Written whole and moved into place. A run killed mid-write would
        # otherwise leave truncated JSON, and the resume that was meant to
        # rescue the day would fail to parse it.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
