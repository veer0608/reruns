"""Voice against its text control, and what the line had to do with it.

    python -m evals.compare runs/text-control-report.json runs/voice-report.json runs/voice-aware-report.json

One text run and one or more voice runs, in any order: each is recognised by
the channel recorded in its metadata. Any may be a run file or a checkpoint. Two refusals come first,
because each one has already been needed:

- **Different models are not a comparison.** v3 ran on a moving alias, and the
  pinned model failed a task over voice for a reason visible in the transcript
  to have nothing to do with the line. A gap across two models books the model
  difference as voice damage.
- **An incomplete run gets no score**, the same rule the runner enforces. Tasks
  that have all k trials on both sides are still shown, as counts.

Failures are then split by what the line did in that trial, from the
transcript alone. "Exposed" means some email or order id the customer said did
not survive the recogniser. That is exposure, not blame: an exposed trial can
still fail for a reason of its own. A failure on a clean line, though, cannot
be the line's, which is what makes the split worth printing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from reruns.dataset import DEFAULT_DOMAIN, Domain
from reruns.grade import Verdict
from reruns.voice import ENTITIES, _squash, channel_stats, entities_kept

from .runner import load_verdicts


@dataclass
class Side:
    path: str
    meta: dict
    verdicts: list[Verdict]

    @classmethod
    def load(cls, path: str) -> "Side":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(path=path, meta=raw.get("meta", {}), verdicts=load_verdicts(path))

    @property
    def channel(self) -> str:
        return self.meta.get("channel", "text")

    def by_task(self) -> dict[str, list[Verdict]]:
        out: dict[str, list[Verdict]] = {}
        for verdict in self.verdicts:
            out.setdefault(verdict.task_id, []).append(verdict)
        return out


def lost_on_the_line(verdict: Verdict) -> list[str]:
    """Entity kinds said by the customer that the recogniser did not deliver."""
    lost = []
    for event in verdict.transcript:
        if event.get("kind") != "user" or "spoken" not in event:
            continue
        for kind, (said, kept) in entities_kept(event["spoken"], event["text"]).items():
            lost += [kind] * (said - kept)
    return lost


FATES = ("kept", "dropped", "misheard")


def email_fate(spoken: str, heard: str) -> str | None:
    """What the line did to the first email in an utterance, or None if none was said.

    kept      it survived, counting spoken "at" and "dot" as recoverable
    dropped   the transcript has no email at all
    misheard  the transcript has an email, and it is the wrong one
    """
    said = ENTITIES["email"].search(spoken)
    if not said:
        return None
    heard_squashed = _squash(heard)
    if said.group(0).lower().rstrip(".") in heard_squashed:
        return "kept"
    return "misheard" if ENTITIES["email"].search(heard_squashed) else "dropped"


def opening_email_fate(verdict: Verdict) -> str | None:
    """The email fate of the customer's first message in a voice trial."""
    users = [e for e in verdict.transcript if e.get("kind") == "user" and "spoken" in e]
    return email_fate(users[0]["spoken"], users[0]["text"]) if users else None


def whole(side: Side, tasks: list[str], k: int) -> bool:
    counts = side.by_task()
    return all(len(counts.get(task, [])) == k for task in tasks)


def pass_at_1(verdicts: list[Verdict]) -> float:
    return sum(v.passed for v in verdicts) / len(verdicts)


def pass_hat_k(side: Side, tasks: list[str]) -> float:
    counts = side.by_task()
    return sum(all(v.passed for v in counts[task]) for task in tasks) / len(tasks)


def violation_rate(verdicts: list[Verdict]) -> float:
    return sum(not v.policy_ok for v in verdicts) / len(verdicts)


#: The caller in this task gives a genuinely wrong email, so a line that
#: "mishears" it is not the line's doing. Left out of the fate table.
FATE_EXCLUDED = frozenset({"unknown_email"})


def fate_table(verdicts: list[Verdict]) -> dict[str, list[bool]]:
    """Pass or fail per trial, grouped by what the line did to the opening email."""
    table: dict[str, list[bool]] = {fate: [] for fate in FATES}
    for verdict in verdicts:
        if verdict.task_id in FATE_EXCLUDED:
            continue
        fate = opening_email_fate(verdict)
        if fate:
            table[fate].append(verdict.passed)
    return table


def arm_name(side: Side) -> str:
    return "text" if side.channel == "text" else ("aware" if side.channel.endswith("+aware") else "voice")


def refusal(sides: list[Side]) -> str | None:
    models = {side.meta.get("model") for side in sides}
    if len(models) > 1:
        return (f"refused: the runs are on {sorted(map(str, models))}. "
                "A gap across two models is not a voice effect.")
    texts = [side for side in sides if side.channel == "text"]
    if len(texts) != 1 or len(sides) < 2:
        return f"refused: expected one text run and at least one voice run, got {[s.channel for s in sides]}."
    channels = [side.channel for side in sides]
    if len(set(channels)) != len(channels):
        return "refused: two runs on the same channel are one measurement, not a comparison."
    return None


def compare(sides: list[Side], tasks: list[str], k: int) -> tuple[list[str], int]:
    """One text control against one or more voice arms, in any order."""
    why = refusal(sides)
    if why:
        return [why], 2
    text = next(side for side in sides if side.channel == "text")
    # Fixed column order, whatever the argument order: text, voice, aware.
    voices = sorted((side for side in sides if side.channel != "text"),
                    key=lambda side: side.channel.endswith("+aware"))
    arms = [text] + voices
    names = [arm_name(side) for side in arms]

    all_whole = all(whole(side, tasks, k) for side in arms)
    lines = [f"model {text.meta.get('model')}, {len(tasks)} tasks x {k} trials"]
    lines += [f"line  {side.channel}" for side in voices]
    lines.append("")
    if all_whole:
        lines.append("  " + " " * 18 + "".join(f"{n:>9}" for n in names))
        lines.append("  pass@1            " + "".join(f"{pass_at_1(s.verdicts):>9.3f}" for s in arms))
        lines.append(f"  pass^{k}            " + "".join(f"{pass_hat_k(s, tasks):>9.3f}" for s in arms))
        lines.append("  policy violated   " + "".join(f"{violation_rate(s.verdicts):>9.3f}" for s in arms))
    else:
        counts = ", ".join(f"{n} {len(s.verdicts)}" for n, s in zip(names, arms))
        lines.append(f"  NO SCORE: {counts} of {len(tasks) * k} trials. "
                     "Counts below cover only tasks whole in every run.")

    by_task = [side.by_task() for side in arms]
    shown = [task for task in tasks if all(len(b.get(task, [])) == k for b in by_task)]
    lines.append("")
    lines.append(f"  {'task':<30}" + "".join(f"{n:>7}" for n in names))
    for task in shown:
        lines.append(f"  {task:<30}" + "".join(f"{sum(v.passed for v in b[task])}/{k}".rjust(7) for b in by_task))

    for name, side, tasks_of in zip(names[1:], voices, by_task[1:]):
        verdicts = [v for task in shown for v in tasks_of[task]]
        fails = [v for v in verdicts if not v.passed]
        exposed = [v for v in verdicts if lost_on_the_line(v)]
        exposed_fail = sum(1 for v in fails if lost_on_the_line(v))
        lines.append("")
        lines.append(f"  {name}")
        lines.append(f"    failures on an exposed trial   {exposed_fail}")
        lines.append(f"    failures on a clean line       {len(fails) - exposed_fail}   (cannot be the line)")
        if exposed:
            lines.append(f"    trials where the line lost an entity {len(exposed)}, "
                         f"and the agent still passed {sum(v.passed for v in exposed)}")
        table = fate_table(verdicts)
        for fate in FATES:
            outcomes = table[fate]
            lines.append(f"    opening email {fate:<9} {sum(outcomes):>3} passed of {len(outcomes)}")

    heard = channel_stats([v.transcript for task in shown for v in by_task[1][task]]) if voices else None
    if heard:
        lines.append("")
        lines.append(f"  phone line  {heard['utterances']} utterances, mean WER {heard['mean_wer']:.3f}, "
                     f"{heard['verbatim']:.0%} word-perfect")
        for kind, tally in heard["entities"].items():
            lines.append(f"    {kind:<10} {tally['kept']} of {tally['said']} survived")
    return lines, 0 if all_whole else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.compare", description=__doc__)
    parser.add_argument("runs", nargs="+", help="one text run and one or more voice runs, any order")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--domain", default=str(DEFAULT_DOMAIN))
    args = parser.parse_args(argv)
    tasks = [task.id for task in Domain.load(args.domain).tasks]
    lines, code = compare([Side.load(path) for path in args.runs], tasks, args.k)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
