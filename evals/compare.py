"""Voice against its text control, and what the line had to do with it.

    python -m evals.compare runs/voice-report.json runs/text-control-report.json

Either argument may be a run file or a checkpoint. Two refusals come first,
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
from reruns.voice import channel_stats, entities_kept

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


def compare(voice: Side, text: Side, tasks: list[str], k: int) -> tuple[list[str], int]:
    lines: list[str] = []

    if voice.meta.get("model") != text.meta.get("model"):
        return [
            f"refused: voice ran on {voice.meta.get('model')!r} and text on "
            f"{text.meta.get('model')!r}. A gap across two models is not a voice effect.",
        ], 2
    if text.channel != "text" or voice.channel == "text":
        return [f"refused: expected a voice run and a text run, got {voice.channel!r} "
                f"and {text.channel!r}."], 2

    both_whole = whole(voice, tasks, k) and whole(text, tasks, k)
    lines.append(f"model {voice.meta.get('model')}, {len(tasks)} tasks x {k} trials")
    lines.append(f"line  {voice.channel}")
    lines.append("")
    if both_whole:
        lines.append(f"                    text     voice")
        lines.append(f"  pass@1            {pass_at_1(text.verdicts):.3f}    {pass_at_1(voice.verdicts):.3f}")
        lines.append(f"  pass^{k}            {pass_hat_k(text, tasks):.3f}    {pass_hat_k(voice, tasks):.3f}")
        lines.append(f"  policy violated   {violation_rate(text.verdicts):.3f}    "
                     f"{violation_rate(voice.verdicts):.3f}")
    else:
        lines.append(f"  NO SCORE: text has {len(text.verdicts)} and voice {len(voice.verdicts)} of "
                     f"{len(tasks) * k} trials. Counts below cover only tasks whole on both sides.")

    vt, tt = voice.by_task(), text.by_task()
    shown = [task for task in tasks if len(vt.get(task, [])) == k and len(tt.get(task, [])) == k]
    lines.append("")
    lines.append(f"  {'task':<30} text  voice  voice failures: exposed / clean line")
    exposed_fail = clean_fail = exposed_total = exposed_pass = 0
    for task in shown:
        fails = [v for v in vt[task] if not v.passed]
        exposed = [v for v in fails if lost_on_the_line(v)]
        exposed_fail += len(exposed)
        clean_fail += len(fails) - len(exposed)
        for verdict in vt[task]:
            if lost_on_the_line(verdict):
                exposed_total += 1
                exposed_pass += verdict.passed
        marks = f"{len(exposed)} / {len(fails) - len(exposed)}" if fails else "-"
        lines.append(f"  {task:<30} {sum(v.passed for v in tt[task])}/{k}   "
                     f"{sum(v.passed for v in vt[task])}/{k}    {marks}")

    lines.append("")
    lines.append(f"  voice failures on an exposed trial   {exposed_fail}")
    lines.append(f"  voice failures on a clean line       {clean_fail}   (cannot be the line)")
    if exposed_total:
        lines.append(f"  trials where the line lost an entity {exposed_total}, "
                     f"and the agent still passed {exposed_pass}")

    heard = channel_stats([v.transcript for task in shown for v in vt[task]])
    if heard:
        lines.append("")
        lines.append(f"  phone line  {heard['utterances']} utterances, mean WER {heard['mean_wer']:.3f}, "
                     f"{heard['verbatim']:.0%} word-perfect")
        for kind, tally in heard["entities"].items():
            lines.append(f"    {kind:<10} {tally['kept']} of {tally['said']} survived")
    return lines, 0 if both_whole else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.compare", description=__doc__)
    parser.add_argument("voice")
    parser.add_argument("text")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--domain", default=str(DEFAULT_DOMAIN))
    args = parser.parse_args(argv)
    tasks = [task.id for task in Domain.load(args.domain).tasks]
    lines, code = compare(Side.load(args.voice), Side.load(args.text), tasks, args.k)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
