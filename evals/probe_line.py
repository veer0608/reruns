"""The phone line on its own: what survives it, with no agent involved.

    python -m evals.probe_line                      the whole grid, 3 seeds
    python -m evals.probe_line --configs baseline   one config

Every task's first two scripted lines go down each line config. No Gemini is
called; the only cost is Groq Whisper, whose free tier is a separate budget
from the agent's. Transcripts land in the same cache as runs (keyed on line,
seed and text), so a probe stopped by the daily cap resumes for free, and the
baseline's openings on seeds 1 to 3 are the voice run's own transcripts.

The grid moves one thing at a time away from the line the voice run uses:
accent, noise, recogniser, and trailing silence, the candidate fix for
Whisper dropping a short fragment at the end of an utterance.

A config missing any line or seed gets no numbers, the runner's rule one level
down: a partial average describes whichever lines happened to finish.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from reruns.dataset import DEFAULT_DOMAIN, Domain
from reruns.llm import LLMError, QuotaExhausted
from reruns.voice import ASR_KEY_ENV, LineConfig, build_line, channel_stats, seed_for

from .compare import FATES, email_fate

REPO = Path(__file__).resolve().parent.parent

GRID: dict[str, LineConfig] = {
    "baseline": LineConfig(),
    "voice en-IN male": LineConfig(voice="en-IN-PrabhatNeural"),
    "voice en-US": LineConfig(voice="en-US-AriaNeural"),
    "voice en-GB": LineConfig(voice="en-GB-SoniaNeural"),
    "noise 20 dB": LineConfig(snr_db=20),
    "noise 10 dB": LineConfig(snr_db=10),
    "noise 5 dB": LineConfig(snr_db=5),
    "whisper-large-v3": LineConfig(asr_model="whisper-large-v3"),
    "pad 500 ms": LineConfig(pad_ms=500),
    "pad 500 ms, large-v3": LineConfig(pad_ms=500, asr_model="whisper-large-v3"),
}


def scripted_lines(domain: Domain, per_task: int = 2) -> list[tuple[str, int, str]]:
    """(task id, turn, text) for the first scripted lines of every task."""
    return [(task.id, turn, text)
            for task in domain.tasks
            for turn, text in enumerate(task.scripted_user[:per_task])]


def summarise(rows: list[dict]) -> dict:
    stats = channel_stats([[{"kind": "user", "spoken": r["spoken"], "text": r["heard"]}] for r in rows])
    fates = {fate: 0 for fate in FATES}
    for row in rows:
        fate = email_fate(row["spoken"], row["heard"])
        if fate:
            fates[fate] += 1
    return {**stats, "email_fates": fates}


def probe(line_for, configs: dict[str, LineConfig], lines, seeds: int, quiet: bool = False) -> dict:
    """Run every (config, line, seed), config by config. Stops at the daily cap."""
    results: dict[str, dict] = {}
    stopped = None
    for name, config in configs.items():
        line = line_for(config)
        rows = []
        try:
            for seed in range(1, seeds + 1):
                for task_id, turn, text in lines:
                    heard = line.hear(text, seed_for(task_id, seed, turn))
                    rows.append({"task": task_id, "turn": turn, "seed": seed,
                                 "spoken": text, "heard": heard.heard, "cached": heard.cached})
        except (QuotaExhausted, LLMError) as exc:
            stopped = f"{name}: {type(exc).__name__}: {str(exc)[:160]}"
            results[name] = {"line": config.descriptor, "complete": False, "rows": rows}
            break
        results[name] = {"line": config.descriptor, "complete": True,
                         "summary": summarise(rows), "rows": rows}
        if not quiet:
            fresh = sum(not r["cached"] for r in rows)
            print(f"  {name:<22} done, {fresh} new transcriptions")
    return {"seeds": seeds, "lines": len(lines), "stopped": stopped, "configs": results}


def report(result: dict) -> str:
    out = [f"\n{result['lines']} lines x {result['seeds']} seeds per config\n",
           f"  {'config':<22} {'WER':>6} {'perfect':>8} {'emails kept':>12} {'dropped':>8} "
           f"{'misheard':>9} {'order ids':>10}"]
    for name, entry in result["configs"].items():
        if not entry["complete"]:
            out.append(f"  {name:<22} INCOMPLETE, no numbers ({len(entry['rows'])} rows banked)")
            continue
        s = entry["summary"]
        fates = s["email_fates"]
        said = sum(fates.values())
        order = s["entities"].get("order_id", {"kept": 0, "said": 0})
        out.append(f"  {name:<22} {s['mean_wer']:>6.3f} {s['verbatim']:>8.0%} "
                   f"{fates['kept']:>5} of {said:<4} {fates['dropped']:>8} {fates['misheard']:>9} "
                   f"{order['kept']:>4} of {order['said']:<3}")
    if result["stopped"]:
        out.append(f"\n  stopped: {result['stopped']}. Re-run the same command; finished lines are cached.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.probe_line", description=__doc__)
    parser.add_argument("--configs", default=None, help="comma separated names from the grid")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--domain", default=str(DEFAULT_DOMAIN))
    parser.add_argument("--out", default=str(REPO / "runs" / f"probe-{date.today().isoformat()}.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.configs.split(",")] if args.configs else list(GRID)
    unknown = [n for n in names if n not in GRID]
    if unknown:
        print(f"unknown config {unknown}; the grid has {list(GRID)}")
        return 2
    cache = REPO / "runs" / "voice-cache"
    if build_line(LineConfig(), cache) is None:
        print(f"the probe needs {ASR_KEY_ENV} for speech recognition, and it is not set.")
        return 2

    result = probe(lambda config: build_line(config, cache), {n: GRID[n] for n in names},
                   scripted_lines(Domain.load(args.domain)), args.seeds, args.quiet)
    print(report(result))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {target}")
    return 3 if result["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
