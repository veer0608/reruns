# reruns

A multi-turn support-agent benchmark scored on final state, tool calls and
policy adherence, reported as pass^k. See README.md for what it is and why.
This file is the operational half.

## Layout

- `reruns/` -- the package: `store.py`, `tools.py`, `policy.py`, `agent.py`,
  `user.py`, `grade.py`, `dataset.py`, `llm.py`, `cli.py`
- `evals/` -- `runner.py` (pass@1 and pass^k), `checkpoint.py` (trial-level resume)
- `domains/retail/` -- `policy.md`, `seed.json`, `tasks.json`. A domain is a
  directory with those three files; nothing in the package hardcodes retail.
- `runs/` -- gitignored run JSON. Keep every one. Per-trial tokens live there, so
  a price correction never costs another run.

## Commands

Run from the repo root. Python 3.11. Everything below is free and offline.

```bash
python -m pytest -q
```

```bash
python -m evals.runner --check
```

```bash
python -m reruns replay refund_cable_only
```

## Before any run that calls a model

`~/.claude/CLAUDE.md` has the machine-wide version of this. The parts that bite
hardest here:

- **A trial is two model calls, not one.** The agent and the simulated customer
  both cost tokens, and `--k 5` multiplies the lot. Budget as though a task
  costs five to eight times a single-turn eval question.
- Probe the budget with a **full-size** request (`--solvers model --tasks
  refund_kettle --k 1`). A small probe succeeds while the day's allowance is gone.
- Groq's binding limit is **tokens per day, per model**, and it is in no header.
  Prefer Gemini for anything long-running.
- Always pass `--checkpoint runs/NAME.json`. Without it, a cap on trial 4 of
  task 12 throws away everything already paid for.
- **`--out` and `--checkpoint` must be different files.** They are different
  formats and the run file is written last, so pointing both at one path
  overwrites the checkpoint at the moment you most need it. The runner now
  refuses, because this destroyed a 60 trial checkpoint on the first real run.
- A grading bug does not need a re-run. `--regrade` re-scores saved transcripts
  offline. Reach for it before spending another day's tokens.
- **`--dry-run` before every resume.** It lists the trials that would execute
  and estimates the tokens from what the banked ones actually cost. A resume
  that has silently lost its checkpoint looks identical to one that has not
  until the allowance is gone.
- `--scripted-user` makes a run deterministic and much cheaper. Use it while
  debugging the agent loop. **Never report a pass^k from it**: with a fixed
  script the five trials differ only in model sampling, which is not the
  question pass^k is asking.

## Rules that are not style preferences

- **A run that hits the cap is abandoned and gets no score.** Trials that never
  ran are indistinguishable from trials that failed. `report()` already refuses
  to print a percentage for an incomplete run; do not compute one by hand.
- **The oracle must score 1.000 and the mute solver 0.000.** If either drifts,
  the scorer is broken and no model number from it is worth reading. CI asserts
  this on every push.
- **The store enforces nothing.** `StoreError` is for structurally impossible
  requests only: an unknown id, a non-numeric amount. The moment it starts
  meaning "not allowed", the policy column stops measuring the agent, because a
  refused violation leaves a compliant-looking trace.
- **Every mutation stays deterministic.** Refund ids come from a counter,
  timestamps from `seed.now`. A `datetime.now()` anywhere in `store.py` would
  show up as an agent that is inconsistent when the harness is.
- **Each trial gets a fresh `Store`.** Never share one across trials.
- **Every rule number in `policy.py` must appear in `policy.md`.** The agent is
  graded on the document it was handed. `test_policy.py` asserts it.
- **Every task that expects no state change must have a `must_call`.** Otherwise
  it passes on silence and quietly lifts every score in the table.

## Environment

The key lives in a gitignored `.env` (`GROQ_API_KEY`, `GEMINI_API_KEY`), parsed
by `llm.py`'s own `load_dotenv`, which handles the UTF-16 PowerShell writes.
`llm.py` is lifted from schemablind on purpose; fixes worth having belong in
both.

## The failure mode to watch for

The simulated customer inventing a requirement the task never gave it. It did
this on the very first live trial: asked to be refunded to its card, the agent
correctly obliged, and a task expecting store credit failed for a reason that
had nothing to do with the agent. Two guards are in place and both have tests:
`USER_SYSTEM` forbids inventing preferences, and every task whose expected
state names a refund method says so in the customer's brief.

If a run shows tasks failing on a field the customer could have opinions about,
read the transcript in the run JSON before touching the agent. It is more often
the brief than the model.
