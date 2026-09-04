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

## Escalation is asked for, not required

Rule 12 of `policy.md` still tells the agent to escalate with
`escalate_to_human` and stop. The grader no longer insists on the call: refusal
tasks require `find_customer` instead, so a considered refusal in words passes
and an agent that never read the message still fails.

This was Veer's call after the first run, where six of eleven failures were
agents that refused correctly, explained the policy accurately, offered a
sensible alternative, and simply did not reach for the tool. Re-scoring the
banked trials moved those six to passes and nothing else, which is worth
knowing about the headline: a single grading decision was most of the gap.
Say so wherever the number is reported.

`policy.md` was deliberately left unedited. Its text goes into the system
prompt verbatim, so changing a word of it changes agent behaviour and
invalidates every banked trial. The wording can be revisited at the start of
the next measurement, not the middle of this one.

`must_call` counts attempts and `forbid_call` counts successes, for a reason
`check_calls` documents: `unknown_email`'s correct first move is a lookup that
fails, so requiring a successful one requires the impossible.

## Known limitation: the customer can hang up mid-action

The episode ends when the simulated customer stops, and it stops on what the
agent *says*. An agent that announces "I am going to refund 45.99" and would
have called the tool on its next turn never gets that turn: the customer reads
the announcement as completion, replies with the stop token, and the loop ends
with the world unchanged.

This cost `ambiguous_order` two of its four measured trials in the first run.
Both transcripts end on the announcement; the two that passed are identical
except the agent called the tool in the same turn it spoke.

**Deliberately not fixed during the first run.** The fix -- give the agent one
final turn after the customer stops, to complete anything it has announced --
changes agent behaviour, so it invalidates every trial measured without it.
The plan is to finish the current run, publish it with this named as a
limitation, then fix it and re-run as a second measurement. Do not quietly
change this mid-run.
