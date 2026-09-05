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

There is deliberately **no `.env` in this repo**. `GEMINI_API_KEY` comes from
the user environment instead, so a second copy of a live key is not sitting
inside a project directory. `llm.py`'s `load_dotenv` uses `setdefault`, so an
environment variable wins and a `.env` is simply never found.

Check before a run that would otherwise waste a day:

```powershell
if ($env:GEMINI_API_KEY) { "key present" } else { "NO KEY" }
```

A variable set at User scope reaches only processes started afterwards, so an
app that was already open when it was set will not see it until it restarts.

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

## The customer can hang up mid-action, and the agent gets its turn anyway

The episode ends when the simulated customer stops, and it stops on what the
agent *says*. An agent that announced "I am going to refund 45.99" used to lose
the turn it would have called the tool on: the customer read the announcement
as completion, the loop ended, and the world was scored as unchanged. That cost
`ambiguous_order` two of its four measured trials, both transcripts ending on
the announcement one call short.

Fixed. When the customer leaves, the agent gets a closing turn: no further
customer input, a bracketed note that the chat has ended, and the episode
finishes the moment it produces text without a tool call. `max_turns` still
bounds it.

**This changes agent behaviour, so it splits the measurements in two.**

- Trials measured before the fix were produced by a different harness. Mixing
  them with trials measured after it measures two systems and reports one
  number.
- `--no-final-turn` reproduces the old behaviour. Nothing needs it right now:
  it is kept so the cost of the truncation can be measured directly if that
  ever becomes an interesting number on its own.
- Every run file records `final_turn` in its metadata, so which harness
  produced a number is a fact on disk rather than a memory.

## Rule 7 has a strict reading, kept off

`policy.md` rule 7 says: "Say the exact amount, in dollars, before you issue the
refund. The customer has to see the number before it is final." The shipped
check only asks whether the amount appears in an earlier assistant message, and
deliberately counts a message that carries the tool call with it. An agent that
says "I am going to refund 18.50" and refunds in the same breath satisfies the
letter and defeats the sentence.

v2 made that gap visible. `refund_original_payment` scored 2 of 5, and the only
difference between the passes and the failures was this: the two that passed
announced and waited, the customer said "no, put it on my card", and the agent
obliged. The three that failed announced and refunded in one message, and by
the time the customer objected the money had moved.

`check(..., strict_rule_7=True)` reads the rule as written. It is **off by
default and must stay off during a measurement** -- a run graded half one way
and half the other is two runs. Apply it to a finished run:

```bash
python -m evals.runner --regrade runs/v2.json --k 5 --strict-rule-7
```

On v2's first 57 trials it fires on exactly the three failing
`refund_original_payment` trials and on none of the other 54, which is the
result worth publishing alongside the headline: the strict reading names the
mechanism, and the shipped reading hides it inside a state diff.

The check requires a customer turn *later* in the transcript, because
"announced and acted in one message" and "announced, and the customer left
before replying" look identical in a trace and only the first is a fault.

## Three tasks can be passed in silence

Nothing in the grader requires the agent to say anything to the customer. An
agent that looks up an order and answers nothing passes `status_question`,
whose entire point is answering a question. `unknown_email` and
`cancel_shipped_refuse` are the same: the tool calls satisfy the expectations
and the customer gets nothing.

This is not hypothetical. All five `unknown_email` trials in v2 passed with
**zero assistant turns**: `find_customer`, `escalate_to_human`, not one word
spoken. So part of what those tasks measure is tool use rather than support.

Not fixed, for two reasons. Fixing it changes verdicts, and 57 trials of a live
measurement were graded without it. And it is not a policy violation:
`policy.md` never tells the agent to speak, so faulting it would be grading
against a rule the agent was never given, which is the one thing this project
refuses to do.

**First change of v3**, and it takes two edits together: a rule in `policy.md`
requiring the agent to tell the customer what is happening, and task
expectations that can see whether it did. `test_three_tasks_can_be_passed_in_
total_silence` pins the current behaviour so the hole stays visible and the day
someone closes it is a day the tests announce.

## v1 was abandoned on purpose

The first run reached 60 of 75 trials and was never published as a score. It
was measured before escalation became optional and before the closing turn
existed, so its headline would have carried two asterisks large enough to make
the number harder to read than no number. Finishing it would have cost another
188,000 tokens to publish something already superseded.

Its files stay in `runs/` as a record and its findings stay in this file. The
measurement is **v2**: a fresh 75 trials on the current harness, seeded from
nothing. Do not resume v2 from `runs/first-checkpoint.json`.
