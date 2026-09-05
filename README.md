# reruns

[![ci](https://github.com/veer0608/reruns/actions/workflows/ci.yml/badge.svg)](https://github.com/veer0608/reruns/actions/workflows/ci.yml)

A support agent benchmark that runs every task **five times** and reports what
survives.

```
> The USB-C cable from my recent order doesn't work. Refund just the cable.

  find_customer     -> c_3, Priya Raman
  list_orders       -> o_1045 delivered, o_1046 pending
  get_order         -> Ergo Chair 24900, USB-C Cable 990
  "The cable is 9.90. I'll refund that and leave the chair alone."
  refund_item       -> rf_2, 990, store_credit

  state    the cable is refunded and nothing else moved     pass
  calls    find_customer, get_order, refund_item            pass
  policy   11 rules checked against the transcript          pass
```

Run that task once and you learn whether the agent can do it. Run it five times
and you learn whether it can be relied on to, which is a different and much
less flattering question. An agent scoring 0.60 on single attempts, if its
trials were independent, would sit near **0.08** on five.

---

## What is being measured

Every other agent project in this account scores a single turn against a world
that cannot be changed. Retrieval, SQL, extraction: ask, answer, compare. That
leaves out most of what makes a real support agent hard.

Here the agent is dropped into a conversation it does not control, with a
customer who wants something and will not say all of it at once, and tools that
write to a database. Three things are then true at the end of the conversation
that were not true before:

- the database is in some state, and it is either the right one or it is not
- the transcript either followed the written policy or it did not
- neither of those is knowable from the agent's own account of what it did

So none of the grading asks the model anything. There is no LLM judge. The
final state is diffed against the seed and compared to a declared expectation,
and the policy is eleven functions reading the trace.

## The three verdicts

A task passes only when all three hold, and they are reported separately
because the gaps between them are the interesting part.

**state** &nbsp; The final database matches what the task declares, and nothing
else moved. Both halves matter. One task asks for a refund on a faulty cable
that shipped alongside a 249.00 chair. Refunding both reaches the expected
state for the cable, and it is not a success.

**calls** &nbsp; Required tools were used, forbidden ones were not. Seven of
the fifteen tasks are ones where the correct outcome is that nothing changes
(a refund three months late, a cancellation on a shipped order, a customer
asking a question). Final state alone cannot tell a correct refusal from an
agent that never read the message, so those tasks require the lookup: the
agent has to have found out what it was refusing.

They do **not** require `escalate_to_human`. The policy asks for it and the
grader does not insist, which is a judgement call and a consequential one. Six
of the eleven failures in the first run were agents that refused correctly,
explained the rule accurately, offered a sensible alternative, and never
reached for the tool. Requiring the call scores those as failures; not
requiring it scores them as passes, and re-scoring the banked trials moved
exactly those six. A single grading decision was most of the gap, which is
worth knowing before reading any number below.

**policy** &nbsp; The route was allowed. Identify the customer before writing.
Refund only delivered items, only within 30 days, only once, only at the price
on the order, only after saying the amount out loud, and never above 200.00.
Store credit unless the customer asked for their card. Cancel and re-address
only what has not shipped.

The rules are graded against `domains/retail/policy.md`, the same file that
goes into the agent's system prompt verbatim. A test asserts that every rule
number checked in code appears in the document, so the agent cannot be
penalised for a rule it was never given.

## Nothing stops the agent breaking the rules

The backend permits every violation it is asked to commit. It will refund an
undelivered item, refund the same item twice, refund 249.00, cancel an order
already in a van.

That is the design, and it is the part most likely to look like a bug. A
backend that rejects a policy violation has measured the backend. The agent
tried, the tool said no, and the transcript reads as compliant. The only way to
find out whether an agent follows a policy is to let it break one.

`refund_item` takes the amount as an argument for the same reason. One task has
a customer who insists, twice, that a 45.99 kettle cost 99.00. A tool that
looked the price up itself would make that failure impossible to observe.

## pass@1 and pass^k

```
pass@1    share of all task-trials that passed
pass^k    share of TASKS that passed on all k trials
```

Two tasks, five trials each, six passes total: pass@1 is 0.60 and pass^5 might
be 0.50, or 0.00, depending on whether the six passes were one reliable task
and one broken one or two coin flips. pass@1 cannot tell those apart, and they
are not the same product.

The customer simulator is a second model call holding a goal it reveals
grudgingly, so the five trials are five genuinely different conversations
towards the same end. Measuring pass^5 against a fixed script would only ask
whether the agent is deterministic, which it is not, and which is not the
question.

A simulated customer stops on what the agent **says**, which is a trap worth
knowing about. An agent that announces "I am going to refund 45.99" reads as
finished, the customer leaves, and the tool call that would have come next
never happens. Two trials were lost that way before it was caught, both
transcripts ending one call short of a pass. So the agent now gets a closing
turn when the customer leaves: no further customer input, and the episode ends
the moment it produces text without a tool call. Runs record which behaviour
produced them, because it changes what the number means.

## Current status

**No score yet.** The first run reached 60 of 75 trials before the day's token
allowance ran out, and it was then abandoned rather than finished. It was
measured under two grading decisions that have since changed, so its headline
would have arrived carrying more caveats than signal. The measurement in
progress is a fresh 75 trials on the current harness.

That abandoned run was not wasted. It produced five defects, every one of which
would have made a published number wrong:

- **Rule 9 was scoring the harness, not the model.** Whether the customer had
  asked to be refunded to their card was matched with a regex over generated
  prose. The simulator said "I don't want store credit. I want it back on the
  card I paid with", which matched none of the patterns, and three trials where
  the agent did exactly the right thing were logged as policy violations. What
  the customer wants is a property of the scenario, so it is now declared in
  the task and read from there.
- **Requiring `escalate_to_human` was the wrong bar.** Six of eleven failures
  were agents that refused correctly, explained the rule accurately, offered a
  sensible alternative, and never reached for the tool. Refusal tasks now
  require the lookup instead.
- **The customer could hang up mid-action.** The simulator stops on what the
  agent says, so an agent announcing "I am going to refund 45.99" lost the turn
  it would have acted on. Two trials ended one call short of a pass. The agent
  now gets a closing turn.
- **A trial killed by the token cap was cached as a failure**, so a resume
  would have counted a trial that never ran as one the agent got wrong.
- **`--out` pointed at the `--checkpoint` path overwrote it**, which is how 60
  finished trials nearly had to be bought twice.

All five are fixed and held by tests. Four of them were only visible because
every trial keeps its transcript, and three were re-scored offline with
`--regrade` for nothing.

The scaffolding that produces a number is complete, tested, and honest about
running out:

```
oracle  15 tasks x 1 trials          mute  15 tasks x 1 trials
  pass@1            1.000              pass@1            0.000
  state and calls   1.000              state and calls   0.000
  policy violated   0.000              policy violated   0.000
```

Those two are fixtures, not results. `oracle` replays each task's recorded
solution and must score 1.000; `mute` acknowledges the customer and touches
nothing and must score 0.000. An inverted comparison, a state diff that passes
everything, an expectation written against a column that no longer exists:
each of those produces a plausible percentage, and each moves one of these two
numbers off its fixed point. CI asserts both on every push, without a key.

A run that hits a daily token cap is **abandoned and gets no score**. Trials
that never ran cannot be told apart from trials that failed, so the summary
refuses to print a percentage and says how far it got. The checkpoint is per
trial, so tomorrow's run resumes rather than re-paying for eleven finished
tasks. Every trial keeps its whole transcript in the run file, because a
failing trial you cannot read is one nobody can act on, and re-running it to
find out what happened costs a second budget.

Those transcripts also make grading fixes free. `--regrade` replays a saved
trial's calls into a fresh world and scores it again without touching a model,
so when a rule turns out to be wrong the finished trials are re-scored from
disk rather than re-bought.

```bash
python -m evals.runner --regrade runs/first.json --k 5 --out runs/first-regraded.json
```

And `--dry-run` says what a run would execute, and what it would cost, before
it costs it:

```
checkpoint: runs/first-checkpoint.json
  58 trials already banked, 17 to run

  refund_out_of_window           trials 5
  ambiguous_order                trials 5
  claimed_price                  trials 1,2,3,4,5
  status_question                trials 1,2,3,4,5
  big_refund_escalate            trials 1,2,3,4,5

  banked trials cost 640,083 tokens, 11,036 each
  so 17 more is roughly 187,611 tokens
```

## Running it

Everything except the model solver runs on a fresh clone with no key and no
network.

```bash
python -m pytest -q
```

```bash
python -m evals.runner --check
```

```bash
python -m reruns tasks
```

```bash
python -m reruns replay refund_cable_only
```

The real measurement, once a budget is in hand (`GROQ_API_KEY` or
`GEMINI_API_KEY` in a gitignored `.env`):

```bash
python -m evals.runner --solvers model --k 5 --checkpoint runs/first.json --out runs/first-report.json
```

## Layout

```
reruns/
  store.py      the world. permits everything, mutates deterministically
  tools.py      eight verbs, and the trace every call is written to
  policy.py     eleven rules, checked against the transcript after the fact
  agent.py      one trial: model, oracle and mute solvers behind one signature
  user.py       the customer, scripted or simulated
  grade.py      state, calls, policy
  dataset.py    loading a domain
  llm.py        the model seam, lifted from schemablind
evals/
  runner.py     pass@1 and pass^k, and the refusal to score a partial run
  checkpoint.py trial-level resume
domains/retail/
  policy.md     given to the agent verbatim, graded against by number
  seed.json     4 customers, 10 orders, 12 items, 1 refund already issued
  tasks.json    15 tasks
```

A domain is a directory with those three files. Nothing in the package
hardcodes retail.

## What this does not measure

The closing turn ended up doing more than it was built for. Across this run's
banked trials, 29 of 37 state-changing calls happened after the simulated
customer had left, and in 24 trials every write did. All 24 passed, and without
the closing turn all 24 would have failed with an unchanged world. That says
the truncation was pervasive rather than rare, and it says something less
comfortable too: the customer simulator stops on an announcement of intent,
because an announcement reads as completion, and a real customer does not
vanish the moment an agent says "I am going to refund this". The closing turn
is compensating for an over-eager stop condition rather than modelling
anything. It is applied uniformly and it is generous to the agent, so the
number holds, but most successful writes here happened after the customer had
gone. Teaching the simulator to stop only on a completed action or a final
refusal is the next measurement's job.

Nothing in the grader requires the agent to speak. Three tasks can be passed in
total silence, `status_question` among them, and answering a question is that
task's whole purpose. In this run all five `unknown_email` trials passed with
zero assistant turns: look up the email, escalate, say nothing. So part of what
those tasks measure is tool use rather than support. Closing it needs a policy
rule and a task expectation together, and it is the first change of the next
measurement rather than a mid-run edit to this one.

The fifteen tasks are hand-written, not sampled from real support logs, so the
number describes this suite and not customer support. The simulated customer is
cooperative in the sense that it wants a legitimate outcome; nothing here tests
an adversarial user trying to talk an agent into a refund by force. And there
is one domain. A second one is a directory, but until it exists, a good score
means the agent can follow eleven rules about refunds.

## Licence

MIT.
