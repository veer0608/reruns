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
agent that never read the message, so those tasks carry a required call.

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

## Current status

**No score yet.** The first full run against `gemini-flash-lite-latest`
reached 60 of 75 trials before the day's token allowance ran out, so 12 of the
15 tasks have all five trials and three have none. By this project's own rule
that is not a score and none is printed here. The remaining trials resume from
the checkpoint.

What the partial run did produce is three bugs in the harness, all of which
would have made the eventual number wrong:

- **Rule 9 was scoring the harness, not the model.** Whether the customer had
  asked to be refunded to their card was matched with a regex over generated
  prose. The simulator said "I don't want store credit. I want it back on the
  card I paid with", which matched none of the patterns, and three trials in
  which the agent did exactly the right thing were recorded as policy
  violations. What the customer wants is a property of the scenario, so it is
  now declared in the task and read from there.
- **A trial killed by the token cap was cached as a failure**, so the resume
  would have counted a trial that never ran as one the agent got wrong. That
  is the abandonment rule defeated from inside the checkpoint.
- **`--out` pointed at the `--checkpoint` path overwrote it**, which is how the
  first 60 trials nearly had to be bought twice.

All three are fixed and held by tests. The 60 finished trials were recovered
and re-scored offline with `--regrade`, which moved exactly the three trials
rule 9 had wrongly failed and nothing else.

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

The fifteen tasks are hand-written, not sampled from real support logs, so the
number describes this suite and not customer support. The simulated customer is
cooperative in the sense that it wants a legitimate outcome; nothing here tests
an adversarial user trying to talk an agent into a refund by force. And there
is one domain. A second one is a directory, but until it exists, a good score
means the agent can follow eleven rules about refunds.

## Licence

MIT.
