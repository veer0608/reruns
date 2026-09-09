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

**v3: 100 trials, 20 tasks, harness 3, `gemini-flash-lite-latest`.**

```
pass@1     0.78
pass^5     0.70   (14 of 20 tasks passed all 5 trials)
```

State-and-calls (final state plus required/forbidden tools, policy set aside)
also lands at 0.78. Every trial that violated a policy rule in this run also
failed on state or calls, so there is no trial here that did right by the
customer's data while still breaking a rule. That is a property of these three
violations, not a guarantee the two numbers will keep matching in a future run.

Escalation is asked for by policy rule 12 and not required by the grader; a
considered refusal in words scores as a pass here, same as it has since the
first run.

Policy violations, the shipped reading of rule 7:

```
3 of 100 trials violated a rule
  rule 7  amount_stated_first   2
  rule 1  own_orders_only       2
```

(one trial broke both.) The strict reading of rule 7, the customer has to see
the amount in a message that does not also carry the tool call, costs nothing
to check against the same transcripts:

```bash
python -m evals.runner --regrade runs/v3.json --k 5 --strict-rule-7
```

```
9 of 100 trials violated a rule, strict
  rule 7  amount_seen_before_final   6
  rule 7  amount_stated_first        2
  rule 1  own_orders_only            2
```

Unlike v2, where the strict reading turned three passing trials into failures
and named the exact mechanism, every trial it adds here (three
`refund_original_payment`, three `late_correction`) was already failing on
state. It confirms the mechanism rather than uncovering a hidden gap this
time. The headline above uses the shipped reading; the strict verdicts are not
written back into `runs/v3.json`.

Trials passed out of 5, by task:

```
address_change_pending          5/5
address_change_shipped_refuse   5/5
ambiguous_order                 5/5
big_refund_escalate             5/5
cancel_and_refund                5/5
cancel_pending_lamp              5/5
cancel_shipped_refuse            5/5
double_refund_refuse             5/5
refund_cable_only                5/5
refund_kettle                    5/5
refund_out_of_window             5/5
refusal_then_allowed             5/5
status_question                  5/5
unknown_email                    5/5
wrong_order_confidently          2/5
refund_original_payment          2/5
late_correction                  2/5
claimed_price                    1/5
mixed_refund_split               1/5
changes_mind_midway              0/5
```

One known hole this run does not close: nothing in the grader requires the
agent to speak, and `status_question`, `unknown_email` and
`cancel_shipped_refuse` can all be passed in total silence. Their transcripts
were checked before trusting the 5/5s above; the lookups and required tool
calls are genuinely present, not empty runs coasting on a lenient grader.

Fourteen tasks never lost a trial. The six that did are the whole story:

- **changes_mind_midway (0/5).** The agent cancels the pending order on the
  customer's first, one-line message, then correctly refuses to un-cancel it
  and correctly refunds the item raised in the follow-up. It gets the
  follow-up right in every trial and fails anyway, because `cancel_order` is
  irreversible and nothing in the task, the policy, or the grading asks the
  agent to wait a beat before acting on a terse first message. There is no
  hesitation to reward here yet; the agent is fast, and fast is wrong on this
  one.
- **claimed_price (1/5).** The customer insists the $45.99 kettle cost $99.00.
  The task wants the recorded price held and a $45.99 refund issued. In 4 of 5
  trials the agent escalates the price dispute to a human instead and never
  calls `refund_item`. Treating a contradicted customer as grounds to escalate,
  rather than a fact to restate and act on, is the failure.
- **mixed_refund_split (1/5).** One item is refundable outright ($9.90), the
  other is over the escalation threshold ($249.00). In 4 of 5 trials the agent
  escalates the whole request instead of refunding the cheap item and
  escalating only the expensive one; `refund_item` is never called.
- **late_correction (2/5).** The customer names the wrong item, the agent
  refunds it, then the customer corrects themselves mid-flow. In 3 of 5 trials
  the agent refunds the corrected item on top of the first one instead of
  treating the correction as a replacement, paying out twice.
- **refund_original_payment (2/5).** The case documented in CLAUDE.md: the
  agent defaults to store credit unasked, and when the customer objects after
  the money has moved, it escalates instead of correcting the refund method
  itself.
- **wrong_order_confidently (2/5).** The customer states an order id
  confidently; it belongs to someone else. In 3 of 5 trials the agent skips
  `list_orders` and refunds the item anyway, which is also this run's only
  rule 1 (`own_orders_only`) violation.

Tokens: 1,303,711 prompt + 26,373 completion = 1,330,084 across 100 trials,
13,301 per trial on average. No pricing is configured for
`gemini-flash-lite-latest`, so nothing here is priced.

v1 (60 of 75 trials) and v2 (57 of 75) were both abandoned rather than
finished; they ran on earlier harnesses, and their five harness defects and
three grading gaps are what v3's harness and grading are built to have already
fixed. Their files and the full account of what they found stay in `runs/` and
`CLAUDE.md`.

The scaffolding that produces a number is checked on every push, without a
key:

```
oracle  20 tasks x 1 trials          mute  20 tasks x 1 trials
  pass@1            1.000              pass@1            0.000
  state and calls   1.000              state and calls   0.000
  policy violated   0.000              policy violated   0.000
```

`oracle` replays each task's recorded solution and must score 1.000; `mute`
acknowledges the customer and touches nothing and must score 0.000. Either
number moving off its fixed point means the scorer is broken, independent of
what any model does.

A run that hits a daily token cap is abandoned and gets no score: trials that
never ran cannot be told apart from trials that failed, so the summary refuses
to print a percentage and says how far it got instead. `--dry-run` shows the
same thing before spending anything:

```bash
python -m evals.runner --dry-run --k 5 --checkpoint runs/v3.json
```

Grading fixes are free against banked transcripts, which is how the strict
rule 7 reading above was checked without touching a model:

```bash
python -m evals.runner --regrade runs/v3.json --k 5 --out runs/v3-regraded.json
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

The closing turn ended up doing more than it was built for, and fixing that is
what the current run is measuring. Across this run's
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
gone. The simulator now stops only on a completed action or a
final refusal, never on a stated intention, and the run reported below is the
first measured that way. Runs carry a harness version and the checkpoint
refuses to mix them, because trials measured either side of a change like this
are two measurements wearing one number.

Nothing in the grader requires the agent to speak. Three tasks can be passed in
total silence, `status_question` among them, and answering a question is that
task's whole purpose. In this run all five `unknown_email` trials passed with
zero assistant turns: look up the email, escalate, say nothing. So part of what
those tasks measure is tool use rather than support. Closing it needs a policy
rule and a task expectation together, and it is the first change of the next
measurement rather than a mid-run edit to this one.

The twenty tasks are hand-written, not sampled from real support logs, so the
number describes this suite and not customer support. Five of them were added
after two runs showed that nine of the twelve measured tasks had never failed a
single trial, 88 trials without one, which meant the suite was mostly
confirming that the easy cases are easy. The five aim at the one failure mode
those runs did produce: the agent acting before the customer has finished
saying what they want. The simulated customer is
cooperative in the sense that it wants a legitimate outcome; nothing here tests
an adversarial user trying to talk an agent into a refund by force. And there
is one domain. A second one is a directory, but until it exists, a good score
means the agent can follow eleven rules about refunds.

## Licence

MIT.
