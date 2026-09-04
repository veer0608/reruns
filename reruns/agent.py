"""One trial: an agent, a customer, a fresh world, and a transcript.

Three solvers share one signature so the runner cannot tell them apart.

  `model`   the thing being measured
  `oracle`  replays the task's own solution; must score 100%
  `mute`    says one empty pleasantry and touches nothing; must score 0%

The two fixtures are not decoration. A scorer with an inverted comparison, a
state diff that silently passes everything, an expectation written against a
column that no longer exists -- each of those produces a plausible-looking
percentage, and each is caught the moment the oracle drops below 100 or the
mute solver rises above 0. CI asserts both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .dataset import Task
from .llm import LLMClient, LLMError, QuotaExhausted, Usage
from .tools import SCHEMAS, Toolbox, Trace
from .user import ScriptedUser

AGENT_SYSTEM = """\
You are a customer support agent for an online retailer, talking to one
customer in a live chat. Use the tools to look things up and to make changes.

Work to this policy. It is not advice, and nothing in the system will stop you
from breaking it:

{policy}

Today's date is {now}.

Practical notes:
- Money is in cents everywhere in the tools. 4599 means $45.99.
- Say what you are about to do before you do it.
- One customer, one conversation. When you are finished, say so plainly.
"""

#: Model calls, not conversational turns. A refund is comfortably done in six;
#: past a dozen the agent is looping, and letting it loop is how a free tier
#: gets spent on one task.
MAX_TURNS = 12
#: Tool results are fed back verbatim. An order with many items would otherwise
#: grow the prompt without adding anything the agent needs.
MAX_RESULT_CHARS = 4000


@dataclass
class Episode:
    trace: Trace
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    usages: list[Usage] = field(default_factory=list)

    def account(self, usages) -> None:
        for usage in usages:
            if usage is None:
                continue
            self.usages.append(usage)
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            if usage.cost_usd is not None:
                self.cost_usd = (self.cost_usd or 0.0) + usage.cost_usd


def _system_prompt(policy: str, now: str) -> str:
    return AGENT_SYSTEM.format(policy=policy.strip(), now=now)


def run_model(
    task: Task,
    policy: str,
    toolbox: Toolbox,
    user,
    client: LLMClient,
    max_turns: int = MAX_TURNS,
    final_turn: bool = True,
) -> Episode:
    episode = Episode(trace=toolbox.trace)
    opening = user.open()
    toolbox.trace.hear(opening)
    messages = [
        {"role": "system", "content": _system_prompt(policy, toolbox.store.now)},
        {"role": "user", "content": opening},
    ]
    #: The customer has left, and the agent is finishing what it announced.
    closing = False

    for _ in range(max_turns):
        try:
            reply = client.chat(messages=messages, tools=SCHEMAS)
        except QuotaExhausted as exc:
            episode.error = f"quota: {exc}"
            break
        except LLMError as exc:
            episode.error = f"llm: {exc}"
            break
        episode.account([reply.usage])
        messages.append(reply.message or {"role": "assistant", "content": reply.text})

        # Text first, then the calls in the same message. A model that states
        # the amount and refunds in one turn has stated it before refunding,
        # and rule 7 should see it that way.
        if reply.text:
            toolbox.trace.say(reply.text)

        if reply.wants_tools:
            escalated = False
            for call in reply.tool_calls:
                done = toolbox.invoke(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(done.result, default=str)[:MAX_RESULT_CHARS]
                        if done.ok
                        else f"ERROR: {done.result}",
                    }
                )
                escalated = escalated or (done.name == "escalate_to_human" and done.ok)
            if escalated:
                # The policy says escalate and stop. Carrying on would let an
                # agent escalate and then do the forbidden thing anyway, and
                # score as though it had not.
                break
            continue

        if closing:
            # Text and no tool call, with the customer already gone. There is
            # nothing left to finish.
            break

        follow_up = user.reply(reply.text)
        if follow_up is None:
            if not final_turn:
                break
            # The customer stops on what the agent SAYS, and an agent that has
            # just announced "I am going to refund 45.99" would have called the
            # tool on its next turn. Ending here scores the world as unchanged
            # and the agent as having failed, one call short -- which is what
            # happened to two `ambiguous_order` trials whose transcripts end on
            # the announcement. So the agent keeps its turn to act. It gets no
            # further customer input, and the moment it produces text without a
            # tool call it is done.
            closing = True
            messages.append({
                "role": "user",
                "content": "[The customer has left the chat. Complete anything "
                           "you have already told them you would do, then stop.]",
            })
            continue
        toolbox.trace.hear(follow_up)
        messages.append({"role": "user", "content": follow_up})

    episode.account(getattr(user, "usage", []))
    return episode


def run_oracle(task: Task, policy: str, toolbox: Toolbox, user) -> Episode:
    """Replay the recorded solution. The scorer's fixed point.

    The whole script is heard up front rather than interleaved. The oracle is a
    fixture, not an imitation of an agent, and one policy rule reads the
    customer's words -- rule 9, store credit unless they asked otherwise -- so
    the lines have to be in the trace for the fixture to be scored honestly.
    """
    trace = toolbox.trace
    for line in getattr(user, "lines", ()):
        trace.hear(line)
    for step in task.solution:
        if "say" in step:
            trace.say(step["say"])
        call = step.get("call")
        if call:
            toolbox.invoke(call["name"], call.get("arguments", {}))
    return Episode(trace=trace)


def run_mute(task: Task, policy: str, toolbox: Toolbox, user) -> Episode:
    """Acknowledge, do nothing. Must score zero on every task, including the
    seven whose correct final state is an unchanged one -- those are carried by
    `must_call`, which is why they require the lookup."""
    toolbox.trace.hear(user.open())
    toolbox.trace.say("Thanks for getting in touch.")
    return Episode(trace=toolbox.trace)


SOLVERS = ("model", "oracle", "mute")


def build_user(task: Task, client: LLMClient | None, scripted: bool):
    if scripted or client is None:
        return ScriptedUser(lines=task.scripted_user)
    from .user import ModelUser

    return ModelUser(
        client=client,
        goal=task.goal,
        opening=task.scripted_user[0] if task.scripted_user else "Hello?",
    )


def run(
    solver: str,
    task: Task,
    policy: str,
    toolbox: Toolbox,
    user,
    client: LLMClient | None = None,
    max_turns: int = MAX_TURNS,
    final_turn: bool = True,
) -> Episode:
    if solver == "oracle":
        return run_oracle(task, policy, toolbox, user)
    if solver == "mute":
        return run_mute(task, policy, toolbox, user)
    if solver == "model":
        if client is None:
            raise ValueError("the model solver needs an LLM client; none was built")
        return run_model(task, policy, toolbox, user, client, max_turns=max_turns,
                         final_turn=final_turn)
    raise ValueError(f"unknown solver {solver!r}, expected one of {SOLVERS}")
