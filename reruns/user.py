"""The other half of the conversation.

Two simulators, and the choice between them is the difference between a free
deterministic harness check and a real measurement.

`ScriptedUser` replays a fixed list of lines. It costs nothing, never varies,
and is what CI runs -- but it also cannot be surprised, so an agent that asks a
question the script does not answer gets the next line regardless. It measures
the harness, not the agent.

`ModelUser` is a second model call holding a goal it will not state all at
once. It is what makes the task multi-turn in any interesting sense, and it is
also a second source of variance -- which is fine, and in fact the point:
pass^k over a varying user is the number worth having. A pass^5 measured
against a fixed script would only be asking whether the agent is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLMClient, Usage

STOP = "###STOP###"

USER_SYSTEM = """\
You are playing a customer talking to a support agent. Stay in character.

{goal}

Rules for how you play it:
- Write one short message at a time, the way a person types in a chat box.
- Do not volunteer everything at once. Answer what you are asked.
- You are a customer, not an assistant. Never offer to help, never look
  anything up, never mention order ids or internal details you would not know.
- **Want only what is written above.** Do not invent preferences, conditions or
  extra requests of your own. If the agent offers you a choice your brief does
  not cover, take whatever they propose.
- **Saying it is about to do something is not doing it.** "I am going to refund
  45.99" is a plan, not a result. Stay in the conversation and let them carry it
  out. Reply briefly, agree, or just say "ok".
- Reply with exactly {stop} and nothing else only once one of these is true:
  the agent says the thing is **done**; or it has clearly and finally refused
  and offered nothing further; or the conversation is going in circles.
"""
# The intent rule is not politeness either. Measured over v2's banked trials,
# 29 of 37 state-changing calls happened after the customer had already left,
# and in 24 trials every write did -- because the simulator read "I am going to
# refund 45.99" as completion and hung up. The agent's closing turn was then
# carrying the whole measurement instead of rescuing the rare case. A real
# customer does not vanish the moment an agent states an intention, and neither
# does this one now.
#
# That fourth rule is not politeness. Without it the simulator improvises: on
# the first live trial of `refund_kettle` it asked, unprompted, for the money
# back on its card. The agent obliged, correctly, and failed a task whose
# expected state says store credit -- a task failure caused entirely by the
# customer inventing a requirement. Across five trials that is noise landing
# squarely on the number the project exists to report, so the brief is the only
# thing the customer is allowed to want.


@dataclass
class ScriptedUser:
    """A fixed script. Runs out politely rather than looping."""

    lines: tuple[str, ...]
    _at: int = 0
    usage: list[Usage] = field(default_factory=list)

    def open(self) -> str:
        self._at = 1
        return self.lines[0] if self.lines else "Hello?"

    def reply(self, assistant_text: str) -> str | None:
        if self._at >= len(self.lines):
            return None
        line = self.lines[self._at]
        self._at += 1
        return None if line.strip() == STOP else line


@dataclass
class ModelUser:
    """A model given a goal and told to reveal it grudgingly."""

    client: LLMClient
    goal: str
    opening: str
    max_turns: int = 8
    _turns: int = 0
    _history: list[dict] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)

    def open(self) -> str:
        self._history = [
            {"role": "system", "content": USER_SYSTEM.format(goal=self.goal, stop=STOP)},
            # The opening line is scripted even here. Letting the model invent
            # its own first message would change the task between trials, and
            # then pass^k would be measuring five slightly different tasks.
            {"role": "assistant", "content": self.opening},
        ]
        return self.opening

    def reply(self, assistant_text: str) -> str | None:
        if self._turns >= self.max_turns:
            return None
        self._turns += 1
        # Roles are inverted from the agent's view: what the agent said is what
        # this model is replying to, so it arrives as a user message.
        self._history.append({"role": "user", "content": assistant_text or "(no reply)"})
        reply = self.client.chat(messages=self._history, tools=[])
        if reply.usage:
            self.usage.append(reply.usage)
        text = (reply.text or "").strip()
        self._history.append({"role": "assistant", "content": text})
        if not text or STOP in text:
            return None
        return text
