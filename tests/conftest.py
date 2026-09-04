from __future__ import annotations

import pytest

from reruns.dataset import Domain
from reruns.tools import Toolbox, Trace


@pytest.fixture(scope="session")
def domain() -> Domain:
    return Domain.load()


@pytest.fixture
def store(domain):
    live = domain.store()
    yield live
    live.close()


@pytest.fixture
def toolbox(store):
    return Toolbox(store, Trace())


class FakeReply:
    """Enough of `llm.Reply` for the agent loop, without a network."""

    def __init__(self, text="", tool_calls=(), usage=None):
        self.text = text
        self.tool_calls = tuple(tool_calls)
        self.finish_reason = "stop"
        self.message = {"role": "assistant", "content": text}
        self.usage = usage

    @property
    def wants_tools(self):
        return bool(self.tool_calls)


class FakeClient:
    """Replays a fixed list of replies, then repeats the last one forever.

    Repeating rather than raising is deliberate: an agent loop with a broken
    exit condition should show up as a test that hits max_turns, not as an
    IndexError that looks like a bug in the fake.
    """

    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, *, messages, tools, force=None):
        self.calls.append(messages)
        if not self.replies:
            return FakeReply(text="Anything else?")
        reply = self.replies[0]
        if len(self.replies) > 1:
            self.replies.pop(0)
        return reply
