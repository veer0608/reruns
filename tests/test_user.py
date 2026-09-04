from __future__ import annotations

from conftest import FakeReply

from reruns.user import STOP, ModelUser, ScriptedUser


class Replies:
    """A user-simulator client: no tools, one text reply per call."""

    model = "fake"

    def __init__(self, texts):
        self.texts = list(texts)
        self.seen = []

    def chat(self, *, messages, tools, force=None):
        self.seen.append(messages[-1]["content"])
        assert tools == [], "the customer does not get tools"
        return FakeReply(text=self.texts.pop(0) if self.texts else STOP)


def test_the_script_runs_out_instead_of_looping():
    user = ScriptedUser(lines=("hello", "yes please"))
    assert user.open() == "hello"
    assert user.reply("how can I help?") == "yes please"
    assert user.reply("done") is None


def test_an_empty_script_still_opens():
    assert ScriptedUser(lines=()).open() == "Hello?"


def test_the_stop_token_ends_a_script_early():
    user = ScriptedUser(lines=("hello", STOP, "never reached"))
    user.open()
    assert user.reply("hi") is None


def test_the_opening_line_is_scripted_even_for_the_model_customer():
    """Letting the simulator invent its own first message would change the task
    between trials, and pass^k would be comparing five different tasks."""
    client = Replies(["and it was dented"])
    user = ModelUser(client=client, goal="get a refund", opening="I want a refund")
    assert user.open() == "I want a refund"
    assert client.seen == []


def test_the_agent_message_arrives_as_a_user_turn():
    client = Replies(["the copper kettle"])
    user = ModelUser(client=client, goal="get a refund", opening="hello")
    user.open()
    assert user.reply("which item do you mean?") == "the copper kettle"
    assert client.seen == ["which item do you mean?"]


def test_the_customer_can_end_the_conversation():
    client = Replies([f"thanks, that's sorted {STOP}"])
    user = ModelUser(client=client, goal="get a refund", opening="hello")
    user.open()
    assert user.reply("all done") is None


def test_an_empty_reply_ends_it_rather_than_hanging():
    client = Replies(["   "])
    user = ModelUser(client=client, goal="get a refund", opening="hello")
    user.open()
    assert user.reply("anything else?") is None


def test_a_customer_who_will_not_stop_is_stopped():
    client = Replies(["no" for _ in range(50)])
    user = ModelUser(client=client, goal="argue forever", opening="hello", max_turns=3)
    user.open()
    assert [user.reply("?") for _ in range(4)] == ["no", "no", "no", None]


def test_the_goal_is_in_the_customers_system_prompt():
    client = Replies(["ok"])
    user = ModelUser(client=client, goal="You are Nina and the kettle is dented.",
                     opening="hello")
    user.open()
    user.reply("hi")
    assert "the kettle is dented" in user._history[0]["content"]
    assert STOP in user._history[0]["content"]


def test_the_customer_is_told_not_to_invent_requirements():
    """A regression guard on a bug that cost a live trial.

    Without this instruction the simulator asked, unprompted, to be refunded
    to its card. The agent obliged and failed a task whose expected state says
    store credit. That is harness noise landing directly on pass^k.
    """
    from reruns.user import USER_SYSTEM

    filled = USER_SYSTEM.format(goal="g", stop=STOP)
    assert "Do not invent preferences" in filled
