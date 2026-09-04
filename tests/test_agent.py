from __future__ import annotations

import pytest

from conftest import FakeClient, FakeReply

from reruns import agent
from reruns.llm import QuotaExhausted, ToolCall
from reruns.tools import Toolbox, Trace
from reruns.user import ScriptedUser


def call(name, **arguments):
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


def episode_for(domain, task_id, replies, max_turns=agent.MAX_TURNS, final_turn=True):
    task = domain.task(task_id)
    store = domain.store()
    trace = Trace()
    box = Toolbox(store, trace)
    user = ScriptedUser(lines=task.scripted_user)
    client = FakeClient(replies)
    episode = agent.run_model(task, domain.policy, box, user, client,
                              max_turns=max_turns, final_turn=final_turn)
    return episode, trace, store, client


def test_a_conversation_that_does_the_job(domain):
    episode, trace, store, _ = episode_for(domain, "refund_kettle", [
        FakeReply(tool_calls=[call("find_customer", email="nina.kapoor@example.com")]),
        FakeReply(tool_calls=[call("get_order", order_id="o_1041")]),
        FakeReply(text="That is 45.99, refunding to store credit.",
                  tool_calls=[call("refund_item", item_id="i_1", amount_cents=4599,
                                   method="store_credit")]),
        FakeReply(text="All done."),
    ])
    assert [c.name for c in trace.calls] == ["find_customer", "get_order", "refund_item"]
    assert episode.error is None
    assert store.snapshot()["order_items"]["i_1"]["status"] == "refunded"
    store.close()


def test_text_is_recorded_before_the_call_it_accompanies(domain):
    """Rule 7 reads the trace in order. A model that announces the amount in
    the same message as the tool call has announced it first."""
    _, trace, store, _ = episode_for(domain, "refund_kettle", [
        FakeReply(text="Refunding 45.99 now.",
                  tool_calls=[call("refund_item", item_id="i_1", amount_cents=4599,
                                   method="store_credit")]),
        FakeReply(text="Done."),
    ])
    kinds = [event["kind"] for event in trace.events]
    assert kinds.index("assistant") < kinds.index("call")
    store.close()


def test_escalating_ends_the_conversation(domain):
    """Otherwise an agent could escalate and then do the forbidden thing
    anyway, and the trace would show a compliant refusal."""
    _, trace, store, client = episode_for(domain, "big_refund_escalate", [
        FakeReply(tool_calls=[call("escalate_to_human", reason="over the limit")]),
        FakeReply(tool_calls=[call("refund_item", item_id="i_5", amount_cents=24900,
                                   method="store_credit")]),
    ])
    assert [c.name for c in trace.calls] == ["escalate_to_human"]
    assert len(client.calls) == 1
    assert store.snapshot()["refunds"].keys() == {"rf_1"}
    store.close()


def test_the_customer_gets_the_last_word_only_while_the_script_lasts(domain):
    _, trace, store, client = episode_for(domain, "status_question", [
        FakeReply(text="It shipped on the 25th."),
    ])
    heard = [e["text"] for e in trace.events if e["kind"] == "user"]
    assert heard == list(domain.task("status_question").scripted_user)
    store.close()


def test_a_looping_agent_is_stopped_by_max_turns(domain):
    _, trace, store, client = episode_for(
        domain, "refund_kettle",
        [FakeReply(tool_calls=[call("get_order", order_id="o_1041")])],
        max_turns=4,
    )
    assert len(client.calls) == 4
    assert len(trace.calls) == 4
    store.close()


def test_a_bad_tool_name_is_an_error_the_agent_can_read(domain):
    _, trace, store, _ = episode_for(domain, "refund_kettle", [
        FakeReply(tool_calls=[call("issue_apology", to="nina")]),
        FakeReply(text="Sorry about that."),
    ])
    assert trace.calls[0].ok is False
    assert "no such tool" in trace.calls[0].result
    store.close()


def test_the_daily_cap_ends_the_episode_rather_than_the_process(domain):
    class Capped:
        model = "capped"

        def chat(self, *, messages, tools, force=None):
            raise QuotaExhausted("429: tokens per day (TPD) exceeded")

    task = domain.task("refund_kettle")
    store = domain.store()
    box = Toolbox(store, Trace())
    episode = agent.run_model(task, domain.policy, box, ScriptedUser(task.scripted_user), Capped())
    store.close()
    assert episode.error.startswith("quota")


def test_a_transport_failure_is_reported_not_raised(domain):
    from reruns.llm import LLMError

    class Broken:
        model = "broken"

        def chat(self, *, messages, tools, force=None):
            raise LLMError("connection reset")

    task = domain.task("refund_kettle")
    store = domain.store()
    box = Toolbox(store, Trace())
    episode = agent.run_model(task, domain.policy, box, ScriptedUser(task.scripted_user), Broken())
    store.close()
    assert episode.error.startswith("llm:")


def test_the_policy_is_in_the_system_prompt(domain):
    _, _, store, client = episode_for(domain, "refund_kettle", [FakeReply(text="hi")])
    system = client.calls[0][0]["content"]
    assert "Refund only delivered items" in system
    assert "2026-03-01" in system
    store.close()


def test_the_model_solver_refuses_to_run_without_a_client(domain):
    store = domain.store()
    box = Toolbox(store, Trace())
    with pytest.raises(ValueError, match="needs an LLM client"):
        agent.run("model", domain.task("refund_kettle"), domain.policy, box,
                  ScriptedUser(()), None)
    store.close()


def test_an_unknown_solver_is_refused(domain):
    store = domain.store()
    box = Toolbox(store, Trace())
    with pytest.raises(ValueError, match="unknown solver"):
        agent.run("vibes", domain.task("refund_kettle"), domain.policy, box, ScriptedUser(()))
    store.close()


def test_without_a_key_the_customer_is_the_script(domain):
    user = agent.build_user(domain.task("refund_kettle"), None, scripted=False)
    assert isinstance(user, ScriptedUser)
    assert user.open() == domain.task("refund_kettle").scripted_user[0]


def _announce_then_act(domain, task_id, **kwargs):
    """The exact shape of the trials the truncation cost: the agent says what
    it is about to do, the customer reads that as done and leaves, and the tool
    call is still one turn away."""
    return episode_for(domain, task_id, [
        FakeReply(tool_calls=[call("find_customer", email="nina.kapoor@example.com")]),
        FakeReply(tool_calls=[call("get_order", order_id="o_1041")]),
        FakeReply(text="I am going to issue a refund of 45.99 as store credit."),
        FakeReply(tool_calls=[call("refund_item", item_id="i_1", amount_cents=4599,
                                   method="store_credit")]),
        FakeReply(text="Done."),
    ], **kwargs)


def test_the_agent_finishes_what_it_announced_after_the_customer_leaves(domain):
    episode, trace, store, _ = _announce_then_act(domain, "refund_kettle")
    assert "refund_item" in [c.name for c in trace.calls]
    assert store.snapshot()["order_items"]["i_1"]["status"] == "refunded"
    store.close()


def test_the_legacy_behaviour_still_loses_that_trial(domain):
    """Kept so a run measured before the fix can be extended honestly rather
    than mixed with one measured after it."""
    task = domain.task("refund_kettle")
    store = domain.store()
    trace = Trace()
    box = Toolbox(store, trace)
    agent.run_model(
        task, domain.policy, box, ScriptedUser(lines=task.scripted_user[:1]),
        FakeClient([
            FakeReply(tool_calls=[call("find_customer", email="nina.kapoor@example.com")]),
            FakeReply(text="I am going to issue a refund of 45.99 as store credit."),
            FakeReply(tool_calls=[call("refund_item", item_id="i_1", amount_cents=4599,
                                       method="store_credit")]),
        ]),
        final_turn=False,
    )
    assert "refund_item" not in [c.name for c in trace.calls]
    assert store.snapshot()["order_items"]["i_1"]["status"] == "delivered"
    store.close()


def test_the_closing_prompt_is_not_recorded_as_the_customer_speaking(domain):
    """It is a harness instruction. Putting it in the trace would make the
    customer appear to say something they never said, which several policy
    rules would then read."""
    _, trace, store, _ = _announce_then_act(domain, "refund_kettle")
    heard = [e["text"] for e in trace.events if e["kind"] == "user"]
    assert not any("left the chat" in text for text in heard)
    assert heard == list(domain.task("refund_kettle").scripted_user)
    store.close()


def test_the_customer_gets_no_further_say_once_they_have_left(domain):
    task = domain.task("refund_kettle")
    store = domain.store()
    trace = Trace()
    box = Toolbox(store, trace)
    user = ScriptedUser(lines=task.scripted_user[:1])
    agent.run_model(task, domain.policy, box, user, FakeClient([
        FakeReply(text="I am going to issue a refund of 45.99."),
        FakeReply(tool_calls=[call("refund_item", item_id="i_1", amount_cents=4599,
                                   method="store_credit")]),
        FakeReply(text="Done."),
    ]))
    assert user._at == 1, "the script must not advance after the customer has gone"
    store.close()


def test_the_closing_turn_does_not_run_forever(domain):
    """One extra turn to act, then it ends on the first message with no tool
    call. The customer leaving must not become an open-ended budget."""
    task = domain.task("refund_kettle")
    store = domain.store()
    box = Toolbox(store, Trace())
    client = FakeClient([
        FakeReply(text="I will sort that out."),
        FakeReply(text="Anything else?"),
    ])
    agent.run_model(task, domain.policy, box,
                    ScriptedUser(lines=task.scripted_user[:1]), client, max_turns=8)
    assert len(client.calls) == 2
    store.close()


def test_escalating_still_ends_it_with_no_closing_turn(domain):
    _, trace, store, client = episode_for(domain, "big_refund_escalate", [
        FakeReply(tool_calls=[call("escalate_to_human", reason="over the limit")]),
        FakeReply(tool_calls=[call("refund_item", item_id="i_5", amount_cents=24900,
                                   method="store_credit")]),
    ])
    assert [c.name for c in trace.calls] == ["escalate_to_human"]
    assert len(client.calls) == 1
    store.close()
