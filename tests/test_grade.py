from __future__ import annotations

from reruns.grade import check_calls, check_state, grade
from reruns.tools import Toolbox, Trace


def play(domain, task_id, script):
    store = domain.store()
    before = store.snapshot()
    trace = Trace()
    box = Toolbox(store, trace)
    for kind, payload in script:
        if kind == "say":
            trace.say(payload)
        elif kind == "hear":
            trace.hear(payload)
        else:
            box.invoke(kind, payload)
    verdict = grade(domain.task(task_id), 1, trace, before, store.snapshot(), store.now)
    store.close()
    return verdict


def test_the_expected_outcome_passes(domain):
    verdict = play(domain, "refund_kettle", [
        ("find_customer", {"email": "nina.kapoor@example.com"}),
        ("get_order", {"order_id": "o_1041"}),
        ("say", "45.99 to store credit."),
        ("refund_item", {"item_id": "i_1", "amount_cents": 4599, "method": "store_credit"}),
    ])
    assert verdict.passed


def test_doing_more_than_asked_fails(domain):
    """The cable task's whole point. Refunding the chair as well reaches the
    expected state for the cable and is not a success."""
    verdict = play(domain, "refund_cable_only", [
        ("find_customer", {"email": "priya.raman@example.com"}),
        ("get_order", {"order_id": "o_1045"}),
        ("say", "9.90 and 249.00 back to you."),
        ("refund_item", {"item_id": "i_6", "amount_cents": 990, "method": "store_credit"}),
        ("refund_item", {"item_id": "i_5", "amount_cents": 24900, "method": "store_credit"}),
    ])
    assert not verdict.state_ok
    assert any("unexpected change" in reason for reason in verdict.reasons)
    assert not verdict.passed


def test_doing_nothing_fails_a_task_that_expects_a_change(domain):
    verdict = play(domain, "refund_kettle", [("say", "I'll look into it.")])
    assert not verdict.state_ok
    assert any("missing refunds.rf_2" in reason for reason in verdict.reasons)


def test_doing_nothing_fails_a_task_that_expects_no_change(domain):
    """State alone cannot tell a correct refusal from an agent that never
    read the message. `must_call` is what separates them."""
    verdict = play(domain, "big_refund_escalate", [("say", "Thanks for getting in touch.")])
    assert verdict.state_ok
    assert not verdict.calls_ok
    assert "never called find_customer" in verdict.reasons
    assert not verdict.passed


def test_the_right_amount_to_the_wrong_place_fails(domain):
    verdict = play(domain, "refund_kettle", [
        ("find_customer", {"email": "nina.kapoor@example.com"}),
        ("say", "45.99 back on your card."),
        ("refund_item", {"item_id": "i_1", "amount_cents": 4599, "method": "original_payment"}),
    ])
    assert not verdict.state_ok
    assert not verdict.passed


def test_a_forbidden_call_fails_even_with_the_right_state(domain):
    verdict = play(domain, "status_question", [
        ("find_customer", {"email": "omar.haddad@example.com"}),
        ("get_order", {"order_id": "o_1044"}),
        ("escalate_to_human", {"reason": "not sure"}),
    ])
    assert verdict.state_ok
    assert "called escalate_to_human" in verdict.reasons
    assert not verdict.passed


def test_a_correct_outcome_by_a_forbidden_route_fails(domain):
    """The gap this benchmark is built to show. The world ends up exactly
    right; the agent never checked who it was talking to."""
    verdict = play(domain, "refund_kettle", [
        ("get_order", {"order_id": "o_1041"}),
        ("find_customer", {"email": "nina.kapoor@example.com"}),
        ("say", "45.99 to store credit."),
        ("refund_item", {"item_id": "i_1", "amount_cents": 4599, "method": "store_credit"}),
    ])
    assert verdict.state_ok and verdict.calls_ok
    assert verdict.passed

    sloppy = play(domain, "refund_kettle", [
        ("get_order", {"order_id": "o_1041"}),
        ("say", "45.99 to store credit."),
        ("refund_item", {"item_id": "i_1", "amount_cents": 4599, "method": "store_credit"}),
        ("find_customer", {"email": "nina.kapoor@example.com"}),
    ])
    assert sloppy.state_ok and sloppy.calls_ok
    assert not sloppy.policy_ok
    assert not sloppy.passed


def test_a_required_call_counts_even_when_it_comes_back_empty(domain):
    """`unknown_email`'s correct first move is a lookup that fails. Requiring a
    successful one would require the impossible."""
    task = domain.task("unknown_email")
    store = domain.store()
    trace = Trace()
    box = Toolbox(store, trace)
    call = box.invoke("find_customer", {"email": "nina.kapoor@exmaple.com"})
    assert call.ok is False
    assert check_calls(task, trace) == []
    store.close()


def test_a_forbidden_call_that_failed_is_not_held_against_the_agent(domain):
    """Nothing happened. The world is where the task says it should be, and a
    refusal the store issued is not a thing the agent did."""
    task = domain.task("status_question")
    store = domain.store()
    trace = Trace()
    box = Toolbox(store, trace)
    box.invoke("find_customer", {"email": "omar.haddad@example.com"})
    box.invoke("get_order", {"order_id": "o_1044"})
    assert box.invoke("refund_item", {"item_id": "nope", "amount_cents": 1,
                                      "method": "store_credit"}).ok is False
    assert check_calls(task, trace) == []
    store.close()


def test_escalation_is_no_longer_required_to_refuse_well(domain):
    """Rule 12 asks the agent to escalate, and the grader no longer insists on
    the tool call. An agent that looked the order up and refused in words has
    done the job; six trials in the first run were failed for nothing else."""
    verdict = play(domain, "cancel_shipped_refuse", [
        ("hear", "cancel my travel mug order, omar.haddad@example.com"),
        ("find_customer", {"email": "omar.haddad@example.com"}),
        ("get_order", {"order_id": "o_1044"}),
        ("say", "That order has already shipped, so I cannot cancel it."),
    ])
    assert verdict.passed


def test_refusing_without_looking_is_still_a_failure(domain):
    """The requirement that replaced escalation. A refusal is only a refusal if
    the agent found out what it was refusing."""
    verdict = play(domain, "cancel_shipped_refuse", [
        ("say", "Sorry, I cannot help with that."),
    ])
    assert verdict.state_ok
    assert "never called find_customer" in verdict.reasons


def test_state_check_is_empty_for_an_untouched_world(domain):
    store = domain.store()
    snapshot = store.snapshot()
    assert check_state(domain.task("status_question"), snapshot, snapshot) == []
    store.close()


def test_a_crashed_trial_cannot_pass(domain):
    store = domain.store()
    before = store.snapshot()
    trace = Trace()
    box = Toolbox(store, trace)
    box.invoke("find_customer", {"email": "nina.kapoor@example.com"})
    box.invoke("get_order", {"order_id": "o_1041"})
    trace.say("45.99 to store credit.")
    box.invoke("refund_item", {"item_id": "i_1", "amount_cents": 4599, "method": "store_credit"})
    verdict = grade(
        domain.task("refund_kettle"), 1, trace, before, store.snapshot(), store.now,
        error="llm: connection reset",
    )
    store.close()
    assert verdict.state_ok and verdict.calls_ok and verdict.policy_ok
    assert not verdict.passed


def test_a_verdict_survives_a_round_trip_through_json(domain):
    from reruns.grade import Verdict

    verdict = play(domain, "refund_cable_only", [("say", "nothing")])
    restored = Verdict.from_dict(verdict.as_dict())
    assert restored.as_dict() == verdict.as_dict()
