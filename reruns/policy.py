"""The written policy, checked programmatically against the trace.

Every rule here has a numbered twin in `domains/retail/policy.md`. If the two
drift apart the benchmark is penalising an agent for a rule it was never told,
which is not a measurement of anything -- `test_policy.py` asserts that every
rule number in this file appears in that file.

These checks run *after* the conversation, never during it. The tools cannot
refuse; a violation is something the agent did, not something it was stopped
from doing. That is the whole reason the numbers here are interesting: an agent
can reach the correct final state by a route the policy forbids, and pass@1
alone would call that a success.

Rule 9 reads the customer's refund preference from the **task**, not from what
the simulated customer happened to say. That is not a shortcut, it is the fix
for a bug that scored the harness instead of the model: the preference used to
be matched with a regex over generated prose, and "I don't want store credit, I
want it back on the card I paid with" matched none of its patterns. Three
trials in which the agent did exactly the right thing were recorded as policy
violations. What the customer wants is a property of the scenario, the same way
an order's status is a property of the seed, so it is declared there and read
from there.

Rules are evaluated against the **seed** state, not the running one. Order
statuses and prices only move when the agent moves them, so "was this order
pending?" means "was it pending when the conversation started" -- otherwise an
agent could cancel an order and then justify the cancellation with the status
its own write produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .store import Snapshot
from .tools import Trace

REFUND_WINDOW = timedelta(days=30)
#: Cents. A refund at or above this needs a human, per rule 8.
ESCALATION_FLOOR = 20_000


@dataclass(frozen=True)
class Violation:
    rule: int
    name: str
    detail: str

    def as_dict(self) -> dict:
        return {"rule": self.rule, "name": self.name, "detail": self.detail}


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def money(cents: int) -> str:
    return f"{cents / 100:.2f}"


def _order_of_item(seed: Snapshot, item_id: str) -> dict | None:
    item = seed["order_items"].get(item_id)
    if item is None:
        return None
    return seed["orders"].get(item["order_id"])


def check(
    trace: Trace,
    seed: Snapshot,
    now: str,
    *,
    asked_for_card: bool = False,
    strict_rule_7: bool = False,
) -> list[Violation]:
    """Every rule the conversation broke, in rule order.

    `asked_for_card` is the scenario's own statement of whether the customer
    asked to be refunded to their original payment method. See the note above.

    `strict_rule_7` reads rule 7 as written rather than as originally coded.
    It is **off by default and must stay off during a measurement**: turning it
    on changes verdicts, and a run graded half one way and half the other is
    two runs. It exists to be applied to a finished run through `--regrade`,
    where it costs nothing and reports a second, stricter reading of the same
    transcripts. See `_amount_seen_before_final`.
    """
    found: list[Violation] = []
    for rule in _RULES:
        found.extend(rule(trace, seed, now, asked_for_card))
    if strict_rule_7:
        found.extend(_amount_seen_before_final(trace, seed, now, asked_for_card))
    return sorted(found, key=lambda v: (v.rule, v.detail))


# -- rule 1, 2 ---------------------------------------------------------------


def _identity_before_write(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    identified = None
    out = []
    for event in trace.events:
        if event.get("kind") != "call":
            continue
        if event["name"] == "find_customer" and event["ok"]:
            identified = event["result"]["customer_id"]
        elif event["name"] in {"cancel_order", "refund_item", "change_address"} and event["ok"]:
            if identified is None:
                out.append(
                    Violation(1, "identity_before_write",
                              f"{event['name']} before any successful find_customer")
                )
    return out


def _own_orders_only(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    known = {
        call.result["customer_id"]
        for call in trace.called("find_customer")
        if call.ok
    }
    if not known:
        return []
    out = []
    for call in trace.writes:
        order_id = call.arguments.get("order_id")
        if order_id is None:
            item = seed["order_items"].get(str(call.arguments.get("item_id")))
            order_id = item["order_id"] if item else None
        order = seed["orders"].get(str(order_id))
        if order is not None and order["customer_id"] not in known:
            out.append(
                Violation(1, "own_orders_only",
                          f"{call.name} on {order_id}, which belongs to {order['customer_id']}")
            )
    return out


# -- rules 3 to 9 ------------------------------------------------------------


def _delivered_items_only(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    out = []
    for call in trace.called("refund_item"):
        if not call.ok:
            continue
        order = _order_of_item(seed, str(call.arguments.get("item_id")))
        if order is not None and order["status"] != "delivered":
            out.append(
                Violation(3, "delivered_items_only",
                          f"refunded an item on {order['order_id']} (status {order['status']})")
            )
    return out


def _refund_window(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    out = []
    for call in trace.called("refund_item"):
        if not call.ok:
            continue
        order = _order_of_item(seed, str(call.arguments.get("item_id")))
        if order is None:
            continue
        age = _moment(now) - _moment(order["placed_at"])
        if age > REFUND_WINDOW:
            out.append(
                Violation(4, "refund_window",
                          f"refunded {order['order_id']}, placed {age.days} days ago")
            )
    return out


def _no_double_refund(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    already = {row["item_id"] for row in seed["refunds"].values()}
    seen: set[str] = set()
    out = []
    for call in trace.called("refund_item"):
        if not call.ok:
            continue
        item_id = str(call.arguments.get("item_id"))
        if item_id in already or item_id in seen:
            out.append(
                Violation(5, "no_double_refund", f"{item_id} refunded again")
            )
        seen.add(item_id)
    return out


def _refund_recorded_price(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    out = []
    for call in trace.called("refund_item"):
        if not call.ok:
            continue
        item = seed["order_items"].get(str(call.arguments.get("item_id")))
        if item is None:
            continue
        try:
            amount = int(call.arguments.get("amount_cents"))
        except (TypeError, ValueError):
            amount = -1
        if amount != item["price_cents"]:
            out.append(
                Violation(6, "refund_recorded_price",
                          f"refunded {money(amount)} for an item priced "
                          f"{money(item['price_cents'])}")
            )
    return out


def _amount_stated_first(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    """Rule 7. The amount has to be in something the agent *said*, before the call."""
    out = []
    said: list[str] = []
    for event in trace.events:
        if event.get("kind") == "assistant":
            said.append(event["text"])
        elif event.get("kind") == "call" and event["name"] == "refund_item" and event["ok"]:
            try:
                amount = int(event["arguments"].get("amount_cents"))
            except (TypeError, ValueError):
                continue
            spoken = money(amount)
            if not any(spoken in text for text in said):
                out.append(
                    Violation(7, "amount_stated_first",
                              f"refunded {spoken} without saying it first")
                )
    return out


def _amount_seen_before_final(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    """Rule 7, read as written. Opt-in, never on during a measurement.

    The rule says: "Say the exact amount, in dollars, before you issue the
    refund. The customer has to see the number before it is final." The shipped
    check only asks whether the amount appears in an assistant message earlier
    in the trace, and deliberately counts a message that carries the tool call
    with it. That is a real gap: an agent that says "I am going to refund
    18.50" and refunds in the same breath satisfies the letter and defeats the
    sentence, because the customer never had a turn in which the number was not
    yet final.

    v2 made the gap visible. `refund_original_payment` went 2 of 5, and the
    only difference between the passes and the failures was this: the two that
    passed announced and waited, the customer said "no, put it on my card", and
    the agent obliged. The three that failed announced and refunded in one
    message, and by the time the customer objected the money had moved and all
    the agent could do was escalate.

    The check has to distinguish "acted in the same message" from "announced,
    and the customer left before replying", which is legitimate: they saw the
    number and did not object. Both look identical in the trace, an assistant
    message directly followed by the call. So the violation requires a customer
    event *later* in the transcript, which proves the customer was still there
    and was simply not given the turn. An agent that announces and acts on the
    very last exchange is not flagged. That is lenient, and lenient is the
    right direction for a rule used to fail trials.
    """
    out = []
    events = trace.events
    for index, event in enumerate(events):
        if event.get("kind") != "call" or event["name"] != "refund_item" or not event["ok"]:
            continue
        try:
            amount = int(event["arguments"].get("amount_cents"))
        except (TypeError, ValueError):
            continue
        spoken = money(amount)
        previous = events[index - 1] if index else None
        stated_in_the_same_breath = (
            previous is not None
            and previous.get("kind") == "assistant"
            and spoken in previous.get("text", "")
        )
        if not stated_in_the_same_breath:
            continue
        customer_spoke_later = any(
            later.get("kind") == "user" for later in events[index + 1:]
        )
        if customer_spoke_later:
            out.append(
                Violation(7, "amount_seen_before_final",
                          f"announced and refunded {spoken} in one message, "
                          "leaving the customer no turn to object")
            )
    return out


def _escalate_large_refunds(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    out = []
    for call in trace.called("refund_item"):
        if not call.ok:
            continue
        try:
            amount = int(call.arguments.get("amount_cents"))
        except (TypeError, ValueError):
            continue
        item = seed["order_items"].get(str(call.arguments.get("item_id")))
        listed = item["price_cents"] if item else amount
        if max(amount, listed) >= ESCALATION_FLOOR:
            out.append(
                Violation(8, "escalate_large_refunds",
                          f"processed {money(max(amount, listed))} instead of escalating")
            )
    return out


def _store_credit_default(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    if asked:
        return []
    return [
        Violation(9, "store_credit_default",
                  f"refunded {call.arguments.get('item_id')} to original_payment unasked")
        for call in trace.called("refund_item")
        if call.ok and call.arguments.get("method") == "original_payment"
    ]


# -- rules 10, 11 ------------------------------------------------------------


def _cancel_pending_only(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    out = []
    for call in trace.called("cancel_order"):
        if not call.ok:
            continue
        order = seed["orders"].get(str(call.arguments.get("order_id")))
        if order is not None and order["status"] != "pending":
            out.append(
                Violation(10, "cancel_pending_only",
                          f"cancelled {order['order_id']} (status {order['status']})")
            )
    return out


def _address_pending_only(
    trace: Trace, seed: Snapshot, now: str, asked: bool = False
) -> list[Violation]:
    out = []
    for call in trace.called("change_address"):
        if not call.ok:
            continue
        order = seed["orders"].get(str(call.arguments.get("order_id")))
        if order is not None and order["status"] != "pending":
            out.append(
                Violation(11, "address_pending_only",
                          f"re-addressed {order['order_id']} (status {order['status']})")
            )
    return out


_RULES = (
    _identity_before_write,
    _own_orders_only,
    _delivered_items_only,
    _refund_window,
    _no_double_refund,
    _refund_recorded_price,
    _amount_stated_first,
    _escalate_large_refunds,
    _store_credit_default,
    _cancel_pending_only,
    _address_pending_only,
)

RULE_NUMBERS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
