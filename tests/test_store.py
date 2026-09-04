from __future__ import annotations

import pytest

from reruns.store import Store, StoreError, diff


def test_seed_loads_every_table(store):
    snapshot = store.snapshot()
    assert len(snapshot["customers"]) == 4
    assert len(snapshot["orders"]) == 10
    assert len(snapshot["order_items"]) == 12
    assert len(snapshot["refunds"]) == 1


def test_two_stores_from_one_seed_are_identical(domain):
    first, second = domain.store(), domain.store()
    assert first.snapshot() == second.snapshot()
    first.close()
    second.close()


def test_a_trial_cannot_leak_into_the_next(domain):
    """The single most dangerous bug this benchmark could have.

    pass^k runs the same task five times. If a mutation from trial 1 survived
    into trial 2, trials 2-5 would score against a world the agent never saw,
    and the resulting number would look like inconsistency in the agent.
    """
    first = domain.store()
    first.set_order_status("o_1042", "cancelled")
    first.record_refund("i_1", 4599, "store_credit")
    first.close()

    second = domain.store()
    assert second.snapshot()["orders"]["o_1042"]["status"] == "pending"
    assert "rf_2" not in second.snapshot()["refunds"]
    second.close()


def test_refund_ids_are_a_counter_not_a_clock(store):
    first = store.record_refund("i_1", 4599, "store_credit")
    second = store.record_refund("i_3", 2400, "store_credit")
    assert first["refund_id"] == "rf_2"
    assert second["refund_id"] == "rf_3"


def test_refund_timestamps_come_from_the_task_not_the_wall_clock(store):
    refund = store.record_refund("i_1", 4599, "store_credit")
    assert refund["created_at"] == store.now == "2026-03-01T12:00:00Z"


def test_store_credit_moves_the_balance_and_card_refunds_do_not(store):
    store.record_refund("i_1", 4599, "store_credit")
    assert store.snapshot()["customers"]["c_1"]["credit_balance_cents"] == 2400 + 4599

    store.record_refund("i_9", 4599, "original_payment")
    assert store.snapshot()["customers"]["c_4"]["credit_balance_cents"] == 1200


def test_the_store_permits_what_the_policy_forbids(store):
    """If this test ever fails, the benchmark has stopped measuring the agent.

    A backend that refuses a policy violation makes the violation unobservable
    -- the agent tried, the tool said no, and the trace looks compliant.
    """
    store.record_refund("i_4", 1850, "store_credit")           # never delivered
    store.record_refund("i_11", 2400, "store_credit")          # already refunded
    store.record_refund("i_5", 24900, "store_credit")          # over the limit
    store.set_order_status("o_1044", "cancelled")              # already shipped
    assert len(store.snapshot()["refunds"]) == 4


def test_structural_errors_are_still_errors(store):
    with pytest.raises(StoreError):
        store.order("o_nope")
    with pytest.raises(StoreError):
        store.customer_by_email("nobody@example.com")
    with pytest.raises(StoreError):
        store.record_refund("i_1", "forty-six dollars", "store_credit")


def test_email_lookup_ignores_case_and_padding(store):
    assert store.customer_by_email("  NINA.KAPOOR@example.com ")["customer_id"] == "c_1"


def test_diff_reports_only_what_moved(store):
    before = store.snapshot()
    store.set_address("o_1050", "22 Spice Market Rd, Chennai 600002")
    delta = diff(before, store.snapshot())
    assert delta == {
        "orders": {
            "o_1050": {
                "address": [
                    "4 Harbour Rd, Chennai 600001",
                    "22 Spice Market Rd, Chennai 600002",
                ]
            }
        }
    }


def test_diff_of_an_insert_names_every_column(store):
    before = store.snapshot()
    store.record_refund("i_1", 4599, "store_credit")
    delta = diff(before, store.snapshot())
    assert set(delta["refunds"]["rf_2"]) == {
        "refund_id", "order_id", "item_id", "amount_cents", "method", "created_at",
    }
    assert delta["refunds"]["rf_2"]["amount_cents"] == [None, 4599]


def test_an_unchanged_world_diffs_to_nothing(store):
    assert diff(store.snapshot(), store.snapshot()) == {}
