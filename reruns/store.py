"""The world the agent changes, and the snapshot the grader reads.

Two things here are load-bearing and easy to get wrong.

**The store enforces nothing.** It will happily refund an order that was never
delivered, refund the same item twice, and cancel something already in a van.
That is deliberate. A backend that rejects a policy violation has measured the
backend, not the agent -- the agent is scored on what it *tried*, so the try has
to be allowed to land. The only errors here are structural: an id that does not
exist, a number that is not a number.

**Every mutation is deterministic.** Refund ids come from a counter, timestamps
come from the task's fixed `now`, never the wall clock. A grader that diffs
final state cannot tolerate a field that changes between two identical runs --
and pass^k compares five identical runs, so a stray `datetime.now()` would show
up as an agent that is inconsistent when the harness is.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    customer_id           TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    email                 TEXT NOT NULL UNIQUE,
    tier                  TEXT NOT NULL,
    credit_balance_cents  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE products (
    product_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    price_cents  INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id     TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    status       TEXT NOT NULL,
    placed_at    TEXT NOT NULL,
    address      TEXT NOT NULL,
    total_cents  INTEGER NOT NULL
);
CREATE TABLE order_items (
    item_id      TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL REFERENCES orders(order_id),
    product_id   TEXT NOT NULL REFERENCES products(product_id),
    qty          INTEGER NOT NULL,
    price_cents  INTEGER NOT NULL,
    status       TEXT NOT NULL
);
CREATE TABLE refunds (
    refund_id     TEXT PRIMARY KEY,
    order_id      TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    amount_cents  INTEGER NOT NULL,
    method        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""

TABLES = ("customers", "products", "orders", "order_items", "refunds")
KEYS = {
    "customers": "customer_id",
    "products": "product_id",
    "orders": "order_id",
    "order_items": "item_id",
    "refunds": "refund_id",
}

#: Snapshot type: {table: {primary key: {column: value}}}.
Snapshot = dict[str, dict[str, dict]]


class StoreError(ValueError):
    """A structurally impossible request -- an unknown id, a bad number.

    Never a policy refusal. If this class starts carrying "not allowed", the
    benchmark has stopped measuring the agent.
    """


@dataclass(frozen=True)
class Seed:
    now: str
    rows: dict[str, list[dict]]

    @classmethod
    def load(cls, path: str | Path) -> "Seed":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(now=raw["now"], rows={t: raw.get(t, []) for t in TABLES})


class Store:
    """An in-memory retail backend, seeded fresh for every single trial."""

    def __init__(self, seed: Seed) -> None:
        self.now = seed.now
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        for table in TABLES:
            for row in seed.rows.get(table, []):
                columns = ", ".join(row)
                marks = ", ".join("?" for _ in row)
                self._conn.execute(
                    f"INSERT INTO {table} ({columns}) VALUES ({marks})",
                    tuple(row.values()),
                )
        self._conn.commit()
        self._refund_seq = len(seed.rows.get("refunds", []))

    # -- reads ---------------------------------------------------------------

    def _one(self, table: str, key: str) -> dict:
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE {KEYS[table]} = ?", (key,)
        ).fetchone()
        if row is None:
            raise StoreError(f"no such {table[:-1]}: {key}")
        return dict(row)

    def customer_by_email(self, email: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM customers WHERE lower(email) = lower(?)", (email.strip(),)
        ).fetchone()
        if row is None:
            raise StoreError(f"no customer with email {email}")
        return dict(row)

    def order(self, order_id: str) -> dict:
        order = self._one("orders", order_id)
        rows = self._conn.execute(
            "SELECT i.*, p.name AS product_name FROM order_items i "
            "JOIN products p ON p.product_id = i.product_id "
            "WHERE i.order_id = ? ORDER BY i.item_id",
            (order_id,),
        ).fetchall()
        order["items"] = [dict(r) for r in rows]
        return order

    def orders_of(self, customer_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT order_id, status, placed_at, total_cents FROM orders "
            "WHERE customer_id = ? ORDER BY placed_at DESC, order_id DESC",
            (customer_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def item(self, item_id: str) -> dict:
        return self._one("order_items", item_id)

    def refunds_for(self, item_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM refunds WHERE item_id = ? ORDER BY refund_id", (item_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- writes --------------------------------------------------------------

    def set_order_status(self, order_id: str, status: str) -> dict:
        self._one("orders", order_id)
        self._conn.execute(
            "UPDATE orders SET status = ? WHERE order_id = ?", (status, order_id)
        )
        self._conn.commit()
        return self._one("orders", order_id)

    def set_address(self, order_id: str, address: str) -> dict:
        self._one("orders", order_id)
        self._conn.execute(
            "UPDATE orders SET address = ? WHERE order_id = ?",
            (address.strip(), order_id),
        )
        self._conn.commit()
        return self._one("orders", order_id)

    def record_refund(self, item_id: str, amount_cents, method: str) -> dict:
        item = self._one("order_items", item_id)
        try:
            amount = int(amount_cents)
        except (TypeError, ValueError) as exc:
            raise StoreError(f"amount_cents is not a number: {amount_cents!r}") from exc
        self._refund_seq += 1
        refund_id = f"rf_{self._refund_seq}"
        self._conn.execute(
            "INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?)",
            (refund_id, item["order_id"], item_id, amount, method, self.now),
        )
        self._conn.execute(
            "UPDATE order_items SET status = 'refunded' WHERE item_id = ?", (item_id,)
        )
        if method == "store_credit":
            self._conn.execute(
                "UPDATE customers SET credit_balance_cents = credit_balance_cents + ? "
                "WHERE customer_id = (SELECT customer_id FROM orders WHERE order_id = ?)",
                (amount, item["order_id"]),
            )
        self._conn.commit()
        return self._one("refunds", refund_id)

    # -- grading surface -----------------------------------------------------

    def snapshot(self) -> Snapshot:
        out: Snapshot = {}
        for table in TABLES:
            key = KEYS[table]
            rows = self._conn.execute(f"SELECT * FROM {table} ORDER BY {key}").fetchall()
            out[table] = {row[key]: dict(row) for row in rows}
        return out

    def close(self) -> None:
        self._conn.close()


def diff(before: Snapshot, after: Snapshot) -> dict:
    """Every cell that moved, as {table: {pk: {column: [old, new]}}}.

    A deleted row shows as `{column: [old, None]}` across all its columns, and
    an inserted one is the mirror of that. Nothing here special-cases a table,
    so a column added to the schema is diffed without touching this function.
    """
    out: dict = {}
    for table in sorted(set(before) | set(after)):
        rows_before = before.get(table, {})
        rows_after = after.get(table, {})
        table_delta: dict = {}
        for pk in sorted(set(rows_before) | set(rows_after)):
            old = rows_before.get(pk)
            new = rows_after.get(pk)
            if old == new:
                continue
            columns = sorted(set(old or {}) | set(new or {}))
            table_delta[pk] = {
                column: [(old or {}).get(column), (new or {}).get(column)]
                for column in columns
                if (old or {}).get(column) != (new or {}).get(column)
            }
        if table_delta:
            out[table] = table_delta
    return out
