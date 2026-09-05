from __future__ import annotations

import json
import shutil

import pytest

from reruns.dataset import Domain, Task, validate


def test_the_shipped_domain_is_valid(domain):
    assert validate(domain) == []


def test_the_suite_is_split_between_acting_and_leaving_alone(domain):
    """Twelve tasks where the world must move and eight where it must not.

    The split is deliberate and worth keeping near even. A suite weighted
    towards acting rewards an eager agent; one weighted towards refusing
    rewards a timid one, and both would report a number that says more about
    the task mix than about the model.
    """
    assert len(domain.tasks) == 20
    still = [task for task in domain.tasks if task.read_only]
    assert len(still) == 8, [t.id for t in still]


def test_the_hard_five_are_present_and_shaped_right(domain):
    """Nine of the first twelve measured tasks never failed a trial across two
    runs and two harnesses, so the suite was mostly measuring nothing. These
    five target the one failure mode those runs actually produced -- acting
    before the requirements are in -- and the shapes around it.
    """
    hard = {
        "mixed_refund_split": "refund what you can, escalate what you cannot",
        "changes_mind_midway": "the request is withdrawn before it is carried out",
        "wrong_order_confidently": "the named order belongs to somebody else",
        "late_correction": "the customer switches item mid-flow",
        "refusal_then_allowed": "a no on one request is not a no on the next",
    }
    for task_id in hard:
        task = domain.task(task_id)
        assert task.scripted_user and task.solution, task_id
        assert task.must_call, task_id


def test_every_refusal_task_requires_a_call(domain):
    """A task with no expected state change and nothing it must call would
    pass on silence, and would quietly lift every score in the table."""
    for task in domain.tasks:
        if task.read_only:
            assert task.must_call, task.id


def test_selecting_tasks_by_id(domain):
    picked = domain.select("refund_kettle,status_question")
    assert [task.id for task in picked] == ["refund_kettle", "status_question"]
    assert domain.select(None) == domain.tasks
    with pytest.raises(KeyError):
        domain.select("no_such_task")


def broken(tmp_path, domain, mutate):
    root = tmp_path / "retail"
    shutil.copytree(domain.root, root)
    tasks = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    mutate(tasks)
    (root / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    return validate(Domain.load(root))


def test_a_task_naming_a_tool_that_does_not_exist(tmp_path, domain):
    problems = broken(tmp_path, domain, lambda t: t[0]["must_call"].append("issue_apology"))
    assert any("unknown tool" in p for p in problems)


def test_a_task_expecting_a_column_that_does_not_exist(tmp_path, domain):
    def mutate(tasks):
        tasks[0]["expect"]["customers"]["c_1"] = {"loyalty_points": 5}

    assert any("unknown column" in p for p in broken(tmp_path, domain, mutate))


def test_a_task_expecting_a_table_that_does_not_exist(tmp_path, domain):
    def mutate(tasks):
        tasks[0]["expect"]["vouchers"] = {"v_1": {"amount": 1}}

    assert any("unknown table" in p for p in broken(tmp_path, domain, mutate))


def test_two_tasks_with_the_same_id(tmp_path, domain):
    problems = broken(tmp_path, domain, lambda t: t.append(dict(t[0])))
    assert any("duplicate task id" in p for p in problems)


def test_a_tool_both_required_and_forbidden(tmp_path, domain):
    problems = broken(tmp_path, domain, lambda t: t[0]["forbid_call"].append("refund_item"))
    assert any("both required and forbidden" in p for p in problems)


def test_a_task_with_no_script_cannot_run_free(tmp_path, domain):
    problems = broken(tmp_path, domain, lambda t: t[0].update(scripted_user=[]))
    assert any("cannot run without a key" in p for p in problems)


def test_a_solution_calling_a_tool_that_does_not_exist(tmp_path, domain):
    def mutate(tasks):
        tasks[0]["solution"].append({"call": {"name": "apologise", "arguments": {}}})

    assert any("unknown tool" in p for p in broken(tmp_path, domain, mutate))


def test_read_only_is_about_expectations_not_intentions():
    acting = Task.from_dict({"id": "a", "goal": "g", "expect": {"orders": {"o": {"status": "x"}}}})
    refusing = Task.from_dict({"id": "b", "goal": "g"})
    assert not acting.read_only
    assert refusing.read_only


def test_refund_tasks_say_how_the_customer_wants_their_money_back(domain):
    """The expected state names a refund method, so the brief has to as well.

    A customer left free to choose will sometimes ask for their card, the agent
    will correctly oblige, and the task will fail for a reason that has nothing
    to do with the agent. That happened on the first live trial.
    """
    for task in domain.tasks:
        methods = {row.get("method") for row in task.expect.get("refunds", {}).values()}
        if not methods:
            continue
        assert "store credit" in task.goal or "original payment" in task.goal or \
               "card" in task.goal, task.id
