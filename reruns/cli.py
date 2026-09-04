"""Looking at the domain without running an eval.

    python -m reruns tasks              what the 15 tasks are
    python -m reruns show refund_kettle what one of them expects
    python -m reruns replay refund_kettle   the oracle transcript, graded
    python -m reruns policy             the policy the agent is given
"""

from __future__ import annotations

import argparse
import json
import sys

from .dataset import DEFAULT_DOMAIN, Domain, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reruns", description=__doc__)
    parser.add_argument("--domain", default=str(DEFAULT_DOMAIN))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks", help="list the tasks")
    sub.add_parser("policy", help="print the policy the agent is given")
    sub.add_parser("validate", help="check the task file")
    show = sub.add_parser("show", help="one task in full")
    show.add_argument("task_id")
    replay = sub.add_parser("replay", help="run the oracle and print the transcript")
    replay.add_argument("task_id")
    args = parser.parse_args(argv)

    domain = Domain.load(args.domain)

    if args.command == "policy":
        print(domain.policy)
        return 0

    if args.command == "tasks":
        for task in domain.tasks:
            shape = "no change expected" if task.read_only else "changes state"
            print(f"{task.id:<28} {shape:<20} {task.title}")
        print(f"\n{len(domain.tasks)} tasks in domain {domain.name}")
        return 0

    if args.command == "validate":
        problems = validate(domain)
        for problem in problems:
            print(problem)
        print("ok" if not problems else f"{len(problems)} problems")
        return 0 if not problems else 1

    if args.command == "show":
        task = domain.task(args.task_id)
        print(f"{task.id} -- {task.title}\n")
        print(f"goal given to the customer:\n  {task.goal}\n")
        print("scripted customer:")
        for line in task.scripted_user:
            print(f"  > {line}")
        print(f"\nmust call:   {', '.join(task.must_call) or '-'}")
        print(f"must not:    {', '.join(task.forbid_call) or '-'}")
        print("\nexpected final state:")
        print("  " + (json.dumps(task.expect, indent=2).replace("\n", "\n  ")
                      if task.expect else "unchanged"))
        return 0

    if args.command == "replay":
        return _replay(domain, args.task_id)

    return 1


def _replay(domain: Domain, task_id: str) -> int:
    from evals.runner import run_trial

    task = domain.task(task_id)
    verdict = run_trial(domain, task, 1, "oracle")
    # Re-run for the transcript. The graded trial above ran in its own world,
    # and reaching into it for the trace would tie the CLI to the runner's
    # internals for the sake of printing.
    from .tools import Toolbox, Trace
    from . import agent as agent_module
    from .user import ScriptedUser

    store = domain.store()
    trace = Trace()
    agent_module.run_oracle(task, domain.policy, Toolbox(store, trace),
                            ScriptedUser(lines=task.scripted_user))
    store.close()

    print(f"{task.id} -- {task.title}\n")
    for event in trace.events:
        if event["kind"] == "user":
            print(f"  customer  {event['text']}")
        elif event["kind"] == "assistant":
            print(f"  agent     {event['text']}")
        else:
            arguments = ", ".join(f"{k}={v!r}" for k, v in event["arguments"].items())
            status = "" if event["ok"] else "  ERROR"
            print(f"            {event['name']}({arguments}){status}")
    print(f"\n  state {verdict.state_ok}  calls {verdict.calls_ok}  "
          f"policy {verdict.policy_ok}  ->  {'pass' if verdict.passed else 'FAIL'}")
    for reason in verdict.reasons:
        print(f"    {reason}")
    for violation in verdict.violations:
        print(f"    rule {violation.rule} {violation.name}: {violation.detail}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
