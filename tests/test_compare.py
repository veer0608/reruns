from __future__ import annotations

import json

from evals.compare import Side, compare, lost_on_the_line, main
from reruns.grade import Verdict

LINE = "voice:en-IN-NeerjaNeural>g711-8000hz-quiet>whisper-large-v3-turbo"
LOST = [{"kind": "user", "text": "Refund please.", "spoken": "Refund please. nina.kapoor@example.com."}]
KEPT = [{"kind": "user", "text": "Refund please. nina.kapoor at example.com",
         "spoken": "Refund please. nina.kapoor@example.com."}]


def verdict(task, trial, passed, transcript=()):
    return Verdict(task_id=task, trial=trial, state_ok=passed, calls_ok=True,
                   transcript=list(transcript))


def side(verdicts, model="m", channel="text"):
    return Side(path="x", meta={"model": model, "channel": channel}, verdicts=verdicts)


def full(passes, transcript=()):
    """Two tasks, k=2, with the given pass pattern per task."""
    return [verdict(task, t + 1, ok, transcript)
            for task, pattern in passes.items() for t, ok in enumerate(pattern)]


def test_an_entity_the_recogniser_dropped_is_lost_and_one_put_back_is_not():
    assert lost_on_the_line(verdict("a", 1, False, LOST)) == ["email"]
    assert lost_on_the_line(verdict("a", 1, False, KEPT)) == []
    assert lost_on_the_line(verdict("a", 1, False, [{"kind": "user", "text": "hi"}])) == []


def test_two_models_are_refused():
    lines, code = compare([side([], "a", LINE), side([], "b")], ["t"], 1)
    assert code == 2 and "two models" in lines[0]


def test_two_text_runs_are_refused():
    lines, code = compare([side([]), side([])], ["t"], 1)
    assert code == 2


def test_an_incomplete_run_prints_no_score():
    voice = side(full({"a": [True, True]}), channel=LINE)
    text = side(full({"a": [True, True], "b": [True, True]}))
    lines, code = compare([voice, text], ["a", "b"], 2)
    out = "\n".join(lines)
    assert code == 3
    assert "NO SCORE" in out and "pass@1" not in out
    assert "a " in out and "\n  b " not in out


def test_failures_are_split_by_what_the_line_did():
    voice = side(
        [verdict("a", 1, False, LOST), verdict("a", 2, True, LOST),
         verdict("b", 1, False, KEPT), verdict("b", 2, True, KEPT)],
        channel=LINE,
    )
    text = side(full({"a": [True, True], "b": [True, True]}))
    lines, code = compare([voice, text], ["a", "b"], 2)
    out = "\n".join(lines)
    assert code == 0
    assert "pass@1                1.000    0.500" in out
    assert "exposed trial   1" in out
    assert "clean line       1" in out
    assert "lost an entity 2, and the agent still passed 1" in out


def test_main_reads_checkpoint_files(tmp_path, capsys):
    def write(name, meta, verdicts):
        path = tmp_path / name
        path.write_text(json.dumps({"meta": meta, "verdicts": [v.as_dict() for v in verdicts]}),
                        encoding="utf-8")
        return str(path)

    voice = write("v.json", {"model": "m", "channel": LINE}, [verdict("refund_kettle", 1, False, LOST)])
    text = write("t.json", {"model": "m"}, [verdict("refund_kettle", 1, True)])
    assert main([voice, text, "--k", "5"]) == 3
    assert "NO SCORE" in capsys.readouterr().out


AWARE = LINE + "+aware"
MISHEARD = [{"kind": "user", "text": "Refund please. neena.kapoor at example.com",
             "spoken": "Refund please. nina.kapoor@example.com."}]


def test_three_arms_in_any_order_get_one_column_each():
    text = side(full({"a": [True, True]}))
    voice = side([verdict("a", 1, False, MISHEARD), verdict("a", 2, False, MISHEARD)], channel=LINE)
    aware = side([verdict("a", 1, True, MISHEARD), verdict("a", 2, False, MISHEARD)], channel=AWARE)
    lines, code = compare([aware, text, voice], ["a"], 2)
    out = "\n".join(lines)
    assert code == 0
    assert "text    voice    aware" in out
    assert "1.000    0.000    0.500" in out
    assert "opening email misheard    0 passed of 2" in out
    assert "opening email misheard    1 passed of 2" in out


def test_two_runs_on_the_same_channel_are_refused():
    lines, code = compare([side([]), side([], channel=LINE), side([], channel=LINE)], ["t"], 1)
    assert code == 2 and "same channel" in lines[0]


def test_the_fate_table_leaves_out_the_task_whose_email_is_really_wrong():
    from evals.compare import fate_table
    table = fate_table([verdict("unknown_email", 1, True, MISHEARD), verdict("a", 1, False, MISHEARD)])
    assert table["misheard"] == [False]
