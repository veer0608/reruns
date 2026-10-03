from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evals import runner
from evals.checkpoint import Checkpoint, MixedHarness
from evals.compare import Side, arm_name, compare
from reruns.agent import HARNESS_VERSION
from reruns.grade import Verdict
from reruns.policy import RULE_NUMBERS
from reruns.voice import LineConfig, channel_name

RETAIL = Path(__file__).resolve().parent.parent / "domains" / "retail"


def rules(text: str) -> dict[int, str]:
    """Rule number to its full text, continuation lines included."""
    out, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\.\s*(.*)", line)
        if m:
            current = int(m.group(1))
            out[current] = m.group(2)
        elif current is not None and line.startswith("   "):
            out[current] += " " + line.strip()
        else:
            current = None
    return out


def test_the_voice_policy_changes_rule_2_and_nothing_else():
    """Its text goes into the prompt verbatim. A stray edit anywhere else would
    make the policy arm measure two changes and report one."""
    base = rules((RETAIL / "policy.md").read_text(encoding="utf-8"))
    voice = rules((RETAIL / "policy-voice.md").read_text(encoding="utf-8"))
    assert set(base) == set(voice) and RULE_NUMBERS <= set(voice)
    assert [n for n in base if base[n] != voice[n]] == [2]
    assert "spell" in voice[2] and "Escalate" in voice[2]
    base_lines = [l for l in (RETAIL / "policy.md").read_text(encoding="utf-8").splitlines() if not re.match(r"^(\s*2\.|   )", l)]
    voice_lines = [l for l in (RETAIL / "policy-voice.md").read_text(encoding="utf-8").splitlines() if not re.match(r"^(\s*2\.|   )", l)]
    assert base_lines == voice_lines


def test_the_default_policy_leaves_every_channel_name_as_it_was():
    assert channel_name(None) == "text"
    assert channel_name(LineConfig()) == "voice:en-IN-NeerjaNeural>g711-8000hz-quiet>whisper-large-v3-turbo"
    assert channel_name(LineConfig(), policy="policy-voice").endswith("+policy:policy-voice")


def test_a_policy_missing_a_graded_rule_is_refused(tmp_path, capsys):
    thin = tmp_path / "thin.md"
    thin.write_text("1. Identify the customer.\n2. Escalate.\n", encoding="utf-8")
    assert runner.main(["--solvers", "oracle", "--policy", str(thin)]) == 2
    assert "missing rules" in capsys.readouterr().out


def test_a_policy_that_does_not_exist_is_refused(tmp_path, capsys):
    assert runner.main(["--solvers", "oracle", "--policy", str(tmp_path / "nope.md")]) == 2


def test_the_policy_arm_cannot_resume_the_plain_voice_checkpoint(tmp_path):
    path = tmp_path / "voice.json"
    path.write_text(json.dumps({"meta": {"harness": HARNESS_VERSION, "channel": channel_name(LineConfig())},
                                "verdicts": []}), encoding="utf-8")
    with pytest.raises(MixedHarness):
        Checkpoint.load(path, channel_name(LineConfig(), policy="policy-voice"))


def test_the_policy_arm_dry_runs_on_its_own_checkpoint(tmp_path, capsys):
    code = runner.main(["--dry-run", "--voice", "--k", "1", "--policy", str(RETAIL / "policy-voice.md"),
                        "--checkpoint", str(tmp_path / "voice-policy.json")])
    assert code == 0
    assert "0 trials already banked" in capsys.readouterr().out


def side(channel, passed=True):
    return Side(path="x", meta={"model": "m", "channel": channel},
                verdicts=[Verdict(task_id="a", trial=1, state_ok=passed, calls_ok=True)])


def test_four_arms_get_four_columns_in_a_fixed_order():
    line = channel_name(LineConfig())
    arms = [side(channel_name(LineConfig(), policy="policy-voice")), side("text"),
            side(channel_name(LineConfig(), voice_aware=True)), side(line, passed=False)]
    assert [arm_name(s) for s in arms] == ["policy", "text", "aware", "voice"]
    lines, code = compare(arms, ["a"], 1)
    assert code == 0
    assert "text    voice    aware   policy" in "\n".join(lines)
