from __future__ import annotations

from evals import probe_line
from evals.compare import email_fate
from reruns.llm import QuotaExhausted
from reruns.voice import Line, LineConfig

LINES = [("t", 0, "Refund please. nina.kapoor@example.com."), ("t", 1, "Yes.")]


def line_that(heard_for):
    """A line whose recogniser maps each spoken text through `heard_for`."""
    def build(config):
        return Line(config=config, transcribe=lambda wav: (heard_for(wav.decode()), 1.0),
                    synth=lambda text, voice: text.encode(), phone=lambda a, s, seed, **k: (a, 1000.0))
    return build


def test_email_fate_names_the_three_things_a_line_can_do():
    said = "Refund please. nina.kapoor@example.com."
    assert email_fate(said, "Refund please. nina.kapoor at example.com") == "kept"
    assert email_fate(said, "Refund please.") == "dropped"
    assert email_fate(said, "Refund please. neena.kapoor at example.com") == "misheard"
    assert email_fate("Yes.", "Yes.") is None


def test_a_probe_counts_what_survived_per_config():
    result = probe_line.probe(line_that(lambda text: text.split(" nina")[0]),
                              {"baseline": LineConfig()}, LINES, seeds=2, quiet=True)
    entry = result["configs"]["baseline"]
    assert entry["complete"] and len(entry["rows"]) == 4
    assert entry["summary"]["email_fates"] == {"kept": 0, "dropped": 2, "misheard": 0}
    assert result["stopped"] is None


def test_the_daily_cap_leaves_the_config_without_numbers():
    calls = {"n": 0}

    def heard(text):
        calls["n"] += 1
        if calls["n"] > 3:
            raise QuotaExhausted("ASR HTTP 429: requests per day")
        return text

    result = probe_line.probe(line_that(heard), {"a": LineConfig(), "b": LineConfig(snr_db=10)},
                              LINES, seeds=2, quiet=True)
    assert result["configs"]["a"]["complete"] is False
    assert "b" not in result["configs"]
    assert "INCOMPLETE" in probe_line.report(result)
    assert "WER" in probe_line.report(result)


def test_the_grid_starts_from_the_line_the_voice_run_uses():
    assert probe_line.GRID["baseline"] == LineConfig()
    assert len({c.descriptor for c in probe_line.GRID.values()}) == len(probe_line.GRID)


def test_scripted_lines_take_two_per_task(domain):
    lines = probe_line.scripted_lines(domain)
    assert len(lines) == 2 * len(domain.tasks)
    assert lines[0][:2] == (domain.tasks[0].id, 0)
