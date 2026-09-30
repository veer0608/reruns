from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_voice_page", Path(__file__).resolve().parent.parent / "site" / "build_voice_page.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def kinds(said, heard):
    return [kind for kind, _, _ in page.classify(page.word_diff(said, heard)) if kind != "equal"]


@pytest.mark.parametrize("said, heard", [
    ("My email is nina.kapoor@example.com.", "My email is nina.kapoor at example.com"),
    ("I want the 99 dollars back.", "I want the $99 back."),
    ("22 Spice Market Rd, Chennai 600002.", "22 Spice Market Road, Chennai 6-0-0-0-0-2"),
    ("It is definitely o_1047.", "It is definitely O underscore 1047."),
    ("OK good, that's all.", "Okay, good. That's all."),
])
def test_a_spoken_form_of_the_same_thing_is_not_a_loss(said, heard):
    assert set(kinds(said, heard)) <= {"same"}
    assert not page.lossy(page.word_diff(said, heard))


@pytest.mark.parametrize("said, heard, kind", [
    ("My email is nina.kapoor@example.com.", "My email is neena.kapoor at example.com.", "swap"),
    ("Cancel it please. omar.haddad@example.com.", "Cancel it please.", "lost"),
    ("It is definitely o_1047.", "It is definitely O-1047.", "swap"),
    ("22 Spice Market Rd, Chennai 600002.", "22 Spice Market Road, Chinnai 600002.", "swap"),
    ("Chennai 600002.", "Chennai 6000002.", "swap"),
])
def test_a_change_that_loses_information_stays_marked(said, heard, kind):
    assert kind in kinds(said, heard)
    assert page.lossy(page.word_diff(said, heard))


def test_the_rendered_page_marks_the_two_differently():
    same = page.render_diff(page.word_diff("Email is nina.kapoor@example.com.", "Email is nina.kapoor at example.com"))
    wrong = page.render_diff(page.word_diff("Email is nina.kapoor@example.com.", "Email is neena.kapoor at example.com"))
    assert 'class="same"' in same and "<del>" not in same
    assert 'class="swap"' in wrong


def summary(kept, dropped, misheard):
    return {"mean_wer": 0.1, "verbatim": 0.5, "entities": {},
            "email_fates": {"kept": kept, "dropped": dropped, "misheard": misheard}}


def test_the_probe_chart_leaves_out_an_unfinished_setting():
    probe = {"seeds": 3, "configs": {
        "baseline": {"complete": True, "summary": summary(8, 7, 6)},
        "noise 5 dB": {"complete": False, "rows": []},
    }}
    html = page.render_probe(probe)
    assert "8 of 21 kept" in html
    assert "5 dB SNR" not in html
    assert "9 of 10 settings are not finished" in html


def test_the_probe_chart_groups_rows_by_what_changed():
    probe = {"seeds": 3, "configs": {key: {"complete": True, "summary": summary(1, 1, 1)}
                                     for _, key, _ in page.PROBE_GROUPS}}
    html = page.render_probe(probe)
    order = [html.index(group) for group in ("Reference", "Accent", "Noise", "Recogniser", "Trailing silence")]
    assert order == sorted(order)
    assert "not finished" not in html


def test_no_probe_means_no_chart():
    assert page.render_probe(None) == ""


def test_a_finding_appears_only_when_every_setting_it_names_is_complete():
    configs = {key: {"complete": True, "summary": summary(5, 3, 2)} for _, key, _ in page.PROBE_GROUPS}
    everything = page.probe_findings(configs)
    assert len(everything) == 4
    configs["pad 500 ms"] = {"complete": False, "rows": []}
    partial = page.probe_findings(configs)
    assert len(partial) == 3
    assert not any("Trailing silence" in f for f in partial)
