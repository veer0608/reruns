from __future__ import annotations

import array
import io
import json
import math
import shutil
import wave

import pytest

from conftest import FakeClient, FakeReply

from evals import runner
from evals.checkpoint import Checkpoint, MixedHarness
from reruns import agent
from reruns.agent import HARNESS_VERSION
from reruns.grade import check_state
from reruns.llm import QuotaExhausted
from reruns.tools import Toolbox, Trace
from reruns.user import ScriptedUser
from reruns.voice import (
    INAUDIBLE, Line, LineConfig, VoiceUser, channel_name, channel_stats, degrade,
    entities_kept, seed_for, telephone, to_wav, wer,
)


class FakeEar:
    """A recogniser that applies fixed substitutions and counts its calls."""

    def __init__(self, swaps=None, drop_after=None):
        self.swaps = swaps or {}
        self.drop_after = drop_after
        self.calls = 0

    def __call__(self, wav: bytes) -> tuple[str, float]:
        self.calls += 1
        text = wav.decode()
        if self.drop_after and self.drop_after in text:
            text = text.split(self.drop_after)[0] + self.drop_after
        for old, new in self.swaps.items():
            text = text.replace(old, new)
        return text, 12.0


def fake_line(ear, cache_dir=None, config=None):
    """The text rides through the fake stages as bytes, so no audio is made."""
    return Line(
        config=config or LineConfig(),
        transcribe=ear,
        synth=lambda text, voice: text.encode(),
        phone=lambda audio, snr, seed: (audio, 1000.0),
        cache_dir=cache_dir,
    )


# -- measuring the line ----------------------------------------------------------


def test_wer_counts_edits_per_reference_word():
    assert wer("the copper kettle", "the copper kettle") == 0.0
    assert wer("the copper kettle", "the copper cattle") == pytest.approx(1 / 3)
    assert wer("the copper kettle", "the kettle") == pytest.approx(1 / 3)
    assert wer("", "") == 0.0


def test_spoken_punctuation_counts_as_surviving_and_a_wrong_name_does_not():
    said = "My email is nina.kapoor@example.com."
    assert entities_kept(said, "My email is nina.kapoor at example.com") == {"email": (1, 1)}
    assert entities_kept(said, "My email is Nina dot Kapoor at example dot com") == {"email": (1, 1)}
    assert entities_kept(said, "My email is neena.kapoor at example.com.") == {"email": (1, 0)}


def test_a_dropped_email_is_a_lost_email():
    """Seen live: Whisper drops a verbless trailing fragment, email and all."""
    said = "Cancel my travel mug order please. omar.haddad@example.com."
    assert entities_kept(said, "Cancel my travel mug order please.") == {"email": (1, 0)}


def test_an_order_id_survives_only_if_it_can_be_put_back():
    said = "It is definitely o_1047."
    assert entities_kept(said, "It is definitely O underscore 1047.") == {"order_id": (1, 1)}
    assert entities_kept(said, "It is definitely O-1047.") == {"order_id": (1, 0)}


def test_channel_stats_is_none_for_a_text_run_rather_than_a_perfect_zero():
    text_run = [[{"kind": "user", "text": "hello"}, {"kind": "assistant", "text": "hi"}]]
    assert channel_stats(text_run) is None


def test_channel_stats_reads_spoken_against_heard():
    stats = channel_stats([[
        {"kind": "user", "text": "Refund please.", "spoken": "Refund please. nina.kapoor@example.com."},
        {"kind": "user", "text": "Yes.", "spoken": "Yes."},
    ]])
    assert stats["utterances"] == 2
    assert stats["verbatim"] == 0.5
    assert stats["entities"] == {"email": {"said": 1, "kept": 0}}


# -- the audio ---------------------------------------------------------------------


def tone(ms=200, rate=8000):
    return array.array("h", (int(8000 * math.sin(2 * math.pi * 440 * n / rate))
                             for n in range(rate * ms // 1000)))


def test_the_noise_is_seeded_so_a_resume_hears_the_same_audio():
    a = degrade(tone(), 10.0, seed=7)
    assert a == degrade(tone(), 10.0, seed=7)
    assert a != degrade(tone(), 10.0, seed=8)


def test_a_quiet_line_still_goes_through_mu_law():
    clean = tone()
    line = degrade(clean, None, seed=1)
    assert len(line) == len(clean)
    assert line != clean
    assert max(abs(a - b) for a, b in zip(line, clean)) < 400


def test_the_wav_is_narrowband_mono():
    with wave.open(io.BytesIO(to_wav(tone()))) as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) == (8000, 1, 2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_telephone_resamples_anything_to_8khz():
    rich = array.array("h", (int(8000 * math.sin(2 * math.pi * 440 * n / 44100))
                             for n in range(44100 // 2)))
    wav, ms = telephone(to_wav(rich, rate=44100), None, seed=1)
    with wave.open(io.BytesIO(wav)) as reader:
        assert reader.getframerate() == 8000
    assert ms == pytest.approx(500, abs=30)


# -- the line and the customer on it --------------------------------------------------


def test_a_transcript_is_bought_once_and_then_read_from_disk(tmp_path):
    ear = FakeEar()
    line = fake_line(ear, cache_dir=tmp_path)
    first = line.hear("hello there", seed=1)
    again = line.hear("hello there", seed=1)
    assert ear.calls == 1
    assert (first.cached, again.cached) == (False, True)
    assert again.heard == first.heard


def test_a_different_seed_or_line_is_a_different_transcript(tmp_path):
    ear = FakeEar()
    fake_line(ear, cache_dir=tmp_path).hear("hello", seed=1)
    fake_line(ear, cache_dir=tmp_path).hear("hello", seed=2)
    fake_line(ear, cache_dir=tmp_path, config=LineConfig(snr_db=10)).hear("hello", seed=1)
    assert ear.calls == 3


def test_silence_reaches_the_agent_as_inaudible():
    heard = fake_line(lambda wav: ("  ", 5.0)).hear("anything", seed=1)
    assert heard.heard == INAUDIBLE


def test_each_turn_gets_its_own_noise():
    assert seed_for("t", 1, 0) != seed_for("t", 1, 1) != seed_for("t", 2, 0)
    assert seed_for("t", 1, 0) == seed_for("t", 1, 0)


def test_the_agent_hears_the_transcript_and_the_trace_keeps_both(domain):
    """The whole point: the agent only ever sees what the recogniser heard."""
    task = domain.task("refund_kettle")
    ear = FakeEar(swaps={"nina": "neena"})
    user = VoiceUser(inner=ScriptedUser(lines=task.scripted_user), line=fake_line(ear),
                     task_id=task.id, trial=1)
    store = domain.store()
    trace = Trace()
    client = FakeClient([FakeReply(text="Could you spell your email?"), FakeReply(text="Thanks.")])
    agent.run_model(task, domain.policy, Toolbox(store, trace), user, client)
    store.close()

    first_prompt = client.calls[0]
    assert "neena.kapoor@example.com" in first_prompt[1]["content"]
    assert "nina.kapoor" not in first_prompt[1]["content"]
    opening = [e for e in trace.events if e["kind"] == "user"][0]
    assert "neena" in opening["text"]
    assert opening["spoken"] == task.scripted_user[0]


def test_the_customer_remembers_its_own_words_not_the_transcript(domain):
    """A caller knows what it said. Feeding it the mishearing would make the
    simulator argue with its own brief."""
    class Recorder:
        usage = []

        def __init__(self):
            self.lines = ["It's nina.kapoor@example.com.", "Yes."]
            self.at = 0

        def open(self):
            self.at = 1
            return self.lines[0]

        def reply(self, assistant_text):
            if self.at >= len(self.lines):
                return None
            self.at += 1
            return self.lines[self.at - 1]

    inner = Recorder()
    user = VoiceUser(inner=inner, line=fake_line(FakeEar(swaps={"nina": "neena"})),
                     task_id="x", trial=1)
    assert "neena" in user.open()
    assert user.last_spoken == "It's nina.kapoor@example.com."


def test_text_runs_get_the_prompt_v3_was_measured_with(domain):
    """The voice-aware arm must not leak into the text path, or v3 and every
    text number after it would be measured on different prompts."""
    plain = agent._system_prompt(domain.policy, "2026-01-01")
    assert plain == agent.AGENT_SYSTEM.format(policy=domain.policy.strip(), now="2026-01-01")
    assert agent._system_prompt(domain.policy, "2026-01-01", voice_aware=True).startswith(plain)
    assert "phone call" not in plain


def test_oracle_and_mute_never_go_down_the_line(domain):
    ear = FakeEar()
    for solver in ("oracle", "mute"):
        runner.run_trial(domain, domain.task("refund_kettle"), 1, solver,
                         line=fake_line(ear))
    assert ear.calls == 0


def test_a_recogniser_out_of_quota_stops_the_run_rather_than_failing_a_trial(domain):
    def broke(wav):
        raise QuotaExhausted("ASR HTTP 429: requests per day")

    task = domain.task("refund_kettle")
    verdict = runner.run_trial(domain, task, 1, "model", client=FakeClient([]),
                               scripted=True, line=fake_line(broke))
    assert verdict.error.startswith("quota")


# -- grading and bookkeeping ------------------------------------------------------------


@pytest.mark.parametrize("written, ok", [
    ("22 Spice Market Rd, Chennai 600002", True),
    ("22 Spice Market Road, Chennai 600002", True),
    ("22 spice market road chennai 6-0-0-0-0-2", True),
    ("22 Spice Market Road, Chinnai 600002", False),
    ("22 Spice Market Road, Chennai 6000002", False),
])
def test_an_address_is_graded_as_a_place_not_a_spelling(domain, written, ok):
    task = domain.task("address_change_pending")
    store = domain.store()
    before = store.snapshot()
    store.set_address("o_1050", written)
    after = store.snapshot()
    store.close()
    assert (check_state(task, before, after) == []) is ok


def test_regrading_keeps_what_was_said(domain):
    task = domain.task("refund_kettle")
    verdict = runner.run_trial(domain, task, 1, "model", client=FakeClient([FakeReply(text="Hi.")]),
                               scripted=True, line=fake_line(FakeEar(swaps={"nina": "neena"})))
    again = runner.regrade(domain, verdict)
    spoken = [e.get("spoken") for e in again.transcript if e["kind"] == "user"]
    assert spoken[0] == task.scripted_user[0]


def write_checkpoint(path, channel=None):
    meta = {"harness": HARNESS_VERSION}
    if channel is not None:
        meta["channel"] = channel
    path.write_text(json.dumps({"meta": meta, "verdicts": []}), encoding="utf-8")


def test_a_text_checkpoint_is_not_resumed_over_a_phone_line(tmp_path):
    path = tmp_path / "text.json"
    write_checkpoint(path, "text")
    with pytest.raises(MixedHarness):
        Checkpoint.load(path, channel_name(LineConfig()))


def test_a_checkpoint_from_before_the_line_existed_reads_as_text(tmp_path):
    path = tmp_path / "old.json"
    write_checkpoint(path)
    Checkpoint.load(path)
    with pytest.raises(MixedHarness):
        Checkpoint.load(path, channel_name(LineConfig()))


def test_changing_anything_about_the_line_is_a_new_measurement(tmp_path):
    path = tmp_path / "voice.json"
    write_checkpoint(path, channel_name(LineConfig()))
    Checkpoint.load(path, channel_name(LineConfig()))
    for other in (channel_name(LineConfig(snr_db=15)), channel_name(LineConfig(), voice_aware=True),
                  channel_name(LineConfig(voice="en-US-AriaNeural")), "text"):
        with pytest.raises(MixedHarness):
            Checkpoint.load(path, other)


def test_voice_aware_without_a_line_is_refused(capsys):
    assert runner.main(["--solvers", "oracle", "--voice-aware"]) == 2


def test_voice_without_a_recogniser_key_says_so(monkeypatch, capsys):
    monkeypatch.setattr(runner, "build_client", lambda *a, **k: FakeClient([]))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert runner.main(["--solvers", "model", "--voice", "--k", "1"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().out


# -- trailing silence -------------------------------------------------------------


def test_the_descriptor_is_unchanged_without_padding():
    """runs/voice.json was banked under this exact string; changing it would
    orphan a measurement that is still running."""
    assert LineConfig().descriptor == "voice:en-IN-NeerjaNeural>g711-8000hz-quiet>whisper-large-v3-turbo"
    assert LineConfig(pad_ms=500).descriptor == "voice:en-IN-NeerjaNeural>g711-8000hz-quiet-pad500>whisper-large-v3-turbo"


def test_padding_reaches_the_phone_stage_only_when_set():
    seen = []

    def phone(audio, snr, seed, **kwargs):
        seen.append(kwargs)
        return audio, 1000.0

    for pad in (0, 500):
        Line(config=LineConfig(pad_ms=pad), transcribe=FakeEar(),
             synth=lambda text, voice: text.encode(), phone=phone).hear("hello", seed=1)
    assert seen == [{}, {"pad_ms": 500}]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_padding_appends_silence_without_touching_the_speech():
    wav, ms = telephone(to_wav(tone(300)), None, seed=1)
    padded, padded_ms = telephone(to_wav(tone(300)), None, seed=1, pad_ms=500)
    assert padded_ms == pytest.approx(ms + 500, abs=1)
    with wave.open(io.BytesIO(padded)) as reader:
        frames = reader.readframes(reader.getnframes())
    tail = array.array("h")
    tail.frombytes(frames[-800:])
    assert set(tail) == {0}
