"""The phone line between the customer and the agent.

A voice agent never reads what the caller said. It reads what the speech
recogniser *heard*, and the gap between the two is where voice agents fail in
production: an email that comes back as "nina kapoor at example dot com", an
order number that loses its underscore, a postcode read out as a quantity.
This module puts that gap into the harness, so the same twenty tasks can be
measured once over text and once over a phone line and the difference read off.

    customer text --TTS--> speech --8 kHz, mu-law, noise--> audio --ASR--> agent

Only the customer-to-agent direction goes over the line. The agent's replies
reach the simulated customer as text. That is not a shortcut: a simulator that
mishears the agent is a second noise source on the customer side, and every
failure it caused would be the harness scoring itself, which is the failure
mode this project has already paid for three times.

Grading is unchanged. The expected state, `asks_for_card` and the policy rules
all come from the task, which is what the customer actually wanted, so a voice
agent is held to what was said and not to what it heard. That is the honest
system-level reading: a caller whose card refund went to store credit because
"card" came back as "cart" was failed by the product, whichever component did
it.

Every utterance is deterministic. The noise is seeded from (task, trial, turn),
so a resumed run reproduces the audio it would have produced, and every
transcript is cached on disk under that seed, so a resume or a debugging re-run
does not buy the same transcription twice.
"""

from __future__ import annotations

import array
import asyncio
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .llm import _DAILY_LIMIT, _RETRY_HINT, USER_AGENT, LLMError, QuotaExhausted, Usage

#: An Indian English voice. The customers in the seed are called Nina Kapoor,
#: Priya Raman and Omar Haddad and their addresses are in Chennai, and an
#: accent the recogniser was not mostly trained on is the realistic hard case
#: rather than an exotic one.
VOICE_DEFAULT = "en-IN-NeerjaNeural"
ASR_DEFAULT = "whisper-large-v3-turbo"
ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ASR_KEY_ENV = "GROQ_API_KEY"
#: Narrowband telephony. 300-3400 Hz is the band a phone network passes.
PHONE_RATE = 8000
PHONE_BAND = (300, 3400)
MU = 255.0

#: What the agent receives when the recogniser returns nothing at all. An empty
#: user message is refused by some providers and reads as silence to others.
INAUDIBLE = "(inaudible)"

#: Things a support call cannot proceed without. A word error rate says how
#: noisy the line was; whether these came through intact says whether the
#: noise landed anywhere that matters.
ENTITIES = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "order_id": re.compile(r"\bo_\d{4}\b"),
}


@dataclass(frozen=True)
class LineConfig:
    voice: str = VOICE_DEFAULT
    snr_db: float | None = None
    asr_model: str = ASR_DEFAULT

    @property
    def descriptor(self) -> str:
        """Everything about the line that changes what the agent perceives.

        Two runs whose descriptors differ are two measurements, and the
        checkpoint refuses to mix them.
        """
        noise = "quiet" if self.snr_db is None else f"snr{self.snr_db:g}"
        return f"voice:{self.voice}>g711-{PHONE_RATE}hz-{noise}>{self.asr_model}"


def channel_name(config: LineConfig | None, voice_aware: bool = False) -> str:
    """The channel a run was measured on, as the checkpoint compares it.

    A text run is "text" and nothing else, so every checkpoint written before
    the line existed reads as what it was.
    """
    if config is None:
        return "text"
    return config.descriptor + ("+aware" if voice_aware else "")


@dataclass(frozen=True)
class Heard:
    spoken: str
    heard: str
    audio_ms: float = 0.0
    asr_ms: float = 0.0
    cached: bool = False


# -- the three stages ----------------------------------------------------------


def synthesize(text: str, voice: str = VOICE_DEFAULT) -> bytes:
    """Text to MP3 with edge-tts. Imported late so text runs need nothing."""
    import edge_tts

    async def collect() -> bytes:
        chunks = []
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    audio = asyncio.run(collect())
    if not audio:
        raise LLMError(f"edge-tts returned no audio for {text[:60]!r}")
    return audio


def telephone(encoded: bytes, snr_db: float | None, seed: int) -> tuple[bytes, float]:
    """Any audio in, an 8 kHz phone-band WAV out, and its length in ms.

    ffmpeg does the decode, the band limit and the resample. The noise and the
    mu-law step are done here, where the seed and the SNR are exact rather than
    approximated through a filter graph.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise LLMError("ffmpeg is not on PATH, and the phone line needs it")
    low, high = PHONE_BAND
    done = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-af", f"highpass=f={low},lowpass=f={high}",
         "-ac", "1", "-ar", str(PHONE_RATE), "-f", "s16le", "pipe:1"],
        input=encoded, capture_output=True, check=False,
    )
    if done.returncode != 0 or not done.stdout:
        raise LLMError(f"ffmpeg failed: {done.stderr.decode(errors='replace')[:300]}")
    samples = array.array("h")
    samples.frombytes(done.stdout[: len(done.stdout) // 2 * 2])
    line = degrade(samples, snr_db, seed)
    return to_wav(line), 1000.0 * len(line) / PHONE_RATE


def degrade(samples: array.array, snr_db: float | None, seed: int) -> array.array:
    """Add seeded white noise at an exact SNR, then compand to 8-bit mu-law.

    The companding is the continuous mu-law curve quantised to 8 bits, which is
    what a G.711 line does to the signal to within its segment approximation.
    """
    floats = [s / 32768.0 for s in samples]
    if snr_db is not None and floats:
        power = sum(x * x for x in floats) / len(floats)
        if power > 0:
            sigma = math.sqrt(power / (10 ** (snr_db / 10)))
            rng = random.Random(seed)
            floats = [x + rng.gauss(0.0, sigma) for x in floats]
    out = array.array("h")
    log_mu = math.log1p(MU)
    for x in floats:
        x = max(-1.0, min(1.0, x))
        y = math.copysign(math.log1p(MU * abs(x)) / log_mu, x)
        q = round(y * 127) / 127
        back = math.copysign(((1 + MU) ** abs(q) - 1) / MU, q)
        out.append(int(max(-32768, min(32767, round(back * 32767)))))
    return out


def to_wav(samples: array.array, rate: int = PHONE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


class Transcriber:
    """Whisper on Groq's free tier, with the same defences as the chat client.

    The traps are the ones `llm.py` documents: name the client or Cloudflare
    answers 403 "error code: 1010", and a per-day 429 is its own exception
    because retrying it burns the attempts and records a trial that never
    happened.
    """

    RETRY_ON = frozenset({408, 409, 429, 500, 502, 503, 504})
    MAX_BACKOFF = 60.0

    def __init__(self, api_key: str, model: str = ASR_DEFAULT, *,
                 url: str = ASR_URL, attempts: int = 4, timeout: float = 60.0) -> None:
        self.model = model
        self._key = api_key
        self._url = url
        self._attempts = attempts
        self._timeout = timeout
        #: The last response's rate-limit headers, kept for the smoke output.
        self.last_headers: dict[str, str] = {}

    def __call__(self, wav: bytes) -> tuple[str, float]:
        boundary = f"----reruns{uuid.uuid4().hex}"
        body = _multipart(boundary, {
            "model": self.model,
            "language": "en",
            "temperature": "0",
            "response_format": "json",
        }, ("file", "utterance.wav", "audio/wav", wav))
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {self._key}",
        }
        last = "no attempt was made"
        for attempt in range(1, self._attempts + 1):
            request = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode())
                    self.last_headers = {
                        k.lower(): v for k, v in response.headers.items()
                        if k.lower().startswith("x-ratelimit")
                    }
                    return str(payload.get("text") or "").strip(), (
                        time.perf_counter() - started) * 1000
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:1200]
                last = f"ASR HTTP {exc.code}: {detail}"
                if exc.code == 429 and _DAILY_LIMIT.search(detail):
                    raise QuotaExhausted(last) from exc
                if exc.code not in self.RETRY_ON or attempt == self._attempts:
                    raise LLMError(last) from exc
                delay = min(2.0 ** attempt, 30.0)
                header = exc.headers.get("Retry-After") if exc.headers else None
                if header:
                    try:
                        delay = float(header)
                    except ValueError:
                        pass
                hint = _RETRY_HINT.search(detail)
                if hint:
                    delay = max(delay, float(hint.group(1) or hint.group(2)) + 1.0)
                time.sleep(min(delay, self.MAX_BACKOFF))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = f"ASR {type(exc).__name__}: {exc}"
                if attempt == self._attempts:
                    raise LLMError(last) from exc
                time.sleep(min(2.0 ** attempt, 30.0))
        raise LLMError(last)


def _multipart(boundary: str, fields: dict[str, str], upload: tuple[str, str, str, bytes]) -> bytes:
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                  f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    name, filename, content_type, data = upload
    out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
              f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode())
    out.write(data)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue()


# -- the line ------------------------------------------------------------------


@dataclass
class Line:
    """Text in, what a recogniser heard out. Stages are injected for tests."""

    config: LineConfig
    transcribe: Callable[[bytes], tuple[str, float]]
    synth: Callable[[str, str], bytes] = synthesize
    phone: Callable[[bytes, float | None, int], tuple[bytes, float]] = telephone
    cache_dir: Path | None = None

    def hear(self, text: str, seed: int) -> Heard:
        key = hashlib.sha256(f"{self.config.descriptor}|{seed}|{text}".encode()).hexdigest()
        cached = self._read(key)
        if cached is not None:
            return Heard(spoken=text, heard=cached["heard"], audio_ms=cached["audio_ms"],
                         asr_ms=cached["asr_ms"], cached=True)
        audio = self.synth(text, self.config.voice)
        wav, audio_ms = self.phone(audio, self.config.snr_db, seed)
        heard, asr_ms = self.transcribe(wav)
        heard = heard.strip() or INAUDIBLE
        self._write(key, {"spoken": text, "heard": heard, "audio_ms": audio_ms,
                          "asr_ms": asr_ms, "seed": seed, "line": self.config.descriptor})
        return Heard(spoken=text, heard=heard, audio_ms=audio_ms, asr_ms=asr_ms)

    def _read(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        target = self.cache_dir / f"{key}.json"
        if not target.is_file():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def _write(self, key: str, entry: dict) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")


def build_line(config: LineConfig, cache_dir: Path | None = None) -> Line | None:
    """A working line, or None when there is no ASR key -- never one that cannot work."""
    key = os.environ.get(ASR_KEY_ENV)
    if not key:
        return None
    return Line(config=config, transcribe=Transcriber(key, config.asr_model), cache_dir=cache_dir)


def seed_for(task_id: str, trial: int, turn: int) -> int:
    """Stable across processes, unlike hash(), which Python salts per run."""
    digest = hashlib.sha256(f"{task_id}|{trial}|{turn}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class VoiceUser:
    """A simulated customer heard down a phone line.

    The inner simulator keeps its own words in its own history, because a
    customer knows what it said. Only the agent gets the recogniser's version.
    """

    inner: object
    line: Line
    task_id: str
    trial: int
    heard: list[Heard] = field(default_factory=list)

    @property
    def usage(self) -> list[Usage]:
        return getattr(self.inner, "usage", [])

    @property
    def last_spoken(self) -> str | None:
        return self.heard[-1].spoken if self.heard else None

    def open(self) -> str:
        return self._over_the_line(self.inner.open())

    def reply(self, assistant_text: str) -> str | None:
        text = self.inner.reply(assistant_text)
        return None if text is None else self._over_the_line(text)

    def _over_the_line(self, text: str) -> str:
        heard = self.line.hear(text, seed_for(self.task_id, self.trial, len(self.heard)))
        self.heard.append(heard)
        return heard.heard


# -- reading what the line did -------------------------------------------------


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edits to turn the reference into the hypothesis, per word."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


_SPOKEN_FORMS = ((r"\s+at\s+", "@"), (r"\s+dot\s+", "."), (r"\s+underscore\s+", "_"))


def _squash(text: str) -> str:
    """The transcript with spoken punctuation put back, as any agent could.

    "omar.haddad at example.com" survived: nothing was lost that a reader
    cannot restore. "neena.kapoor at example.com" did not.
    """
    text = text.lower()
    for pattern, symbol in _SPOKEN_FORMS:
        text = re.sub(pattern, symbol, text)
    return re.sub(r"\s+", "", text)


def entities_kept(spoken: str, heard: str) -> dict[str, tuple[int, int]]:
    """Per entity kind, (how many were said, how many survived the line)."""
    out = {}
    squashed = _squash(heard)
    for kind, pattern in ENTITIES.items():
        said = [m.group(0).lower().rstrip(".") for m in pattern.finditer(spoken)]
        if said:
            out[kind] = (len(said), sum(1 for s in said if s in squashed))
    return out


def channel_stats(transcripts: list[list[dict]]) -> dict | None:
    """What the line did across a run, from the transcripts alone.

    None for a run with no spoken turns, so a text run reports nothing here
    rather than a perfect-looking zero.
    """
    pairs = [
        (event["spoken"], event["text"])
        for transcript in transcripts
        for event in transcript
        if event.get("kind") == "user" and "spoken" in event
    ]
    if not pairs:
        return None
    rates = [wer(spoken, heard) for spoken, heard in pairs]
    kept: dict[str, list[int]] = {}
    for spoken, heard in pairs:
        for kind, (said, through) in entities_kept(spoken, heard).items():
            tally = kept.setdefault(kind, [0, 0])
            tally[0] += said
            tally[1] += through
    return {
        "utterances": len(pairs),
        "mean_wer": round(sum(rates) / len(rates), 4),
        "verbatim": round(sum(r == 0 for r in rates) / len(rates), 4),
        "entities": {kind: {"said": s, "kept": k} for kind, (s, k) in sorted(kept.items())},
    }
