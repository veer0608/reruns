"""Build site/voice.html: voice trials you can listen to, with what the agent heard.

    python site/build_voice_page.py

Reads runs/voice.json, so re-running it after more trials are banked refreshes
every count. The audio is regenerated through the same line (edge-tts, the
8 kHz phone band, mu-law, the trial's own seed) from the text the customer
said. edge-tts is not bit-stable between calls, so it can differ slightly from
the audio Whisper was given during the run; the transcripts shown are the run's.
Needs edge-tts and ffmpeg, like --voice itself.
"""

from __future__ import annotations

import array
import base64
import difflib
import html
import io
import json
import re
import shutil
import string
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evals.compare import Side  # noqa: E402
from reruns.voice import ENTITIES, _squash, seed_for, synthesize, telephone  # noqa: E402

CHECKPOINT = REPO / "runs" / "voice.json"
OUT = REPO / "site" / "voice.html"
#: GitHub Pages serves main:/docs, so the public copy is a complete document there.
DOCS = REPO / "docs"
K = 5
REPO_URL = "https://github.com/veer0608/reruns"

#: The trials shown, and what each one demonstrates. The notes are claims about
#: these exact transcripts; `build` checks the verdict matches before writing.
CARDS = [
    ("refund_kettle", 1, False, "A wrong email gets trusted",
     "Whisper heard <q>neena</q> for <q>nina</q>. The lookup failed and the agent handed the call "
     "to a human straight away, without asking the caller to repeat or spell it. Over text that is "
     "the right move, because an email that is not on file really is wrong. Over a phone line it "
     "gives away a call that one question would have saved."),
    ("cancel_pending_lamp", 1, False, "Same sentence, a different transcript",
     "The same opening line, with the same local audio processing, came back as <q>neena</q> on "
     "trials 1 and 2 and as <q>nina</q> on trials 3 to 5. The agent failed exactly the two trials "
     "where the name was misheard and passed the three where it was not."),
    ("ambiguous_order", 1, True, "A missing email gets asked for",
     "Whisper dropped the email completely: it tends to cut a short fragment at the end of an "
     "utterance. The agent noticed it had nothing to look up, asked for it, and finished the "
     "refund."),
    ("address_change_pending", 1, True, "Road for Rd, digits for a postcode",
     "The caller said <q>Rd</q> and <q>600002</q>. The agent heard <q>Road</q> and "
     "<q>6-0-0-0-0-2</q> and wrote the address down correctly. This trial passes only because "
     "addresses are graded as places, not spellings. Under the old exact-string check a correct "
     "agent would have failed."),
    ("refund_cable_only", 1, False, "Not every failure is the line",
     "The email came through intact, and nothing else was lost. The agent refunded the cable "
     "before telling the customer the amount, which breaks policy rule 7. That is the model's "
     "habit, not the phone line's, which is why the headline comparison waits for a text run on "
     "the same model."),
]


# -- audio -----------------------------------------------------------------------


def phone_audio(text: str, seed: int) -> tuple[str, list[float], float]:
    """An mp3 data URI of the utterance down the line, its waveform peaks, its length in ms."""
    wav, ms = telephone(synthesize(text), None, seed)
    with wave.open(io.BytesIO(wav)) as reader:
        samples = array.array("h")
        samples.frombytes(reader.readframes(reader.getnframes()))
    buckets = 72
    size = max(1, len(samples) // buckets)
    peaks = [max((abs(s) for s in samples[i:i + size]), default=0) for i in range(0, size * buckets, size)]
    top = max(peaks) or 1
    mp3 = subprocess.run(
        [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", "8000", "-b:a", "24k", "-f", "mp3", "pipe:1"],
        input=wav, capture_output=True, check=True,
    ).stdout
    return ("data:audio/mpeg;base64," + base64.b64encode(mp3).decode(),
            [round(p / top, 3) for p in peaks], ms)


# -- reading transcripts -----------------------------------------------------------


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9@._]", "", token.lower()).strip(".")


def word_diff(spoken: str, heard: str) -> list[tuple[str, str, str]]:
    """(op, said, heard) runs, matching on normalised words, keeping the original text."""
    a, b = spoken.split(), heard.split()
    matcher = difflib.SequenceMatcher(a=[_norm(t) for t in a], b=[_norm(t) for t in b], autojunk=False)
    return [(op, " ".join(a[i1:i2]), " ".join(b[j1:j2])) for op, i1, i2, j1, j2 in matcher.get_opcodes()]


_SPOKEN = ((r"\s+at\s+", "@"), (r"\s+dot\s+", "."), (r"\s+underscore\s+", "_"))
_SAME_WORD = {"okay": "ok", "rd": "road", "st": "street", "ave": "avenue", "ln": "lane"}


def canonical(text: str) -> str:
    """A phrase reduced to what it carries, so a spoken form matches its written one.

    "nina.kapoor at example.com" is "nina.kapoor@example.com", "$99" is "99
    dollars", "Rd" is "Road" (the grader treats them as one address) and
    "6-0-0-0-0-2" is "600002". "neena" is still not "nina", and "O-1047" is
    still not "o_1047", because an agent cannot know which one was meant.
    """
    t = f" {text.lower()} "
    t = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", t)
    for pattern, symbol in _SPOKEN:
        t = re.sub(pattern, symbol, t)
    t = re.sub(r"(?<=\d)-(?=\d)", "", t)
    return "".join(_SAME_WORD.get(w.strip("."), w.strip(".")) for w in re.findall(r"[a-z0-9@._]+", t))


_LOSSY = {"replace": "swap", "delete": "lost", "insert": "extra"}


def classify(ops) -> list[tuple[str, str, str]]:
    """Diff runs labelled equal, same (written differently, nothing lost) or lossy.

    A change is tried alone and then together with one neighbouring word,
    because "99 dollars" heard as "$99" diffs as an equal "99" plus a deleted
    "dollars", and neither half is recoverable on its own.
    """
    out: list[tuple[str, str, str]] = []
    runs = list(ops)
    i = 0
    while i < len(runs):
        op, said, heard = runs[i]
        if op == "equal":
            out.append(("equal", said, heard))
        elif canonical(said) == canonical(heard):
            out.append(("same", said, heard))
        elif out and out[-1][0] == "equal" and _joins(out, said, heard):
            pass
        elif i + 1 < len(runs) and runs[i + 1][0] == "equal" and runs[i + 1][1].split():
            next_said, next_heard = runs[i + 1][1].split(), runs[i + 1][2].split()
            s, h = f"{said} {next_said[0]}".strip(), f"{heard} {next_heard[0]}".strip()
            if canonical(s) == canonical(h):
                out.append(("same", s, h))
                runs[i + 1] = ("equal", " ".join(next_said[1:]), " ".join(next_heard[1:]))
            else:
                out.append((_LOSSY[op], said, heard))
        else:
            out.append((_LOSSY[op], said, heard))
        i += 1
    return [run for run in out if run[1] or run[2]]


def _joins(out, said: str, heard: str) -> bool:
    """Fold the last word of the preceding equal run into this change, if that makes it recoverable."""
    prev_said, prev_heard = out[-1][1].split(), out[-1][2].split()
    if not prev_said or not prev_heard:
        return False
    s, h = f"{prev_said[-1]} {said}".strip(), f"{prev_heard[-1]} {heard}".strip()
    if canonical(s) != canonical(h):
        return False
    out[-1] = ("equal", " ".join(prev_said[:-1]), " ".join(prev_heard[:-1]))
    out.append(("same", s, h))
    return True


def lossy(ops) -> bool:
    return any(kind in {"swap", "lost", "extra"} for kind, _, _ in classify(ops))


def opening_email_fate(verdict) -> str | None:
    """What the line did to the email in the customer's first message."""
    users = [e for e in verdict.transcript if e.get("kind") == "user" and "spoken" in e]
    if not users:
        return None
    said = ENTITIES["email"].search(users[0]["spoken"])
    if not said:
        return None
    heard = _squash(users[0]["text"])
    if said.group(0).lower().rstrip(".") in heard:
        return "kept"
    return "misheard" if ENTITIES["email"].search(heard) else "dropped"


# -- rendering ---------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_diff(ops) -> str:
    parts = []
    for kind, said, heard in classify(ops):
        if kind == "equal":
            parts.append(esc(heard))
        elif kind == "same":
            parts.append(f'<span class="same" title="said: {esc(said)}">{esc(heard)}</span>')
        elif kind == "swap":
            parts.append(f'<span class="swap"><del>{esc(said)}</del><ins>{esc(heard)}</ins></span>')
        elif kind == "lost":
            parts.append(f'<span class="lost"><del>{esc(said)}</del><em>not heard</em></span>')
        else:
            parts.append(f'<span class="swap"><ins>{esc(heard)}</ins></span>')
    return " ".join(parts)


def render_wave(peaks: list[float]) -> str:
    bars = "".join(
        f'<rect x="{i * 4}" y="{20 - max(1, p * 18):.1f}" width="2.4" height="{max(2, p * 36):.1f}" rx="1"/>'
        for i, p in enumerate(peaks)
    )
    svg = (f'<svg class="wave {{layer}}" viewBox="0 0 {len(peaks) * 4} 40" preserveAspectRatio="none" '
           f'aria-hidden="true">{bars}</svg>')
    return f'<span class="wavewrap">{svg.format(layer="base")}{svg.format(layer="top")}</span>'


def render_call(event) -> str:
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (event.get("arguments") or {}).items())
    if event["ok"]:
        result = event.get("result")
        if isinstance(result, dict) and "customer_id" in result and "name" in result:
            outcome = f'{result["customer_id"]}, {result["name"]}'
        elif isinstance(result, dict) and "refund_id" in result:
            outcome = f'{result["refund_id"]} issued'
        elif isinstance(result, dict) and result.get("escalated"):
            outcome = "handed to a human"
        else:
            outcome = "ok"
        cls = "call"
    else:
        outcome = str(event.get("result") or "failed")
        cls = "call err"
    return (f'<li class="{cls}"><code><b>{esc(event["name"])}</b>({esc(args)})</code>'
            f'<span class="out">{esc(outcome)}</span></li>')


def render_card(verdict, title: str, note: str, index: int, extra: str = "") -> str:
    rows, turn = [], 0
    for event in verdict.transcript:
        kind = event.get("kind")
        if kind == "user":
            audio, peaks, ms = phone_audio(event.get("spoken", event["text"]),
                                           seed_for(verdict.task_id, verdict.trial, turn))
            turn += 1
            ops = word_diff(event.get("spoken", event["text"]), event["text"])
            changed = lossy(ops)
            rows.append(
                f'<li class="caller">'
                f'<button class="play" type="button" data-src="{audio}" aria-label="Play the caller, {ms / 1000:.1f} seconds">'
                f'<span class="icon" aria-hidden="true"></span>{render_wave(peaks)}'
                f'<span class="dur">{ms / 1000:.1f}s</span></button>'
                f'<div class="lines"><p class="said"><span class="tag">said</span>{esc(event.get("spoken", ""))}</p>'
                f'<p class="heard{" clean" if not changed else ""}"><span class="tag">heard</span>{render_diff(ops)}</p></div></li>'
            )
        elif kind == "assistant":
            rows.append(f'<li class="agent"><span class="tag">agent</span><p>{esc(event["text"])}</p></li>')
        elif kind == "call":
            rows.append(render_call(event))
    stamp = "pass" if verdict.passed else "fail"
    why = "; ".join(verdict.reasons[:2] + [f"rule {v.rule} {v.name}" for v in verdict.violations])
    return f"""
<article class="record" id="call-{index}">
  <header>
    <div class="meta"><span class="task">{esc(verdict.task_id)}</span><span class="trial">trial {verdict.trial} of {K}</span></div>
    <h3>{esc(title)}</h3>
    <span class="stamp {stamp}">{stamp}</span>
  </header>
  <p class="note">{note}</p>
  {extra}
  <ol class="log">{''.join(rows)}</ol>
  {f'<p class="why"><span class="tag">graded</span>{esc(why)}</p>' if why else ''}
</article>"""


def build() -> None:
    side = Side.load(str(CHECKPOINT))
    by = {(v.task_id, v.trial): v for v in side.verdicts}
    banked = len(side.verdicts)

    fates: dict[str, list[bool]] = {"kept": [], "dropped": [], "misheard": []}
    for verdict in side.verdicts:
        if verdict.task_id == "unknown_email":
            continue
        fate = opening_email_fate(verdict)
        if fate:
            fates[fate].append(verdict.passed)

    cards = []
    for index, (task, trial, expect_pass, title, note) in enumerate(CARDS, 1):
        verdict = by.get((task, trial))
        if verdict is None or verdict.passed != expect_pass:
            raise SystemExit(f"{task} trial {trial} is missing or no longer {'passes' if expect_pass else 'fails'}; "
                             "update CARDS before publishing a note that is no longer true")
        extra = ""
        if task == "cancel_pending_lamp" and (task, 3) in by:
            other = [e for e in by[(task, 3)].transcript if e.get("kind") == "user"][0]
            extra = (f'<p class="aside"><span class="tag">trial 3 heard</span>'
                     f'{render_diff(word_diff(other["spoken"], other["text"]))}</p>')
        cards.append(render_card(verdict, title, note, index, extra))
        print(f"  card {index}: {task} trial {trial}")

    def row(label, key, gloss):
        outcomes = fates[key]
        return (f'<tr><th scope="row">{label}<span>{gloss}</span></th>'
                f'<td>{len(outcomes)}</td><td>{sum(outcomes)}</td><td>{len(outcomes) - sum(outcomes)}</td></tr>')

    page = TEMPLATE.substitute(
        banked=banked,
        total=20 * K,
        cards="".join(cards),
        rows=row("Came through", "kept", "the agent could read it back")
             + row("Dropped entirely", "dropped", "nothing to look up")
             + row("Misheard", "misheard", "a plausible, wrong address"),
        repo=REPO_URL,
        page_url=PAGE_URL,
        card_url=PAGE_URL.rsplit("/", 1)[0] + "/card.png",
    )
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) from {banked} banked trials")
    write_pages(page)
    write_card(by[("refund_kettle", 1)])


def write_pages(page: str) -> None:
    """The same page as a whole document, plus an index, for GitHub Pages.

    The artifact copy has no <html> or <head> because claude.ai wraps it; a
    static host needs both, and the head content is everything before <main>.
    """
    head, _, body = page.partition("<main")
    document = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}</head>\n<body>\n<main{body}</body>\n</html>\n"
    )
    DOCS.mkdir(exist_ok=True)
    (DOCS / "voice.html").write_text(document, encoding="utf-8")
    (DOCS / "index.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=voice.html"><title>reruns</title></head>'
        '<body><a href="voice.html">What the agent heard</a></body></html>\n', encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {DOCS / 'voice.html'} for GitHub Pages")


PAGE_URL = "https://veer0608.github.io/reruns/voice.html"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CARD = string.Template("""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..800&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
html, body { margin: 0; width: 1200px; height: 630px; overflow: hidden; }
body { background: #f2f4f3; color: #152120; font-family: "IBM Plex Mono", Consolas, monospace;
       display: grid; grid-template-rows: auto 1fr auto; padding: 64px 72px 56px; box-sizing: border-box; }
.eyebrow { font-size: 24px; letter-spacing: 0.12em; color: #0b6e6e; font-weight: 500; }
h1 { font-family: "Archivo", "Arial Narrow", sans-serif; font-stretch: 72%; font-weight: 800; font-size: 118px;
     line-height: 1; margin: 22px 0 18px; letter-spacing: -0.01em; }
.claim { font-family: "Archivo", "Arial Narrow", sans-serif; font-stretch: 80%; font-weight: 650; font-size: 44px; line-height: 1.15; }
.claim em { font-style: normal; color: #0b6e6e; }
.row { display: grid; grid-template-columns: 1fr 360px; gap: 40px; align-items: end; }
.lines { display: grid; gap: 10px; font-size: 26px; }
.tag { font-size: 16px; letter-spacing: 0.1em; color: #5a6a67; margin-right: 14px; }
del { color: #a8322b; text-decoration-thickness: 2px; }
ins { text-decoration: none; background: #fbe6c2; color: #7d430a; padding: 0 6px; border-radius: 3px; }
svg rect { fill: #0b6e6e; }
.url { font-size: 22px; color: #5a6a67; margin-top: 26px; }
</style></head><body>
<div class="eyebrow">RERUNS &middot; VOICE MODE</div>
<div><h1>What the agent heard</h1>
<div class="claim">A missing email gets asked for. <em>A wrong one gets trusted.</em></div></div>
<div><div class="row"><div class="lines">
<div><span class="tag">SAID</span>nina.kapoor@example.com</div>
<div><span class="tag">HEARD</span><del>nina</del> <ins>neena</ins>.kapoor at example.com</div></div>
$wave</div><div class="url">veer0608.github.io/reruns/voice.html</div></div>
</body></html>""")


def write_card(verdict) -> None:
    """docs/card.png, the 1200x630 link preview, rendered by headless Edge.

    Skipped with a note where Edge is missing: the page still works, it just
    unfurls without an image.
    """
    if not EDGE.is_file():
        print("  no Edge found, so docs/card.png was not rebuilt")
        return
    opening = [e for e in verdict.transcript if e.get("kind") == "user"][0]
    _, peaks, _ = phone_audio(opening["spoken"], seed_for(verdict.task_id, verdict.trial, 0))
    bars = "".join(f'<rect x="{i * 5}" y="{50 - max(2, p * 46):.1f}" width="3" height="{max(4, p * 92):.1f}" rx="1.5"/>'
                   for i, p in enumerate(peaks))
    wave = f'<svg viewBox="0 0 {len(peaks) * 5} 100" width="360" height="100">{bars}</svg>'
    source = REPO / "site" / "card.html"
    source.write_text(CARD.substitute(wave=wave), encoding="utf-8")
    target = DOCS / "card.png"
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--window-size=1200,630", "--virtual-time-budget=8000",
                    f"--screenshot={target}", source.as_uri()], check=True, capture_output=True, timeout=120)
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)")


TEMPLATE = string.Template(Path(__file__).with_name("voice_template.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    build()
