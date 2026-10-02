"""Pure-Python parts of Whisper Diarization Studio: speaker assignment, transcript formats and
meeting summaries. No GPU libraries are imported here, so this file is easy to test anywhere."""
import bisect
import json
import re
import urllib.request

FORMATS = {"txt": "text/plain", "md": "text/markdown", "srt": "application/x-subrip", "vtt": "text/vtt",
           "json": "application/json"}


# ====================================================================== timestamps
def ts(sec, sep=".", hours=True):
    sec = max(0.0, float(sec or 0))
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return ("%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)) if hours else ("%02d:%02d" % (h * 60 + m, s))


def clock(sec):
    sec = int(max(0, sec or 0))
    return "%d:%02d:%02d" % (sec // 3600, sec // 60 % 60, sec % 60) if sec >= 3600 else "%d:%02d" % (sec // 60, sec % 60)


# ====================================================================== speakers
def _speaker_at(turns, starts, a, b):
    """Speaker whose turns overlap [a, b] the most; the nearest turn within 1 s if none overlaps."""
    if not turns:
        return None
    i = bisect.bisect_right(starts, b)
    best, best_ov = None, 0.0
    j = i - 1
    while j >= 0 and turns[j][0] > a - 600:          # turns are short; look back a bounded window
        s, e, spk = turns[j]
        ov = min(b, e) - max(a, s)
        if ov > best_ov:
            best, best_ov = spk, ov
        j -= 1
    if best is not None:
        return best
    mid, near, dist = (a + b) / 2, None, 1.0
    for k in (i - 1, i):
        if 0 <= k < len(turns):
            s, e, spk = turns[k]
            d = 0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
            if d <= dist:
                near, dist = spk, d
    return near


def assign_speakers(segments, turns, max_gap=2.0, max_len=45.0):
    """Turns whisper segments (with word timestamps) and diarization turns [(start, end, speaker)]
    into utterances [{start, end, speaker, text}]. Without turns, each segment is one utterance."""
    if not turns:
        return [{"start": s["start"], "end": s["end"], "speaker": None, "text": s["text"].strip()}
                for s in segments if s["text"].strip()]
    turns = sorted(turns)
    starts = [t[0] for t in turns]
    words = []
    for seg in segments:
        ws = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": " " + seg["text"].strip()}]
        words.extend(w for w in ws if w.get("word", "").strip())
    out, prev = [], None
    for w in words:
        spk = _speaker_at(turns, starts, w["start"], w["end"]) or prev
        prev = spk
        cur = out[-1] if out else None
        if (cur and cur["speaker"] == spk and w["start"] - cur["end"] <= max_gap
                and not (w["end"] - cur["start"] > max_len and re.search(r"[.!?]$", cur["text"]))):
            cur["text"] += w["word"]
            cur["end"] = w["end"]
        else:
            out.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["word"]})
    for u in out:
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip()
    return [u for u in out if u["text"]]


def speaker_label(spk, names):
    if spk is None:
        return None
    return (names or {}).get(spk) or re.sub(r"^SPEAKER_0*(\d)", lambda m: "Speaker %d" % (int(m.group(1)) + 1), spk)


def speaker_stats(utterances):
    stats = {}
    for u in utterances:
        if u["speaker"] is None:
            continue
        s = stats.setdefault(u["speaker"], {"seconds": 0.0, "words": 0, "turns": 0})
        s["seconds"] += u["end"] - u["start"]
        s["words"] += len(u["text"].split())
        s["turns"] += 1
    for s in stats.values():
        s["seconds"] = round(s["seconds"], 1)
    return stats


# ====================================================================== formats
def cues(segments, utterances, max_chars=84, max_dur=6.0):
    """Subtitle cues built from words, never crossing a speaker change."""
    spk_of = []
    for u in utterances:
        spk_of.append((u["start"], u["end"], u["speaker"]))
    turns = sorted((s, e, k) for s, e, k in spk_of if k is not None)
    starts = [t[0] for t in turns]
    out = []
    for seg in segments:
        ws = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": " " + seg["text"].strip()}]
        for w in ws:
            if not w.get("word", "").strip():
                continue
            spk = _speaker_at(turns, starts, w["start"], w["end"]) if turns else None
            cur = out[-1] if out else None
            if (cur and cur["speaker"] == spk and len(cur["text"]) + len(w["word"]) <= max_chars
                    and w["end"] - cur["start"] <= max_dur and w["start"] - cur["end"] < 1.0):
                cur["text"] += w["word"]
                cur["end"] = w["end"]
            else:
                out.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["word"]})
    for c in out:
        c["text"] = c["text"].strip()
    return out


def render(fmt, data, names=None, title="Transcript"):
    """data: transcript.json content (segments, utterances, language, duration, summary...)."""
    utts = data.get("utterances") or []
    names = names or {}
    if fmt == "json":
        d = dict(data)
        d["speaker_names"] = names
        d["utterances"] = [dict(u, speaker_name=speaker_label(u["speaker"], names)) for u in utts]
        return json.dumps(d, indent=1, ensure_ascii=False)
    if fmt == "txt":
        lines = []
        for u in utts:
            who = speaker_label(u["speaker"], names)
            lines.append("[%s] %s%s" % (clock(u["start"]), who + ": " if who else "", u["text"]))
        return "\n".join(lines) + "\n"
    if fmt in ("srt", "vtt"):
        cs = cues(data.get("segments") or [], utts)
        if fmt == "srt":
            blocks = []
            for i, c in enumerate(cs, 1):
                who = speaker_label(c["speaker"], names)
                blocks.append("%d\n%s --> %s\n%s%s\n" % (i, ts(c["start"], ","), ts(c["end"], ","),
                                                         "[%s] " % who if who else "", c["text"]))
            return "\n".join(blocks)
        blocks = ["WEBVTT\n"]
        for c in cs:
            who = speaker_label(c["speaker"], names)
            blocks.append("%s --> %s\n%s%s\n" % (ts(c["start"]), ts(c["end"]), "<v %s>" % who if who else "", c["text"]))
        return "\n".join(blocks)
    if fmt == "md":
        out = ["# %s" % title, ""]
        meta = ["Duration %s" % clock(data.get("duration"))]
        if data.get("language"):
            meta.append("language `%s`" % data["language"])
        stats = speaker_stats(utts)
        if stats:
            meta.append("%d speakers" % len(stats))
        out += ["_" + " · ".join(meta) + "_", ""]
        if stats:
            total = sum(s["seconds"] for s in stats.values()) or 1
            out += ["| Speaker | Talk time | Share |", "|---|---|---|"]
            for spk, s in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]):
                out.append("| %s | %s | %d%% |" % (speaker_label(spk, names), clock(s["seconds"]),
                                                   round(100 * s["seconds"] / total)))
            out.append("")
        if data.get("summary"):
            out += [data["summary"].strip(), ""]
        out += ["## Transcript", ""]
        for u in utts:
            who = speaker_label(u["speaker"], names)
            out.append("**%s%s** %s\n" % ("[%s] " % clock(u["start"]), who + ":" if who else "", u["text"]))
        return "\n".join(out)
    raise ValueError("format must be one of: " + ", ".join(FORMATS))


# ====================================================================== summaries (Ollama)
SUMMARY_PROMPT = """You are a precise meeting assistant. Below is a transcript with timestamps and speaker names.
Write Markdown with exactly these sections:
## Summary
5-10 bullet points covering what was discussed.
## Decisions
Bullets; write "None recorded." if there are none.
## Action items
A table | Owner | Task | Due | using names from the transcript; "?" when unknown. "None recorded." if there are none.
## Open questions
Bullets; "None recorded." if there are none.
Only use facts from the transcript. Write in the transcript's language.{extra}

Transcript:
{text}"""

PART_PROMPT = """Write dense notes (bullets, keep names, numbers, decisions, tasks and owners) about this part
{n} of {total} of a meeting transcript. Only use facts from the text.{extra}

{text}"""


def ollama_chat(url, model, prompt, num_ctx=32768, timeout=1200):
    body = json.dumps({"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}],
                       "options": {"num_ctx": num_ctx, "temperature": 0.2}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"].strip()


def split_text(text, size):
    parts, cur = [], []
    n = 0
    for line in text.splitlines(keepends=True):
        if n + len(line) > size and cur:
            parts.append("".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line)
    if cur:
        parts.append("".join(cur))
    return parts


def summarize(text, url, model, instructions="", part_chars=60000, progress=None):
    """Map-reduce for long meetings: notes per part, then one summary from the notes."""
    extra = ("\nExtra instructions: " + instructions.strip()) if instructions and instructions.strip() else ""
    if len(text) <= part_chars:
        return ollama_chat(url, model, SUMMARY_PROMPT.format(text=text, extra=extra))
    parts = split_text(text, part_chars)
    notes = []
    for i, p in enumerate(parts, 1):
        if progress:
            progress("summarizing part %d/%d" % (i, len(parts)))
        notes.append("### Part %d\n%s" % (i, ollama_chat(url, model, PART_PROMPT.format(n=i, total=len(parts),
                                                                                     text=p, extra=extra))))
    if progress:
        progress("writing the final summary")
    return ollama_chat(url, model, SUMMARY_PROMPT.format(text="\n\n".join(notes), extra=extra))
