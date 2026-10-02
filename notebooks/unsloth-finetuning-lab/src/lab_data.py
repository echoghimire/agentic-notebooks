"""Dataset handling for the Unsloth Fine-tuning Lab (pure Python, no GPU libraries).

Every record is normalised to one of two shapes:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}   chat / instruction data
    {"text": "..."}                                                                raw text (continued pre-training)
Accepted inputs: OpenAI-style messages, ShareGPT conversations (from/value), Alpaca
(instruction/input/output), prompt/completion, question/answer, input/output, query/response, text.
"""
import csv
import io
import json

ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
         "bot": "assistant", "model": "assistant", "tool": "tool"}
PAIRS = (("prompt", "completion"), ("question", "answer"), ("input", "output"), ("query", "response"),
         ("instruction", "response"))


def _content(c):
    if isinstance(c, list):        # OpenAI content parts
        c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    if not isinstance(c, str):
        raise ValueError("message content must be text")
    return c


def normalize(obj):
    if not isinstance(obj, dict):
        raise ValueError("each record must be a JSON object")
    msgs = obj.get("messages", obj.get("conversations", obj.get("conversation")))
    if msgs is not None:
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("messages must be a non-empty list")
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                raise ValueError("each message must be an object")
            role = ROLES.get(str(m.get("role", m.get("from", ""))).lower())
            if not role:
                raise ValueError("unknown role %r" % m.get("role", m.get("from")))
            out.append({"role": role, "content": _content(m.get("content", m.get("value", "")))})
        if not any(m["role"] == "assistant" and m["content"].strip() for m in out):
            raise ValueError("a conversation needs at least one non-empty assistant message")
        return {"messages": out}
    system = obj.get("system") or obj.get("system_prompt")
    head = [{"role": "system", "content": _content(system)}] if system else []
    if "instruction" in obj and "output" in obj:
        if not _content(obj["output"]).strip() or not _content(obj["instruction"]).strip():
            raise ValueError("instruction and output must not be empty")
        user = _content(obj["instruction"]) + ("\n\n" + _content(obj["input"]) if obj.get("input") else "")
        return {"messages": head + [{"role": "user", "content": user},
                                    {"role": "assistant", "content": _content(obj["output"])}]}
    for q, a in PAIRS:
        if q in obj and a in obj:
            if not _content(obj[a]).strip():
                raise ValueError("%s is empty" % a)
            return {"messages": head + [{"role": "user", "content": _content(obj[q])},
                                        {"role": "assistant", "content": _content(obj[a])}]}
    if isinstance(obj.get("text"), str) and obj["text"].strip():
        return {"text": obj["text"]}
    raise ValueError("unrecognised record; use messages, conversations, instruction/output, prompt/completion, "
                     "question/answer or text (keys found: %s)" % ", ".join(sorted(obj)[:8]))


def parse(text, fmt="auto"):
    """Returns (records, errors) where errors is a list of {line, error}."""
    text = text.lstrip("﻿")
    if fmt == "auto":
        s = text.lstrip()
        fmt = "json" if s.startswith("[") else "jsonl" if s.startswith("{") else "csv"
    recs, errs = [], []
    if fmt == "json":
        try:
            items = json.loads(text)
        except ValueError as e:
            return [], [{"line": 0, "error": "not valid JSON: %s" % e}]
        if not isinstance(items, list):
            return [], [{"line": 0, "error": "a .json file must hold a list of records"}]
        rows = list(enumerate(items, 1))
    elif fmt == "jsonl":
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append((n, json.loads(line)))
                except ValueError as e:
                    errs.append({"line": n, "error": "not valid JSON: %s" % e})
    elif fmt == "csv":
        rows = [(n, {k.strip().lower(): v for k, v in r.items() if k}) for n, r in
                enumerate(csv.DictReader(io.StringIO(text)), 2)]
    else:
        raise ValueError("format must be jsonl, json, csv or auto")
    for n, obj in rows:
        try:
            recs.append(normalize(obj))
        except ValueError as e:
            errs.append({"line": n, "error": str(e)})
    return recs, errs


def preview(rec, n=160):
    if "text" in rec:
        s = rec["text"]
    else:
        user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        bot = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
        s = "%s  →  %s" % (user, bot)
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def chars(rec):
    return len(rec["text"]) if "text" in rec else sum(len(m["content"]) for m in rec["messages"])


def stats(records):
    if not records:
        return {"count": 0}
    lens = sorted(chars(r) for r in records)
    chat = sum(1 for r in records if "messages" in r)
    return {"count": len(records), "chat": chat, "text": len(records) - chat,
            "turns_avg": round(sum(len(r["messages"]) for r in records if "messages" in r) / chat, 1) if chat else 0,
            "chars_median": lens[len(lens) // 2], "chars_max": lens[-1],
            "approx_tokens_total": sum(lens) // 4, "approx_tokens_max": lens[-1] // 4}
