"""Storyboard: the JSON the local LLM writes, its validation, and a fallback that needs no LLM.

storyboard = {
  "title": "...", "tagline": "...",
  "music_prompt": "upbeat electronic, 110 bpm ...",
  "scenes": [{"layout": "title|bullets|image|stats|code|quote|outro", "heading": "...", "bullets": [...],
              "narration": "...", "image_prompt": "...", "code": "...", "quote": "...",
              "stats": [{"value": "12.3k", "label": "stars"}]}]
}
The model never writes HTML: templates in vs_scenes.py animate whatever it puts in these fields.
"""
import json
import re
import urllib.request

LAYOUTS = ("title", "bullets", "image", "stats", "code", "quote", "outro")
LENGTHS = {30: (4, 5), 60: (6, 8), 90: (8, 11)}
WORDS_PER_SECOND = 2.4

PROMPT = """You are a video scriptwriter. Turn the SOURCE below into a {length}-second explainer video storyboard.

Return ONLY a JSON object with this shape:
{{
  "title": "short video title (max 8 words)",
  "tagline": "one-line hook (max 14 words)",
  "music_prompt": "background music description: genre, mood, tempo, instruments (no vocals)",
  "scenes": [
    {{
      "layout": one of "title", "bullets", "image", "stats", "code", "quote", "outro",
      "heading": "on-screen heading, max 7 words",
      "bullets": ["2-4 short on-screen points, max 9 words each (bullets layout only)"],
      "narration": "what the voice says during this scene: 1-3 natural sentences, max 40 words",
      "image_prompt": "a vivid illustration for this scene: subject, style, lighting; no text, no logos, no people's faces",
      "code": "a short real snippet from the source, max 8 lines (code layout only, else empty)",
      "quote": "one striking sentence from the source (quote layout only, else empty)"
    }}
  ]
}}

Rules:
- {n_min} to {n_max} scenes. The first scene uses layout "title", the last uses layout "outro".
- Use "bullets" for most middle scenes; use "image" for a visual moment, "code" only if the source has real code
  (install commands count), "quote" only for a sentence that really appears in the source.{stats_rule}
- Narration in total about {words} words, spoken, friendly and concrete. Never read URLs aloud.
- Every fact must come from the SOURCE. Do not invent numbers, names or features.
- Write in the language of the SOURCE.{extra}

SOURCE ({kind}):
{text}
"""


def build_prompt(src, length):
    n_min, n_max = LENGTHS.get(length, LENGTHS[60])
    stats_rule = ("\n- Do not add a \"stats\" scene: one with the real numbers is added automatically."
                  if src["kind"] in ("github", "youtube") else "")
    extra = ("\n- The user asked: " + src["instructions"]) if src.get("instructions") else ""
    return PROMPT.format(length=length, n_min=n_min, n_max=n_max, words=int(length * WORDS_PER_SECOND),
                         stats_rule=stats_rule, extra=extra, kind=src["kind"], text=src["text"])


def ollama_json(url, model, prompt, keep_alive="10m", timeout=600):
    body = json.dumps({"model": model, "stream": False, "format": "json", "keep_alive": keep_alive,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0.6, "num_ctx": 16384}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("the model did not return JSON")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------- validation
def _s(v, n):
    v = re.sub(r"\s+", " ", str(v or "")).strip()
    return v if len(v) <= n else v[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _words(v, n):
    w = str(v or "").split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def stats_for(src):
    f = src.get("facts") or {}
    if src["kind"] == "github":
        stats = [{"value": f.get("stars"), "label": "GitHub stars"}, {"value": f.get("forks"), "label": "forks"}]
        if f.get("language"):
            stats.append({"value": f["language"], "label": "main language"})
        elif f.get("license"):
            stats.append({"value": f["license"], "label": "license"})
        return [s for s in stats if s["value"]]
    if src["kind"] == "youtube":
        return [s for s in ({"value": f.get("views"), "label": "views"}, {"value": f.get("duration"), "label": "long"},
                            {"value": f.get("channel"), "label": "channel"}) if s["value"]]
    return []


def stats_narration(src):
    f = src.get("facts") or {}
    if src["kind"] == "github":
        return "%s has %s stars and %s forks on GitHub%s." % (
            (f.get("name") or "It").split("/")[-1], f.get("stars"), f.get("forks"),
            ", written mostly in %s" % f["language"] if f.get("language") else "")
    if src["kind"] == "youtube":
        return "The original video by %s has %s views." % (f.get("channel") or "its creator", f.get("views"))
    return ""


def clean(sb, src, length):
    """Validates and repairs a model storyboard; raises ValueError if nothing usable is left."""
    if not isinstance(sb, dict) or not isinstance(sb.get("scenes"), list):
        raise ValueError("storyboard must be an object with a scenes list")
    n_min, n_max = LENGTHS.get(length, LENGTHS[60])
    scenes = []
    for sc in sb["scenes"]:
        if not isinstance(sc, dict):
            continue
        layout = str(sc.get("layout") or "bullets").lower().strip()
        layout = layout if layout in LAYOUTS else "bullets"
        bullets = [_s(b, 80) for b in (sc.get("bullets") or []) if isinstance(b, (str, int, float)) and str(b).strip()][:4]
        out = {"layout": layout, "heading": _s(sc.get("heading"), 70), "bullets": bullets,
               "narration": _words(sc.get("narration"), 60), "image_prompt": _s(sc.get("image_prompt"), 300),
               "code": "\n".join(str(sc.get("code") or "").splitlines()[:10])[:600],
               "quote": _s(sc.get("quote"), 220), "stats": []}
        if layout == "bullets" and not bullets:
            out["layout"] = "image" if out["image_prompt"] else "quote"
            out["quote"] = out["quote"] or out["narration"]
        if layout == "code" and not out["code"].strip():
            out["layout"] = "bullets" if bullets else "image"
        if layout == "quote" and not out["quote"]:
            out["layout"] = "bullets" if bullets else "image"
        if layout == "stats":
            continue                                   # real numbers only: added below
        if not out["heading"] and not out["narration"]:
            continue
        scenes.append(out)
    if len(scenes) < 2:
        raise ValueError("storyboard has fewer than 2 usable scenes")
    if scenes[0]["layout"] != "title":
        scenes.insert(0, {"layout": "title", "heading": "", "bullets": [], "narration": "", "image_prompt": scenes[0]["image_prompt"],
                          "code": "", "quote": "", "stats": []})
    if scenes[-1]["layout"] != "outro":
        scenes.append({"layout": "outro", "heading": "", "bullets": [], "narration": "", "image_prompt": "",
                       "code": "", "quote": "", "stats": []})
    for sc in scenes[1:-1]:
        if sc["layout"] in ("title", "outro"):
            sc["layout"] = "bullets" if sc["bullets"] else "image"
    stats = stats_for(src)
    if stats:
        scenes.insert(min(2, len(scenes) - 1), {"layout": "stats", "heading": "By the numbers", "bullets": [],
                                                "narration": stats_narration(src), "image_prompt": "", "code": "",
                                                "quote": "", "stats": stats})
    while len(scenes) > n_max + 1:                     # keep title/outro, drop from the middle
        scenes.pop(-2)
    title = _s(sb.get("title") or src["title"], 80)
    sb2 = {"title": title, "tagline": _s(sb.get("tagline") or src.get("description"), 120),
           "music_prompt": _s(sb.get("music_prompt") or "light upbeat electronic background music, no vocals", 200),
           "scenes": scenes, "source": {k: src.get(k) for k in ("kind", "url", "title", "facts")}}
    t, o = scenes[0], scenes[-1]
    t["heading"] = t["heading"] or title
    t["narration"] = t["narration"] or sb2["tagline"] or title
    o["heading"] = o["heading"] or title
    o["narration"] = o["narration"] or "Thanks for watching."
    return sb2


def fallback(src, length):
    """A plain storyboard straight from the source, for when the model fails."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", src["text"]) if len(p.split()) >= 8]
    heads = [ln[3:].strip() for ln in src["text"].splitlines() if ln.startswith("## ")]
    n = LENGTHS.get(length, LENGTHS[60])[1] - 2
    scenes = [{"layout": "title", "heading": src["title"], "narration": src.get("description") or src["title"],
               "image_prompt": "abstract technology illustration, soft gradient light, minimal"}]
    for i, p in enumerate(paras[:n]):
        sents = re.split(r"(?<=[.!?])\s+", p)
        scenes.append({"layout": "bullets", "heading": heads[i] if i < len(heads) else "Key point %d" % (i + 1),
                       "bullets": [_s(s, 80) for s in sents[:3]], "narration": _words(" ".join(sents[:2]), 40),
                       "image_prompt": "illustration about: " + _s(sents[0], 120)})
    scenes.append({"layout": "outro", "heading": src["title"], "narration": "Thanks for watching."})
    return clean({"title": src["title"], "tagline": src.get("description"), "scenes": scenes}, src, length)


def write_storyboard(src, length, ollama_url, model, keep_alive="10m", log=None):
    prompt = build_prompt(src, length)
    last = None
    for attempt in range(2):
        try:
            return clean(ollama_json(ollama_url, model, prompt, keep_alive), src, length), None
        except Exception as e:
            last = e
            if log:
                log.warning("storyboard attempt %d failed: %s", attempt + 1, e)
    return fallback(src, length), "the model's storyboard was unusable (%s); used a plain one instead" % last


def validate_user_storyboard(sb, src, length):
    """For storyboards edited in the page or sent by an agent: same repairs, but keep their stats scenes."""
    stats = [sc for sc in sb.get("scenes", []) if isinstance(sc, dict) and sc.get("layout") == "stats"]
    out = clean(dict(sb, scenes=[sc for sc in sb.get("scenes", []) if not (isinstance(sc, dict) and sc.get("layout") == "stats")]),
                dict(src, kind="prompt"), length)
    for sc in stats:
        st = [{"value": _s(x.get("value"), 20), "label": _s(x.get("label"), 30)} for x in sc.get("stats", [])
              if isinstance(x, dict) and x.get("value")][:3]
        if st:
            out["scenes"].insert(min(2, len(out["scenes"]) - 1), {
                "layout": "stats", "heading": _s(sc.get("heading") or "By the numbers", 70), "bullets": [],
                "narration": _words(sc.get("narration"), 60), "image_prompt": "", "code": "", "quote": "", "stats": st})
    out["source"] = sb.get("source") or out["source"]
    return out
