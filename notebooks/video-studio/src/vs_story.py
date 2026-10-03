"""Storyboard: the JSON the local LLM writes, its validation (including the language and script), and a
fallback built from the source's own sentences when the model fails.

storyboard = {
  "title", "tagline", "language": "ne", "category": "news|tech|education|story|promo|other",
  "tone": "tragic|serious|neutral|upbeat|inspiring", "music_prompt": "...",
  "scenes": [{"layout": "headline|photo|bullets|stats|code|quote|outro", "heading", "kicker", "bullets": [...],
              "narration", "photo": <index into the source photos or -1>, "image_prompt", "code", "quote",
              "stats": [{"value", "label"}]}]
}
The model never writes HTML: vs_scenes.py animates whatever it puts in these fields.
"""
import json
import re
import urllib.request

LAYOUTS = ("headline", "photo", "bullets", "stats", "code", "quote", "outro")
ALIASES = {"title": "headline", "image": "photo", "intro": "headline", "end": "outro"}
LENGTHS = {15: (2, 3), 30: (4, 5), 60: (6, 8), 90: (8, 11)}
WPS = {"en": 2.5, "ne": 2.0, "hi": 2.2}            # spoken words per second
LANGS = {"en": ("English", "English", ""), "ne": ("Nepali", "नेपाली", ", in Devanagari script"),
         "hi": ("Hindi", "हिन्दी", ", in Devanagari script"), "es": ("Spanish", "español", ""),
         "fr": ("French", "français", ""), "it": ("Italian", "italiano", ""), "pt": ("Portuguese", "português", ""),
         "bn": ("Bengali", "বাংলা", ", in Bengali script"), "ur": ("Urdu", "اردو", ", in Urdu script")}
DEVANAGARI = {"ne", "hi", "mr", "sa", "mai"}
CATEGORIES = ("news", "tech", "education", "story", "promo", "other")
TONES = ("tragic", "serious", "neutral", "upbeat", "inspiring")

PROMPT = """You are an experienced {role}. Turn the SOURCE below into a {length}-second video storyboard
(it will be rendered both as a vertical reel and as a landscape video).

LANGUAGE: write every "title", "tagline", "heading", "kicker", "bullets", "quote" and "narration" in {lang_name}
({lang_native}){script}. Never use any other language or script for those fields. Only "image_prompt" and
"music_prompt" are written in English.

Return ONLY a JSON object:
{{
  "title": "video title, max 9 words",
  "tagline": "one-line summary, max 16 words",
  "category": one of "news", "tech", "education", "story", "promo", "other",
  "tone": one of "tragic", "serious", "neutral", "upbeat", "inspiring",
  "music_prompt": "background music in English: genre, mood, tempo, instruments, no vocals",
  "scenes": [
    {{
      "layout": one of "headline", "photo", "bullets", "stats", "quote", "code", "outro",
      "heading": "on-screen headline for this scene, max 8 words",
      "kicker": "tiny label above it, e.g. a place and date, max 4 words (may be empty)",
      "bullets": ["for bullets: 2-4 points, max 9 words each; for photo: one optional sub-line"],
      "narration": "what the voice says during this scene: 1-3 natural spoken sentences, max {max_words} words",
      "photo": index of the PHOTOS entry that fits this scene best, or -1,
      "image_prompt": "only if no photo fits and the topic is not a real news event: an illustration idea, else empty",
      "code": "code layout only: a short real snippet from the source",
      "quote": "quote layout only: a sentence that appears in the source"
    }}
  ]
}}

Rules:
- {n_min} to {n_max} scenes. The first scene uses layout "headline", the last uses layout "outro".
- Narration in total about {words} words, so it fits {length} seconds. Spoken, clear and concrete.
- {photo_rule}
- Every fact must come from the SOURCE: never invent numbers, names, places, quotes or causes.{news_rule}{stats_rule}{extra}

PHOTOS:
{photos}

SOURCE ({kind}):
{text}
"""


def lang_info(lang):
    return LANGS.get(lang, (lang, lang, ""))


def build_prompt(src, length, lang, photos):
    n_min, n_max = LENGTHS.get(length, LENGTHS[60])
    name, native, script = lang_info(lang)
    words = int(length * WPS.get(lang, 2.3))
    role = {"github": "tech video producer", "youtube": "video editor", "pdf": "explainer video producer"}.get(
        src["kind"], "news and explainer video producer")
    photo_rule = ("Prefer \"photo\" scenes built on the PHOTOS, and give each scene its own photo when there are "
                  "enough. Pick the photo whose description matches the scene." if photos else
                  "There are no photos; use \"bullets\" and \"quote\" scenes, and image prompts only for non-news topics.")
    news_rule = ("\n- If this is news about real people: be factual, neutral and respectful; attribute claims (\"police said\"); "
                 "no speculation, no graphic detail, no dramatic adjectives; image_prompt must be empty.")
    stats_rule = ("\n- Do not add a \"stats\" scene: one with the real numbers is added automatically."
                  if src["kind"] in ("github", "youtube") else "\n- Use \"stats\" only for numbers stated in the SOURCE.")
    extra = ("\n- The user asked: " + src["instructions"]) if src.get("instructions") else ""
    plist = "\n".join("[%d] %s" % (i, p.get("description") or p.get("caption") or p.get("alt") or "(no description)")
                      for i, p in enumerate(photos)) or "(none)"
    return PROMPT.format(role=role, length=length, lang_name=name, lang_native=native, script=script,
                         max_words=45 if length > 15 else 30, n_min=n_min, n_max=n_max, words=words, photo_rule=photo_rule,
                         news_rule=news_rule, stats_rule=stats_rule, extra=extra, photos=plist, kind=src["kind"],
                         text=src["text"])


def ollama_json(url, model, prompt, keep_alive="10m", timeout=900):
    body = json.dumps({"model": model, "stream": False, "format": "json", "keep_alive": keep_alive,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0.5, "num_ctx": 16384}}).encode()
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


def script_check(sb, lang):
    """Raises ValueError when the on-screen text and narration are not in the expected script."""
    text = " ".join([sb.get("title") or ""] + [" ".join([sc.get("heading") or "", sc.get("narration") or ""] + list(sc.get("bullets") or []))
                                               for sc in sb.get("scenes", []) if isinstance(sc, dict)])
    letters = [c for c in text if c.isalpha()]
    if not letters:
        raise ValueError("the storyboard has no text")
    cjk = sum(1 for c in letters if "぀" <= c <= "鿿") / len(letters)
    deva = sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)
    latin = sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ") / len(letters)
    name = lang_info(lang)[0]
    if lang not in ("zh", "ja") and cjk > 0.02:
        raise ValueError("the text drifted into Chinese/Japanese instead of %s" % name)
    if lang in DEVANAGARI and deva < 0.6:
        raise ValueError("the text is not in %s (Devanagari); only %d%% Devanagari" % (name, deva * 100))
    if lang in ("en", "es", "fr", "it", "pt") and latin < 0.8:
        raise ValueError("the text is not in %s" % name)


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


def _scene(sc, n_photos):
    layout = str(sc.get("layout") or "bullets").lower().strip()
    layout = ALIASES.get(layout, layout)
    layout = layout if layout in LAYOUTS else "bullets"
    bullets = [_s(b, 90) for b in (sc.get("bullets") or []) if isinstance(b, (str, int, float)) and str(b).strip()][:4]
    try:
        photo = int(sc.get("photo", -1))
    except (TypeError, ValueError):
        photo = -1
    out = {"layout": layout, "heading": _s(sc.get("heading"), 80), "kicker": _s(sc.get("kicker"), 40), "bullets": bullets,
           "narration": _words(sc.get("narration"), 60), "image_prompt": _s(sc.get("image_prompt"), 300),
           "photo": photo if 0 <= photo < n_photos else -1,
           "code": "\n".join(str(sc.get("code") or "").splitlines()[:10])[:600], "quote": _s(sc.get("quote"), 220), "stats": []}
    if layout == "stats":
        out["stats"] = [{"value": _s(x.get("value"), 20), "label": _s(x.get("label"), 30)} for x in (sc.get("stats") or [])
                        if isinstance(x, dict) and x.get("value")][:3]
    if layout == "bullets" and not bullets:
        out["layout"] = "photo" if out["photo"] >= 0 else "quote"
        out["quote"] = out["quote"] or out["narration"]
    if layout == "code" and not out["code"].strip():
        out["layout"] = "bullets" if bullets else "photo"
    if layout == "quote" and not out["quote"]:
        out["layout"] = "bullets" if bullets else "photo"
    return out


def clean(sb, src, length, lang, n_photos=0, keep_stats=False):
    """Validates and repairs a storyboard; raises ValueError if nothing usable is left."""
    if not isinstance(sb, dict) or not isinstance(sb.get("scenes"), list):
        raise ValueError("storyboard must be an object with a scenes list")
    n_min, n_max = LENGTHS.get(length, LENGTHS[60])
    scenes = []
    for sc in sb["scenes"]:
        if not isinstance(sc, dict):
            continue
        out = _scene(sc, n_photos)
        if out["layout"] == "stats" and not (keep_stats and out["stats"]):
            continue                                   # only real numbers: added below from the source facts
        if not out["heading"] and not out["narration"]:
            continue
        scenes.append(out)
    if len(scenes) < (1 if length <= 15 else 2):
        raise ValueError("storyboard has too few usable scenes")
    blank = {"heading": "", "kicker": "", "bullets": [], "narration": "", "image_prompt": "", "photo": -1, "code": "", "quote": "", "stats": []}
    if scenes[0]["layout"] != "headline":
        scenes.insert(0, dict(blank, layout="headline", photo=scenes[0]["photo"], image_prompt=scenes[0]["image_prompt"]))
    if scenes[-1]["layout"] != "outro":
        scenes.append(dict(blank, layout="outro"))
    for sc in scenes[1:-1]:
        if sc["layout"] in ("headline", "outro"):
            sc["layout"] = "bullets" if sc["bullets"] else "photo"
    stats = stats_for(src)
    if stats and not any(s["layout"] == "stats" for s in scenes) and length > 15:
        scenes.insert(min(2, len(scenes) - 1), dict(blank, layout="stats", heading="", narration=stats_narration(src), stats=stats))
    while len(scenes) > n_max + 1:                     # keep headline/outro, drop from the middle
        scenes.pop(-2)
    cat = str(sb.get("category") or "").lower()
    tone = str(sb.get("tone") or "").lower()
    title = _s(sb.get("title") or src["title"], 90)
    out = {"title": title, "tagline": _s(sb.get("tagline") or src.get("description"), 140), "language": lang,
           "category": cat if cat in CATEGORIES else ("news" if src["kind"] == "article" else "other"),
           "tone": tone if tone in TONES else "neutral",
           "music_prompt": _s(sb.get("music_prompt") or "light background music, no vocals", 200),
           "scenes": scenes, "source": {k: src.get(k) for k in ("kind", "url", "title", "facts")}}
    if out["category"] == "news" or out["tone"] in ("tragic", "serious"):
        for sc in scenes:
            sc["image_prompt"] = ""                    # never invent pictures of real events
        if out["tone"] in ("tragic", "serious"):
            out["music_prompt"] = "soft slow ambient pad, minimal, somber, warm, no drums, no vocals"
    h, o = scenes[0], scenes[-1]
    h["heading"] = h["heading"] or title
    h["narration"] = h["narration"] or out["tagline"] or title
    o["heading"] = o["heading"] or ""
    o["narration"] = o["narration"] or ""
    return out


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+", text) if len(s.split()) >= 4]


def fallback(src, length, lang, n_photos=0):
    """A plain storyboard straight from the source's own sentences (so it is always in the right language)."""
    body = re.sub(r"^(Title|Site|Date|Author|Summary):.*$", "", src["text"], flags=re.M)
    sents = sentences(body)
    n = LENGTHS.get(length, LENGTHS[60])[1] - 2
    per = max(1, min(3, len(sents) // max(1, n)))
    scenes = [{"layout": "headline", "heading": src["title"], "narration": src.get("description") or (sents[0] if sents else src["title"]),
               "photo": 0 if n_photos else -1}]
    for i in range(max(1, n)):
        chunk = sents[1 + i * per: 1 + (i + 1) * per]
        if not chunk:
            break
        first = chunk[0].split()
        scenes.append({"layout": "photo" if n_photos else "bullets", "heading": " ".join(first[:8]) + ("…" if len(first) > 8 else ""),
                       "bullets": [_s(s, 90) for s in chunk[1:3]] or [_s(chunk[0], 90)], "narration": _words(" ".join(chunk), 40),
                       "photo": (i + 1) % n_photos if n_photos else -1})
    scenes.append({"layout": "outro", "heading": src["title"], "narration": ""})
    return clean({"title": src["title"], "tagline": src.get("description"), "category": "news" if src["kind"] == "article" else "other",
                  "scenes": scenes}, src, length, lang, n_photos)


def write_storyboard(src, length, lang, ollama_url, model, keep_alive="10m", photos=(), log=None):
    prompt = build_prompt(src, length, lang, photos)
    last, nudge = None, ""
    for attempt in range(3):
        try:
            sb = clean(ollama_json(ollama_url, model, prompt + nudge, keep_alive), src, length, lang, len(photos))
            script_check(sb, lang)
            return sb, None
        except Exception as e:
            last = e
            if log:
                log.warning("storyboard attempt %d failed: %s", attempt + 1, e)
            name, native, script = lang_info(lang)
            nudge = ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED: %s. Answer again with valid JSON, every text field in %s (%s)%s."
                     % (str(e)[:200], name, native, script))
    return fallback(src, length, lang, len(photos)), "the model's script was unusable (%s); used the article's own sentences" % last


def validate_user_storyboard(sb, src, length, lang, n_photos):
    """For storyboards edited in the page or sent by an agent: same repairs, keeping their stats scenes."""
    out = clean(sb, dict(src, kind="prompt"), length, sb.get("language") or lang, n_photos, keep_stats=True)
    out["source"] = sb.get("source") or out["source"]
    for k in ("category", "tone", "music_prompt"):
        if sb.get(k):
            out[k] = sb[k] if k == "music_prompt" else out[k]
    return out
