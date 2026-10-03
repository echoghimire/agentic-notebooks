"""Video Studio: any link or idea -> narrated video with music, as landscape (16:9) and reel (9:16) MP4s.

Runs as its own process (started by the notebook with studio_http.spawn):
    APP_PASSWORD=... python vs_server.py --port 7860

Pipeline per job (one job at a time, files in WORK_DIR/jobs/<id>/):
    source (vs_source: text, photos, language) -> photos downloaded, edited, described (vs_media)
    -> storyboard in the video's language (local LLM via Ollama, vs_story) -> [optional review/edit]
    -> visuals: the source's photos first, SDXL only for non-news topics -> narration (vs_tts: Kokoro, Indic
    Parler-TTS, Piper) -> music matched to the tone (MusicGen) -> mix -> render landscape + reel in parallel
Photos, images and narration are cached by content, so editing one scene and re-rendering redoes only that scene.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

import studio_http as K
import vs_media as M
import vs_motion as MO
import vs_scenes as V
import vs_source as SRC
import vs_story as ST
import vs_tts as TTS

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR", "/kaggle/working/video_studio")
JOBS = os.path.join(WORK, "jobs")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:12b")
LLM_KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "10m")
DESCRIBE_PHOTOS = os.environ.get("DESCRIBE_PHOTOS", "1") == "1"
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "25"))
MUSIC_MODEL = os.environ.get("MUSIC_MODEL", "facebook/musicgen-small")
TTS_ENABLED = os.environ.get("TTS", "1") != "0"
MEDIA_DEVICE = os.environ.get("MEDIA_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cpu")
UNLOAD_AFTER = os.environ.get("UNLOAD_AFTER", "0") == "1"
DEPTH_ON = os.environ.get("DEPTH_PARALLAX", "1") == "1"            # 2.5D parallax of photos (Depth Anything V2 Small)
CUTOUT_ON = os.environ.get("SUBJECT_CUTOUT", "1") == "1"           # subject lift-off in headline scenes (BiRefNet)
MAPS_ON = os.environ.get("LOCATION_MAPS", "1") == "1"              # map fly-in for news with a place (OpenStreetMap)
I2V_MODEL = os.environ.get("I2V_MODEL", "")                        # "ltx" / "wan" / a repo id; "" = off
I2V_MAX_CLIPS = int(os.environ.get("I2V_MAX_CLIPS", "3"))
MOTION_CHOICES = ("auto", "parallax", "ai", "none")
DEFAULT_QUALITY = os.environ.get("RENDER_QUALITY", "1080p")
FPS = int(os.environ.get("RENDER_FPS", "30"))
FORMATS = ("landscape", "reel")
STYLE_CHOICES = ("auto",) + tuple(V.STYLES)
LANG_CHOICES = ("auto", "ne", "en", "hi", "es", "fr", "it", "pt")
MAX_WAIT = 85
log = K.file_logger(os.environ.get("APP_LOG", "/kaggle/working/logs/video_studio.log"), "video")

LOCK = threading.RLock()
WAKE = threading.Condition(LOCK)
QUEUE = []
MEDIA = M.Media(IMAGE_MODEL, IMAGE_STEPS, MEDIA_DEVICE, MUSIC_MODEL, MEDIA_DEVICE, TTS_DEVICE, log)
MOTION = MO.Motion(MEDIA_DEVICE, log)


# ====================================================================== jobs on disk
def jdir(jid):
    jid = K.safe_name(jid, 40)
    d = os.path.join(JOBS, jid)
    if not jid or not os.path.isfile(os.path.join(d, "job.json")):
        raise K.HTTPError(404, "unknown job %r" % jid)
    return d


def load(jid):
    return K.read_json(os.path.join(jdir(jid), "job.json"))


def save(job, **kw):
    with LOCK:
        path = os.path.join(JOBS, job["id"], "job.json")
        cur = K.read_json(path, job) or job
        cur.update(kw, updated=time.time())
        K.write_json(path, cur)
        job.clear()
        job.update(cur)
    return job


def all_jobs():
    out = []
    if os.path.isdir(JOBS):
        for name in os.listdir(JOBS):
            j = K.read_json(os.path.join(JOBS, name, "job.json"))
            if j:
                out.append(j)
    return sorted(out, key=lambda j: -j["created"])


def files_of(jid):
    d = os.path.join(JOBS, jid)
    return {f: "/api/jobs/%s/files/%s" % (jid, f) for f in ("landscape.mp4", "reel.mp4", "landscape.jpg", "reel.jpg")
            if os.path.exists(os.path.join(d, f))}


def public(job, full=False):
    keys = ("id", "title", "state", "step", "progress", "created", "updated", "error", "warnings", "options",
            "duration", "render", "seconds", "language", "style", "voice")
    out = {k: job.get(k) for k in keys if k in job}
    out["files"] = files_of(job["id"])
    if full:
        d = os.path.join(JOBS, job["id"])
        sb = K.read_json(os.path.join(d, "storyboard.json"))
        if sb:
            out["storyboard"] = sb
        photos = K.read_json(os.path.join(d, "photos.json"), []) or []
        out["photos"] = [{"index": i, "url": "/api/jobs/%s/files/photos/%s" % (job["id"], os.path.basename(p["path"])),
                          "caption": p.get("caption"), "description": p.get("description"), "kind": p.get("kind") or "page",
                          "credit": p.get("credit") if "credit" in p else None, "source": p.get("url") or None,
                          "video": "/api/jobs/%s/files/photos/%s" % (job["id"], os.path.basename(p["clip"])) if p.get("clip") else None}
                         for i, p in enumerate(photos)]
        out["images"] = {str(i): "/api/jobs/%s/files/%s/%s" % (job["id"], os.path.basename(os.path.dirname(p)), os.path.basename(p))
                         for i, p in (job.get("scene_images") or {}).items() if p}
    return out


def parse_options(b, base=None):
    o = dict(base or {"style": "auto", "length": 60, "voice": "auto", "language": "auto", "music": bool(MUSIC_MODEL),
                      "captions": True, "formats": list(FORMATS), "quality": DEFAULT_QUALITY, "review": False,
                      "motion": "auto", "map": True, "sfx": True})
    if b.get("style") not in (None, ""):
        if b["style"] not in STYLE_CHOICES:
            raise ValueError("style must be one of " + ", ".join(STYLE_CHOICES))
        o["style"] = b["style"]
    if b.get("length") not in (None, ""):
        L = int(b["length"])
        if L not in ST.LENGTHS:
            raise ValueError("length must be 15, 30, 60 or 90 seconds")
        o["length"] = L
    if b.get("voice"):
        if b["voice"] not in TTS.VOICES and b["voice"] not in ("auto", "none"):
            raise ValueError("voice must be auto, none or one of %s" % ", ".join(TTS.VOICES))
        o["voice"] = b["voice"]
    if b.get("language"):
        if b["language"] not in LANG_CHOICES:
            raise ValueError("language must be one of " + ", ".join(LANG_CHOICES))
        o["language"] = b["language"]
    if b.get("motion"):
        if b["motion"] not in MOTION_CHOICES:
            raise ValueError("motion must be one of " + ", ".join(MOTION_CHOICES))
        o["motion"] = b["motion"]
    for k in ("music", "captions", "review", "map", "sfx"):
        if b.get(k) is not None and b.get(k) != "":
            o[k] = b[k] if isinstance(b[k], bool) else str(b[k]).lower() in ("1", "true", "yes", "on")
    if b.get("formats") is not None and b.get("formats") != "":
        f = b["formats"] if isinstance(b["formats"], list) else [x.strip() for x in str(b["formats"]).split(",")]
        if not f or any(x not in FORMATS for x in f):
            raise ValueError("formats must be a list with landscape and/or reel")
        o["formats"] = [x for x in FORMATS if x in f]
    if b.get("quality"):
        if b["quality"] not in ("1080p", "720p"):
            raise ValueError("quality must be 1080p or 720p")
        o["quality"] = b["quality"]
    if o["music"] and not MUSIC_MODEL:
        o["music"] = False
    return o


def create(source, opts):
    source = str(source or "").strip()
    if not source:
        raise ValueError("paste a link (news article, GitHub, YouTube, PDF...) or describe the video")
    if len(source) > 4000:
        raise ValueError("the description is longer than 4000 characters")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
    d = os.path.join(JOBS, jid)
    for sub in ("images", "voice", "photos"):
        os.makedirs(os.path.join(d, sub))
    job = {"id": jid, "title": source.split()[0][:100], "input": source, "options": opts, "state": "queued",
           "step": "waiting", "progress": 0.0, "created": time.time(), "warnings": [], "approved": not opts["review"]}
    K.write_json(os.path.join(d, "job.json"), job)
    return enqueue(job)


def enqueue(job, **kw):
    with WAKE:
        save(job, state="queued", step="waiting", error=None, **kw)
        if job["id"] not in QUEUE:
            QUEUE.append(job["id"])
        WAKE.notify_all()
    return job


# ====================================================================== pipeline
def h(*parts):
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:16]


def scene_duration(sc, narr_dur, length):
    base = {"headline": 3.4, "stats": 4.2, "outro": 3.0}.get(sc["layout"], 3.2)
    if length <= 15:
        base = min(base, 2.6)
    if sc["layout"] == "code":
        base = max(base, 2.6 + len(sc.get("code") or "") / 28.0)
    if sc["layout"] == "bullets":
        base = max(base, 1.4 + 0.45 * len(sc.get("bullets") or []))
    return round(max(base, narr_dur + (0.7 if length <= 15 else 1.0)), 2)


def auto_style(sb, src):
    if sb.get("category") == "news" or sb.get("tone") in ("tragic", "serious"):
        return "broadcast"
    if src["kind"] == "article" and sb.get("category") not in ("tech", "education", "promo", "story"):
        return "broadcast"                              # an article the model did not label: treat it as news
    return {"github": "midnight", "pdf": "paper"}.get(src["kind"]) or {
        "tech": "midnight", "education": "paper", "story": "documentary", "promo": "neon"}.get(sb.get("category"), "documentary")


def step(job, name, frac):
    save(job, state="running", step=name, progress=round(frac, 3))
    log.info("%s: %s", job["id"], name)


def assign_photos(scenes, n_photos):
    """Scenes without a photo get unused ones (no photo twice in a row); headline and outro reuse the lead photo."""
    used = {sc["photo"] for sc in scenes if sc.get("photo", -1) >= 0}           # -2 = the user chose no picture
    free = [i for i in range(n_photos) if i not in used]
    prev = None
    for k, sc in enumerate(scenes):
        if sc.get("photo", -1) == -1 and n_photos and sc["layout"] in ("photo", "bullets", "quote", "headline", "outro"):
            if sc["layout"] in ("headline", "outro"):
                sc["photo"] = 0
            elif free:
                sc["photo"] = free.pop(0)
            elif sc["layout"] == "photo" and n_photos > 1:
                sc["photo"] = (prev + 1) % n_photos if prev is not None else 0
        prev = sc.get("photo") if sc.get("photo", -1) >= 0 else prev
    return scenes


def map_for(place, style, d):
    """Map snapshots for a place name (cached per job); None when the place cannot be found."""
    geo = MO.geocode(place, os.path.join(WORK, "geo"))
    if not geo:
        return None
    out = os.path.join(d, "map_" + h(place, style))
    paths = {k: os.path.join(out, "map_%s.jpg" % k) for k, _ in MO.MAP_ZOOMS}
    if not all(os.path.exists(p) for p in paths.values()):
        paths = MO.place_map(geo[0], geo[1], out, "light" if style in V.LIGHT else "dark")
    return paths


def add_motion(job, scenes, visuals, motion, news, warnings):
    """Adds depth maps, subject cut-outs or AI clips to the scenes' pictures (cached next to each picture)."""
    noted = set()

    def note(msg):
        if msg not in noted:
            noted.add(msg)
            warnings.append(msg)
    use_ai = motion == "ai" and bool(I2V_MODEL) and not news
    if motion == "ai" and news:
        note("AI motion is never used on news (it would invent movement); real photos got 2.5D parallax instead")
    elif motion == "ai" and not I2V_MODEL:
        note("AI motion needs I2V_MODEL in the notebook settings; used 2.5D parallax")
    order = sorted(visuals)
    clips, broken = 0, set()                            # an effect whose model failed once is not retried
    for k, i in enumerate(order):
        v, lay = visuals[i], scenes[i]["layout"]
        base = v["path"]
        if v.get("frames"):
            continue                                    # already moving (the user's video clip)
        step(job, "adding motion %d/%d" % (k + 1, len(order)), 0.2 + 0.2 * k / max(1, len(order)))
        try:
            kind = "ai"
            if use_ai and "ai" not in broken and clips < I2V_MAX_CLIPS and lay in ("headline", "photo", "bullets"):
                fdir = base + ".i2v_" + h(I2V_MODEL)
                if not MO.frames_in(fdir):
                    prompt = (scenes[i].get("image_prompt") or scenes[i].get("heading") or "") + \
                        ", subtle natural motion, slow cinematic camera move, stable, high quality"
                    MOTION.animate(base, prompt, fdir, I2V_MODEL)
                frames = MO.frames_in(fdir)
                if frames:
                    v.update(frames=frames, fps=24)
                    clips += 1
                    continue
            kind = "cut"
            if CUTOUT_ON and "cut" not in broken and motion in ("auto", "ai") and lay == "headline":
                cut, none = base + ".cut.png", base + ".nocut"
                if not os.path.exists(cut) and not os.path.exists(none):
                    if not MOTION.cutout(base, cut)[0]:
                        open(none, "w").close()            # no clear subject: remember, use parallax
                if os.path.exists(cut):
                    v["cut"] = cut
                    continue
            kind = "depth"
            if DEPTH_ON and "depth" not in broken:
                dp = base + ".depth.png"
                if not os.path.exists(dp):
                    MOTION.depth(base, dp)
                v["depth"] = dp
        except Exception as e:
            broken.add(kind)
            log.warning("motion (%s) for scene %d failed: %s\n%s", kind, i + 1, e, traceback.format_exc())
            note("some motion effects were skipped (%s)" % str(e)[:160])
    MOTION.unload()


def run_job(job):
    d = os.path.join(JOBS, job["id"])
    o = job["options"]
    t0 = time.time()
    warnings = []
    # 1. source + photos
    src = K.read_json(os.path.join(d, "source.json"))
    photos = K.read_json(os.path.join(d, "photos.json"))
    if not src:
        step(job, "reading the link", 0.02)
        src = SRC.fetch_source(job["input"])
        step(job, "collecting photos", 0.05)
        photos = M.fetch_photos(src.get("images") or [], os.path.join(d, "photos"), 14, referer=src.get("url"), log=log)
        for im in src.get("images") or []:
            im.pop("bytes", None)
        if photos and DESCRIBE_PHOTOS and LLM_MODEL:
            step(job, "looking at the photos", 0.07)
            photos = M.describe_photos(photos, OLLAMA_URL, LLM_MODEL, LLM_KEEP_ALIVE, log,
                                       topic=src["title"] + ". " + (src.get("description") or ""), keep=8)
        photos = photos[:8]
        K.write_json(os.path.join(d, "source.json"), src)
        K.write_json(os.path.join(d, "photos.json"), photos)
        save(job, title=src["title"][:100])
    photos = photos or []
    lang = o.get("language") if o.get("language") not in (None, "auto") else (src.get("language") or "en")
    # 2. storyboard
    sb = K.read_json(os.path.join(d, "storyboard.json"))
    if not sb:
        step(job, "writing the script in %s (%s)" % (ST.lang_info(lang)[0], LLM_MODEL), 0.1)
        try:
            sb, warn = ST.write_storyboard(src, o["length"], lang, OLLAMA_URL, LLM_MODEL, LLM_KEEP_ALIVE, photos, log)
        except Exception as e:
            sb, warn = ST.fallback(src, o["length"], lang, len(photos)), "script model unavailable (%s); used the article's own sentences" % e
        if warn:
            warnings.append(warn)
        K.write_json(os.path.join(d, "storyboard.json"), sb)
        save(job, title=sb["title"][:100])
    lang = sb.get("language") or lang
    style = o["style"] if o["style"] != "auto" else auto_style(sb, src)
    save(job, language=lang, style=style)
    if not job.get("approved"):
        save(job, state="review", step="review the script, then render", progress=0.15, warnings=warnings)
        return
    scenes = assign_photos([dict(sc) for sc in sb["scenes"]], len(photos))
    news = sb.get("category") == "news" or sb.get("tone") in ("tragic", "serious")
    # 2b. where it happened: a map fly-in after the headline (news with a real place, 30 s and longer)
    if o.get("map", True) and MAPS_ON and news and sb.get("place") and o["length"] > 15 and len(scenes) > 1:
        try:
            step(job, "drawing the map of %s" % sb["place"], 0.12)
            mp = map_for(sb["place"], style, d)
            if mp:
                scenes.insert(1, {"layout": "map", "heading": sb.get("place_local") or "", "kicker": "", "bullets": [],
                                  "narration": "", "photo": -2, "code": "", "quote": "", "stats": [], "inserted": True,
                                  "map": dict(mp, label=sb.get("place_local") or sb["place"].split(",")[0])})
        except Exception as e:
            log.warning("map skipped: %s", e)
            warnings.append("map skipped (%s)" % str(e)[:120])
    n = len(scenes)
    credit = (src.get("facts") or {}).get("site") or ""
    # 3. visuals: source photos first; generated images only for non-news topics
    visuals = {}
    for i, sc in enumerate(scenes):
        p = sc.get("photo", -1)
        if 0 <= p < len(photos):
            ph = photos[p]
            visuals[i] = {"path": ph["path"], "w": ph["w"], "h": ph["h"],     # links credit their own site
                          "credit": ph["credit"] if "credit" in ph else credit}
            if ph.get("frames_dir"):                    # the user's own video clip
                visuals[i].update(frames=MO.frames_in(ph["frames_dir"]), fps=ph.get("fps", 24))
    if IMAGE_MODEL and not news:
        for i, sc in enumerate(scenes):
            if i in visuals or sc["layout"] not in ("headline", "bullets", "photo", "outro") or not sc.get("image_prompt"):
                continue
            path = os.path.join(d, "images", h(sc["image_prompt"], style, IMAGE_MODEL) + ".png")
            if not os.path.exists(path):
                step(job, "drawing image %d/%d" % (i + 1, n), 0.15 + 0.3 * i / n)
                try:
                    MEDIA.image(sc["image_prompt"], style, path, seed=int(h(sc["image_prompt"])[:6], 16))
                except Exception as e:
                    log.error("image failed\n%s", traceback.format_exc())
                    warnings.append("image for scene %d failed: %s" % (i + 1, str(e)[:200]))
                    continue
            visuals[i] = {"path": path, "w": 1024, "h": 1024, "credit": "AI"}
    if 0 in visuals and n - 1 not in visuals and scenes[n - 1].get("photo", -1) != ST.NO_PHOTO:
        visuals[n - 1] = dict(visuals[0])
    # 3b. motion: real photos move in 2.5D (depth parallax) or lift their subject; AI clips only off the news
    motion = o.get("motion", "auto")
    if motion != "none":
        add_motion(job, scenes, visuals, motion, news, warnings)
    # 4. narration
    narr, voice = {}, None
    if TTS_ENABLED and o["voice"] != "none":
        voice = TTS.pick(o["voice"], lang)
        if not voice:
            warnings.append("no voice is installed for %s; the video has captions only" % ST.lang_info(lang)[0])
        elif o["voice"] not in ("auto", voice):
            warnings.append("voice %s is not available; used %s" % (o["voice"], voice))
    if voice:
        for i, sc in enumerate(scenes):
            text = (sc.get("narration") or "").strip()
            if not text:
                continue
            path = os.path.join(d, "voice", h(text, voice, sb.get("tone")) + ".wav")
            if not os.path.exists(path):
                step(job, "recording narration %d/%d" % (i + 1, n), 0.45 + 0.17 * i / n)
                try:
                    TTS.speak(text, voice, path, MEDIA, sb.get("tone"), MEDIA_DEVICE, log)
                except Exception as e:
                    log.error("tts failed\n%s", traceback.format_exc())
                    warnings.append("narration failed (%s); the video has captions only" % str(e)[:200])
                    narr = {}
                    break
            narr[i] = (path, M.duration(path))
    if voice and TTS.VOICES.get(voice, {}).get("engine") == "svara":
        TTS.svara_unload()                              # free the GPU for music and rendering
        MEDIA.unload("-")
    spoken, room = sum(x[1] for x in narr.values()), 0.8 * o["length"]
    if narr and spoken > room * 1.04:                   # too long for the chosen length: speak a little faster
        f = min(1.25, spoken / room)
        log.info("%s: narration %.1fs for %ss; speeding up %.2fx", job["id"], spoken, o["length"], f)
        try:
            for i, (path, _) in list(narr.items()):
                fast = path[:-4] + "_x%.2f.wav" % f
                narr[i] = (fast, M.duration(fast) if os.path.exists(fast) else M.tempo(path, fast, f))
        except Exception as e:
            log.warning("speed-up failed: %s", e)
    save(job, voice=voice)
    # 5. timing
    wps = ST.WPS.get(lang, 2.3)
    nds = [narr[i][1] if i in narr else len((sc.get("narration") or "").split()) / wps for i, sc in enumerate(scenes)]
    durs = [scene_duration(sc, nds[i], o["length"]) for i, sc in enumerate(scenes)]
    if sum(durs) < 0.8 * o["length"]:                   # short narration: let the pictures breathe, up to 1.6x
        f = min(1.6, 0.85 * o["length"] / sum(durs))
        durs = [round(x * f, 2) for x in durs]
    plan_scenes, t = [], 0.0
    for i, sc in enumerate(scenes):
        plan_scenes.append(dict(sc, dur=durs[i], narr_start=0.5, narr_dur=nds[i], photo=visuals.get(i), start=t))
        t += durs[i]
    total = round(t, 2)
    # 6. music
    music = None
    if o["music"] and MUSIC_MODEL:
        music = os.path.join(d, "music_%s.wav" % h(sb.get("music_prompt"), MUSIC_MODEL))
        if not os.path.exists(music):
            step(job, "composing music", 0.64)
            try:
                MEDIA.music(sb.get("music_prompt") or "light background music", music, 30)
            except Exception as e:
                log.error("music failed\n%s", traceback.format_exc())
                warnings.append("music failed: %s" % str(e)[:200])
                music = None
    if UNLOAD_AFTER:
        MEDIA.unload()
        K.stop_process("vs-parler")
    # 7. audio mix
    step(job, "mixing audio", 0.68)
    audio = os.path.join(d, "audio.wav")
    cuts = [ps["start"] for ps in plan_scenes[1:]] if o.get("sfx", True) else None
    if narr or music or cuts:
        M.mix(total, [(ps["start"] + ps["narr_start"], narr[i][0]) for i, ps in enumerate(plan_scenes) if i in narr],
              music, audio, sb.get("tone", "neutral"), cuts)
    elif os.path.exists(audio):
        os.remove(audio)
    # 8. render
    story = dict(sb, language=lang)
    K.write_json(os.path.join(d, "plan.json"), {"story": story, "style": style, "captions": o["captions"], "fps": FPS,
                                                "quality": o["quality"], "scenes": plan_scenes})
    for f in FORMATS:
        for ext in (".mp4", ".jpg"):
            if os.path.exists(os.path.join(d, f + ext)):
                os.remove(os.path.join(d, f + ext))
    procs = {f: subprocess.Popen([sys.executable, os.path.join(HERE, "vs_render.py"), d, f], cwd=HERE,
                                 stdout=open(os.path.join(d, "render_%s.log" % f), "w"), stderr=subprocess.STDOUT)
             for f in o["formats"]}
    while any(p.poll() is None for p in procs.values()):
        prog = {f: K.read_json(os.path.join(d, f, "progress.json"), {}) or {} for f in procs}
        frac = sum((p.get("frames", 0) / max(1, p.get("total_frames", 1))) for p in prog.values()) / len(procs)
        eta = max([p.get("eta", 0) or 0 for p in prog.values()] + [0])
        save(job, state="running", step="rendering %s (about %d:%02d left)" % (" + ".join(procs), eta // 60, eta % 60),
             progress=round(0.7 + 0.3 * frac, 3), render=prog)
        time.sleep(2)
    errors = []
    for f, p in procs.items():
        pr = K.read_json(os.path.join(d, f, "progress.json"), {}) or {}
        if p.returncode != 0 or pr.get("state") != "done":
            errors.append("%s: %s" % (f, pr.get("error") or K.tail(os.path.join(d, "render_%s.log" % f), 5)))
    if errors:
        raise RuntimeError("rendering failed; " + " | ".join(errors))
    save(job, state="done", step="done", progress=1.0, duration=total, warnings=warnings, seconds=round(time.time() - t0),
         scene_images={str(k): visuals[i]["path"] for k, i in enumerate(i for i, sc in enumerate(scenes) if not sc.get("inserted"))
                       if i in visuals},
         render={f: K.read_json(os.path.join(d, f, "progress.json"), {}) for f in procs})
    log.info("%s done in %ds", job["id"], time.time() - t0)


def worker():
    while True:
        with WAKE:
            while not QUEUE:
                WAKE.wait(30)
            jid = QUEUE.pop(0)
        try:
            job = load(jid)
        except K.HTTPError:
            continue
        try:
            run_job(job)
        except Exception as e:
            log.error("job %s failed\n%s", jid, traceback.format_exc())
            save(job, state="error", step="failed", error="%s: %s" % (type(e).__name__, str(e)[:800]))


def wait(jid, seconds):
    t = time.time()
    while True:
        job = load(jid)
        if job["state"] not in ("queued", "running") or time.time() - t >= max(0, min(MAX_WAIT, float(seconds or 0))):
            return job
        time.sleep(1.5)


ASSET_LIMIT = 200 << 20


def _site_name(url):
    return (urllib.parse.urlparse(url).hostname or "").lower().replace("www.", "")


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": SRC.UA, "Accept": "image/*,video/*,text/html,*/*;q=0.5"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        ctype = r.headers.get("Content-Type", "")
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            if f.tell() > ASSET_LIMIT:
                raise ValueError("that file is larger than 200 MB")
        return ctype, r.geturl()


def fetch_link_asset(url, dest):
    """Downloads the picture or video behind a link and works out its credit. Handles a direct image / video file,
    a video page yt-dlp knows (YouTube, Vimeo, Facebook... first 10 s), or any web page (its og:video or og:image,
    credited to the page's site name). Returns (credit, source page URL)."""
    if not re.match(r"^https?://", url or ""):
        raise ValueError("paste a full link starting with http:// or https://")
    if SRC.YT.match(url) or re.search(r"(vimeo\.com|facebook\.com/.+/videos|fb\.watch|tiktok\.com|x\.com/.+/status|"
                                      r"twitter\.com/.+/status|instagram\.com/(reel|p)/)", url, re.I):
        if not shutil.which("yt-dlp"):
            raise ValueError("video links need yt-dlp (installed by the notebook's install cell)")
        info = os.path.join(os.path.dirname(dest), "ytdlp_%s" % uuid.uuid4().hex[:6])
        p = subprocess.run(["yt-dlp", "--no-playlist", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b",
                            "--download-sections", "*0-10", "--force-keyframes-at-cuts", "--print-json", "--no-progress",
                            "-o", info + ".%(ext)s", url], capture_output=True, text=True, timeout=600)
        files = [f for f in os.listdir(os.path.dirname(dest)) if f.startswith(os.path.basename(info)) and not f.endswith(".part")]
        if p.returncode != 0 or not files:
            raise ValueError("could not get that video: %s" % ((p.stderr.strip().splitlines() or ["?"])[-1])[:200])
        shutil.move(os.path.join(os.path.dirname(dest), files[0]), dest)
        meta = {}
        try:
            meta = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        who = meta.get("channel") or meta.get("uploader") or ""
        site = meta.get("extractor_key") or _site_name(url)
        return (" · ".join(x for x in (who, site) if x) or _site_name(url)), meta.get("webpage_url") or url
    ctype, final = _download(url, dest)
    if "html" not in ctype.lower():
        return _site_name(final), final                 # a direct picture or video file
    raw = open(dest, "rb").read().decode("utf-8", "replace")
    pg = SRC._Page(final)
    pg.feed(raw)
    pg.close()
    media = pg.meta.get("og:video:secure_url") or pg.meta.get("og:video:url") or pg.meta.get("og:video") or ""
    if media and not re.search(r"\.(mp4|webm|mov)(\?|$)", media, re.I):
        media = ""                                      # an embed page, not a file
    media = media or pg.meta.get("og:image") or pg.meta.get("twitter:image") or ""
    if not media:
        imgs = SRC.harvest_images(pg, final)
        media = imgs[0]["url"] if imgs else ""
    if not media:
        raise ValueError("that page has no picture or video to use")
    _download(urllib.parse.urljoin(final, media), dest)
    return pg.meta.get("og:site_name") or _site_name(final), final


def add_asset(jid, data_iter=None, url=None, name=""):
    """Adds the user's own picture or video clip, or one behind a link (credited automatically), to a job's photos;
    returns its index. Videos keep up to 10 s, cut into 24 fps frames that the scene plays exactly."""
    d = jdir(jid)
    job = load(jid)
    if job["state"] in ("queued", "running"):
        raise ValueError("the job is busy (%s); wait for it to finish" % job["state"])
    pdir = os.path.join(d, "photos")
    os.makedirs(pdir, exist_ok=True)
    tmp = os.path.join(pdir, "upload_%s.tmp" % uuid.uuid4().hex[:8])
    credit, source = "", ""
    if url:
        try:
            credit, source = fetch_link_asset(url.strip(), tmp)
        except urllib.error.URLError as e:
            raise ValueError("could not open that link (%s)" % getattr(e, "reason", e))
    else:
        with open(tmp, "wb") as f:
            for chunk in data_iter:
                f.write(chunk)
    photos = K.read_json(os.path.join(d, "photos.json"), []) or []
    k = len(photos)
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(tmp)).convert("RGB")
        im.thumbnail((2400, 2400))
        path = os.path.join(pdir, "upload_%02d.jpg" % k)
        im.save(path, "JPEG", quality=92)
        entry = {"path": path, "w": im.size[0], "h": im.size[1], "caption": "", "alt": name, "url": source,
                 "kind": "link" if url else "upload", "credit": credit}
        os.remove(tmp)
    except Exception:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                "-of", "csv=p=0", tmp], capture_output=True, text=True)
        if probe.returncode != 0 or "," not in probe.stdout:
            os.remove(tmp)
            raise ValueError("that file is not a picture or a video ffmpeg can read")
        clip = os.path.join(pdir, "clip_%02d.mp4" % k)
        fdir = os.path.join(pdir, "clip_%02d_frames" % k)
        os.makedirs(fdir, exist_ok=True)
        sz = "scale='if(gt(iw,ih),min(1600,iw),-2)':'if(gt(iw,ih),-2,min(1600,ih))'"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-t", "10", "-an", "-vf", sz, "-c:v", "libx264", "-preset",
                        "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", clip], check=True, timeout=600)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", "fps=24", "-q:v", "3",
                        os.path.join(fdir, "f%03d.jpg")], check=True, timeout=600)
        os.remove(tmp)
        frames = MO.frames_in(fdir)
        if not frames:
            raise ValueError("could not read frames from that video")
        from PIL import Image
        poster = os.path.join(pdir, "clip_%02d.jpg" % k)
        shutil.copy(frames[min(len(frames) - 1, 12)], poster)
        w, hh = Image.open(poster).size
        entry = {"path": poster, "w": w, "h": hh, "caption": "", "alt": name, "url": source, "kind": "video",
                 "clip": clip, "frames_dir": fdir, "fps": 24, "credit": credit}
    photos.append(entry)
    K.write_json(os.path.join(d, "photos.json"), photos)
    return {"index": k, "kind": entry["kind"], "photos": public(load(jid), full=True)["photos"]}


def set_storyboard(jid, sb, render=False):
    d = jdir(jid)
    job = load(jid)
    if job["state"] in ("queued", "running"):
        raise ValueError("the job is busy (%s); wait for it to finish" % job["state"])
    src = K.read_json(os.path.join(d, "source.json")) or {"kind": "prompt", "title": job["title"], "text": "", "facts": {}}
    photos = K.read_json(os.path.join(d, "photos.json"), []) or []
    if isinstance(sb, str):
        sb = json.loads(sb)
    lang = job.get("language") or src.get("language") or "en"
    clean = ST.validate_user_storyboard(sb, src, job["options"]["length"], lang, len(photos))
    K.write_json(os.path.join(d, "storyboard.json"), clean)
    save(job, title=clean["title"][:100])
    if render:
        enqueue(job, approved=True)
    return clean


def rerender(jid, b):
    job = load(jid)
    if job["state"] in ("queued", "running"):
        raise ValueError("the job is already %s" % job["state"])
    b = b or {}
    opts = parse_options(b, job["options"])
    if b.get("rewrite") or (b.get("language") and b["language"] != job["options"].get("language")):
        p = os.path.join(jdir(jid), "storyboard.json")
        if os.path.exists(p):
            os.remove(p)
    return enqueue(job, options=opts, approved=not (b.get("rewrite") and opts["review"]))


def delete(jid):
    job = load(jid)
    if job["state"] == "running":
        raise ValueError("the job is running; wait for it to finish")
    with LOCK:
        if jid in QUEUE:
            QUEUE.remove(jid)
    shutil.rmtree(jdir(jid))
    return {"deleted": jid}


def info():
    jobs = all_jobs()
    return {"styles": list(STYLE_CHOICES), "voices": TTS.voice_options(), "lengths": list(ST.LENGTHS),
            "languages": {k: (ST.lang_info(k)[1] if k != "auto" else "Same as the link") for k in LANG_CHOICES},
            "formats": list(FORMATS), "qualities": ["1080p", "720p"], "llm": LLM_MODEL, "image_model": IMAGE_MODEL or None,
            "music_model": MUSIC_MODEL or None, "tts": TTS_ENABLED, "gpus": K.gpu_info(),
            "motion": {"choices": list(MOTION_CHOICES), "parallax": DEPTH_ON, "cutout": CUTOUT_ON, "maps": MAPS_ON,
                       "ai": I2V_MODEL or None},
            "queue": {s: sum(1 for j in jobs if j["state"] == s) for s in ("queued", "running", "review", "done", "error")}}


# ====================================================================== app
def build_app():
    app = K.App("Video Studio", password=os.environ.get("APP_PASSWORD"), log=log, max_body=8 << 20, instructions=(
        "Makes narrated videos with music from any link (news article in Nepali or English, GitHub repo, YouTube, PDF) "
        "or an idea, as landscape (16:9) and reel (9:16) MP4s. It uses the link's own photos, speaks the link's "
        "language (Nepali and Hindi via Indic Parler-TTS) and picks a style from the content (broadcast for news). "
        "make_video queues a job; poll get_job until state is 'done' (a few minutes), then download files.landscape.mp4 / "
        "files.reel.mp4 with the same password. With review=true it stops at state 'review': read get_storyboard, edit "
        "it with update_storyboard(render=true) or call render."))
    app.page("/", os.path.join(HERE, "vs_ui.html"))
    app.static("/static/", HERE)

    @app.route("GET", "/api/info")
    def _info(req):
        return info()

    @app.route("GET", "/api/jobs")
    def _jobs(req):
        return {"jobs": [public(j) for j in all_jobs()]}

    @app.route("POST", "/api/jobs")
    def _create(req):
        b = req.json()
        return public(create(b.get("source"), parse_options(b)))

    @app.route("GET", r"/api/jobs/(?P<j>[\w-]+)")
    def _job(req):
        return public(wait(req.params["j"], req.arg("wait", 0, float)), full=True)

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/storyboard")
    def _sb(req):
        b = req.json()
        return {"storyboard": set_storyboard(req.params["j"], b.get("storyboard"), bool(b.get("render")))}

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/render")
    def _render(req):
        return public(rerender(req.params["j"], req.json()))

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/assets")
    def _asset(req):
        """The raw picture or video file as the body (any image type, or mp4 / mov / webm up to 200 MB), or
        ?url=<link> to fetch it from a link with automatic credit."""
        if req.arg("url"):
            return add_asset(req.params["j"], url=req.arg("url"))
        return add_asset(req.params["j"], req.iter_body(limit=ASSET_LIMIT), name=req.arg("name", "") or "")

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/delete")
    def _delete(req):
        return delete(req.params["j"])

    @app.route("GET", r"/api/jobs/(?P<j>[\w-]+)/files/(?P<f>(?:images/|photos/)?[\w.-]+)")
    def _file(req):
        d = jdir(req.params["j"])
        f = req.params["f"]
        job = load(req.params["j"])
        name = "%s-%s" % (K.safe_name(job.get("title"), 50) or job["id"], f) if f.endswith(".mp4") else None
        return K.FileResponse(os.path.join(d, f), name=name if req.arg("download") else None)

    # ---------------------------------------------------------------- MCP tools
    opt_schema = {
        "motion": {"type": "string", "enum": list(MOTION_CHOICES), "default": "auto",
                   "description": "auto: 2.5D depth parallax on photos and subject lift-off on headlines; ai: AI "
                                  "image-to-video clips (non-news only, slow; needs I2V_MODEL); parallax; none"},
        "map": {"type": "boolean", "default": True, "description": "map fly-in to the story's place (news, 30 s+)"},
        "sfx": {"type": "boolean", "default": True, "description": "soft transition sounds"},
        "style": {"type": "string", "enum": list(STYLE_CHOICES), "default": "auto",
                  "description": "auto picks broadcast for news, midnight for tech, documentary for stories"},
        "length": {"type": "integer", "enum": list(ST.LENGTHS), "default": 60, "description": "seconds"},
        "language": {"type": "string", "enum": list(LANG_CHOICES), "default": "auto", "description": "auto = the link's language"},
        "voice": {"type": "string", "enum": ["auto", "none"] + list(TTS.VOICES), "default": "auto"},
        "music": {"type": "boolean", "default": True}, "captions": {"type": "boolean", "default": True},
        "formats": {"type": "array", "items": {"type": "string", "enum": list(FORMATS)}, "default": list(FORMATS)},
        "quality": {"type": "string", "enum": ["1080p", "720p"], "default": DEFAULT_QUALITY}}

    @app.tool("make_video", "Make a narrated video with music from a link (news article, GitHub repo, YouTube, PDF) or a "
              "description. Text after a link is passed to the scriptwriter as instructions. Returns a job; poll get_job.",
              dict({"source": {"type": "string"}, "review": {"type": "boolean", "default": False,
                                                             "description": "stop after the script so it can be edited"},
                    "wait_seconds": {"type": "number", "default": 0}}, **opt_schema), ["source"])
    def t_make(source, wait_seconds=0, **opts):
        job = create(source, parse_options(opts))
        return public(wait(job["id"], wait_seconds) if wait_seconds else job)

    @app.tool("get_job", "Job state (queued, running, review, done, error), progress and download links. "
              "wait_seconds (max 85) waits for it to finish.", {
                  "job_id": {"type": "string"}, "wait_seconds": {"type": "number", "default": 0}}, ["job_id"])
    def t_get(job_id, wait_seconds=0):
        return public(wait(job_id, wait_seconds))

    @app.tool("get_storyboard", "The job's script: title, tagline, language, category, tone, music prompt and scenes "
              "(layout, heading, kicker, bullets, narration, photo index, image_prompt, code, quote, stats), plus the "
              "photos found on the link.", {"job_id": {"type": "string"}}, ["job_id"])
    def t_sb(job_id):
        sb = K.read_json(os.path.join(jdir(job_id), "storyboard.json"))
        if not sb:
            raise ValueError("the script is not written yet (state: %s)" % load(job_id)["state"])
        return dict(sb, photos=public(load(job_id), full=True)["photos"])

    @app.tool("update_storyboard", "Replace the job's script (same shape as get_storyboard) and optionally render it. "
              "Unchanged scenes reuse their photos and narration. Each scene's photo is an index into the job's photos "
              "(get_storyboard lists them, add_asset adds more), -1 to let the studio pick, or -2 for no picture.", {
                  "job_id": {"type": "string"}, "storyboard": {"type": "object"},
                  "render": {"type": "boolean", "default": True}}, ["job_id", "storyboard"])
    def t_update(job_id, storyboard, render=True):
        storyboard = {k: v for k, v in storyboard.items() if k != "photos"}
        return {"storyboard": set_storyboard(job_id, storyboard, render), "state": load(job_id)["state"]}

    @app.tool("add_asset", "Add a picture or a short video clip to a job's photos from a link, credited on screen "
              "automatically: a direct image/video file (credited to its site), a video page such as YouTube or Vimeo "
              "(first 10 s, credited to the channel), or any web page (its main picture or video, credited to the site "
              "name). Then use the returned index as a scene's photo in update_storyboard. Files from disk: POST the raw "
              "file to /api/jobs/<id>/assets.", {"job_id": {"type": "string"}, "url": {"type": "string"}}, ["job_id", "url"])
    def t_asset(job_id, url):
        return add_asset(job_id, url=url)

    @app.tool("render", "Render (or re-render) a job, optionally with new options. rewrite=true (or a new language) "
              "writes a new script first.", dict({"job_id": {"type": "string"}, "rewrite": {"type": "boolean", "default": False}},
                                                 **opt_schema), ["job_id"])
    def t_render(job_id, **b):
        return public(rerender(job_id, b))

    @app.tool("list_jobs", "Recent jobs, newest first.", {"limit": {"type": "integer", "default": 20}})
    def t_list(limit=20):
        return [public(j) for j in all_jobs()[:max(1, int(limit))]]

    @app.tool("delete_job", "Delete a job and its files.", {"job_id": {"type": "string"}}, ["job_id"])
    def t_delete(job_id):
        return delete(job_id)

    @app.tool("list_options", "Styles, languages, installed voices, lengths, formats and the models in use.")
    def t_options():
        return info()

    return app


def resume():
    for j in all_jobs():
        if j["state"] in ("running", "queued"):
            QUEUE.append(j["id"])
            save(j, state="queued", step="waiting (restarted)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    os.makedirs(JOBS, exist_ok=True)
    resume()
    threading.Thread(target=worker, daemon=True).start()
    build_app().serve_forever(ap.parse_args().port)
