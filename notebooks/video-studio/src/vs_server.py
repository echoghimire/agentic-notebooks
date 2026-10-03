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
import io
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
import vs_postiz as PZ
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
MUSIC_MODEL = os.environ.get("MUSIC_MODEL", "synth")              # synth = built-in music, always monetisation-safe
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
            "duration", "render", "seconds", "language", "style", "voice", "posts")
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
                          "needs_credit": bool(p.get("needs_credit")),
                          "video": "/api/jobs/%s/files/photos/%s" % (job["id"], os.path.basename(p["clip"])) if p.get("clip") else None}
                         for i, p in enumerate(photos)]
        out["images"] = {str(i): "/api/jobs/%s/files/%s/%s" % (job["id"], os.path.basename(os.path.dirname(p)), os.path.basename(p))
                         for i, p in (job.get("scene_images") or {}).items() if p}
    return out


def parse_options(b, base=None):
    o = dict(base or {"style": "auto", "length": 60, "voice": "auto", "language": "auto", "music": bool(MUSIC_MODEL),
                      "captions": True, "formats": list(FORMATS), "quality": DEFAULT_QUALITY, "review": False,
                      "motion": "auto", "map": True, "sfx": True, "brand": True, "post_to": [], "post_at": "",
                      "music_level": "medium", "clip_sound": True, "footage": True,
                      "post_privacy": "public"})
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
    if b.get("post_to") is not None and b.get("post_to") != "":
        pt = b["post_to"] if isinstance(b["post_to"], list) else [x.strip() for x in str(b["post_to"]).split(",") if x.strip()]
        o["post_to"] = [x if isinstance(x, dict) else {"id": str(x)} for x in pt][:20]
    if b.get("post_at") is not None:
        o["post_at"] = parse_when(b["post_at"])
    if b.get("music_level"):
        if b["music_level"] not in ("low", "medium", "high"):
            raise ValueError("music_level must be low, medium or high")
        o["music_level"] = b["music_level"]
    if b.get("post_privacy"):
        if b["post_privacy"] not in ("public", "unlisted", "private"):
            raise ValueError("post_privacy must be public, unlisted or private")
        o["post_privacy"] = b["post_privacy"]
    for k in ("music", "captions", "review", "map", "sfx", "brand", "clip_sound", "footage"):
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


# ---------------------------------------------------------------------- pause / resume / delete / reorder
STOP = {}                                               # job id -> "pause" | "delete", read at every step and while rendering


class Stopped(Exception):
    pass


def control(jid, action, to=None):
    """pause (keeps everything made so far; resume continues), resume, delete, or move a waiting job in the queue."""
    job = load(jid)
    st = job["state"]
    if action == "pause":
        with WAKE:
            if jid in QUEUE:
                QUEUE.remove(jid)
                save(job, state="paused", step="paused before it started")
                return public(job)
        if st == "running":
            STOP[jid] = "pause"
            save(job, step="pausing…")
            return public(job)
        raise ValueError("only a waiting or running video can be paused (this one is %s)" % st)
    if action == "resume":
        if st not in ("paused", "error", "stopped"):
            raise ValueError("only a paused or failed video can be resumed (this one is %s)" % st)
        return public(enqueue(job))
    if action == "delete":
        if st == "running":
            STOP[jid] = "delete"
            save(job, step="stopping, then deleting…")
            return {"deleting": jid}
        return delete(jid)
    if action == "move":
        with WAKE:
            if jid not in QUEUE:
                raise ValueError("only a waiting video can be moved")
            i = QUEUE.index(jid)
            QUEUE.remove(jid)
            j = {"top": 0, "up": max(0, i - 1), "down": i + 1, "bottom": len(QUEUE)}.get(to)
            if j is None:
                raise ValueError("to must be top, up, down or bottom")
            QUEUE.insert(min(j, len(QUEUE)), jid)
        return {"queue": list(QUEUE)}
    raise ValueError("action must be pause, resume, delete or move")


def check_stop(job):
    if STOP.get(job["id"]):
        raise Stopped(STOP[job["id"]])


def kill_tree(p):
    """Stops a renderer and its Chromium / ffmpeg children (they run in their own process group)."""
    import signal
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=5)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def enqueue(job, **kw):
    STOP.pop(job["id"], None)
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
    check_stop(job)
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


def footage_for(src, d, n):
    """Downloads the source video once (720p at most, the first 20 minutes) and cuts n clips of up to 7 s, each
    starting on a shot change and spread over the video, with their sound; credited to the channel / site."""
    video = os.path.join(d, "source_video.mp4")
    if not os.path.exists(video):
        if src["kind"] == "youtube":
            p = subprocess.run(["yt-dlp", "--no-playlist", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/bv*+ba/b",
                                "--merge-output-format", "mp4", "--download-sections", "*0-1200", "--no-progress",
                                "-o", video, src["video_url"]], capture_output=True, text=True, timeout=1800)
            if p.returncode != 0 or not os.path.exists(video):
                raise RuntimeError((p.stderr.strip().splitlines() or ["download failed"])[-1][:200])
        else:
            tmp = video + ".part"
            _download(src["video_url"], tmp)
            os.replace(tmp, video)
    starts, secs = MO.pick_segments(video, n, 7.0)
    out = []
    for k, st in enumerate(starts):
        e = MO.make_clip(video, os.path.join(d, "photos", "foot_%02d" % k), st, secs)
        e.update(kind="video", credit=src.get("credit") or "", caption="", alt="footage at %d:%02d" % divmod(int(st), 60),
                 url=src.get("url") or "", footage=True)
        out.append(e)
    log.info("cut %d clips from the source video", len(out))
    return out


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
        foot = []
        if src.get("video_url") and o.get("footage", True):
            step(job, "downloading the video", 0.06)
            try:
                foot = footage_for(src, d, ST.LENGTHS.get(o["length"], (4, 5))[1])
            except Stopped:
                raise
            except Exception as e:
                log.error("footage failed\n%s", traceback.format_exc())
                warnings.append("the video's own footage could not be used (%s); the thumbnail is used instead" % str(e)[:160])
        if photos and DESCRIBE_PHOTOS and LLM_MODEL:
            step(job, "looking at the photos", 0.07)
            photos = M.describe_photos(photos, OLLAMA_URL, LLM_MODEL, LLM_KEEP_ALIVE, log,
                                       topic=src["title"] + ". " + (src.get("description") or ""), keep=8)
        photos = photos[:8]
        if foot:                                        # the video's own shots, described so each scene gets a fitting one
            if DESCRIBE_PHOTOS and LLM_MODEL:
                foot = M.describe_photos(foot, OLLAMA_URL, LLM_MODEL, LLM_KEEP_ALIVE, log, topic=src["title"], keep=len(foot))
            photos = photos[:1] + foot + photos[1:]     # thumbnail first (headline), then the footage
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
            if ph.get("frames_dir"):                    # a video clip: its frames, and its own sound if it has one
                visuals[i].update(frames=MO.frames_in(ph["frames_dir"]), fps=ph.get("fps", 24),
                                  audio=ph.get("audio") if ph.get("audio") and os.path.exists(ph["audio"]) else None)
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
    if narr and spoken > room * 1.04:                   # too long: a touch faster, never so fast it slurs
        f = min(1.08, spoken / room)
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
                MEDIA.music(sb.get("music_prompt") or "light background music", music, 30, tone=sb.get("tone") or "neutral")
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
    ambience = [(ps["start"], ps["dur"], ps["photo"]["audio"], i in narr) for i, ps in enumerate(plan_scenes)
                if o.get("clip_sound", True) and ps.get("photo") and ps["photo"].get("audio") and ps["photo"].get("frames")]
    if narr or music or cuts or ambience:
        M.mix(total, [(ps["start"] + ps["narr_start"], narr[i][0]) for i, ps in enumerate(plan_scenes) if i in narr],
              music, audio, sb.get("tone", "neutral"), cuts, o.get("music_level", "medium"), ambience)
    elif os.path.exists(audio):
        os.remove(audio)
    # 8. render
    story = dict(sb, language=lang, brand=brand_for_render(o))
    K.write_json(os.path.join(d, "plan.json"), {"story": story, "style": style, "captions": o["captions"], "fps": FPS,
                                                "quality": o["quality"], "scenes": plan_scenes})
    for f in FORMATS:
        for ext in (".mp4", ".jpg"):
            if os.path.exists(os.path.join(d, f + ext)):
                os.remove(os.path.join(d, f + ext))
    procs = {f: subprocess.Popen([sys.executable, os.path.join(HERE, "vs_render.py"), d, f], cwd=HERE,
                                 stdout=open(os.path.join(d, "render_%s.log" % f), "w"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
             for f in o["formats"]}
    while any(p.poll() is None for p in procs.values()):
        if STOP.get(job["id"]):                         # pause / delete: stop the renderers now
            for p in procs.values():
                kill_tree(p)
            check_stop(job)
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
    if o.get("post_to"):                               # auto-post the finished video
        step(job, "posting to %d channel(s)" % len(o["post_to"]), 0.99)
        try:
            post_job(job["id"], o["post_to"], None, o.get("post_at") or "", o.get("post_privacy", "public"))
            job = load(job["id"])
        except Exception as e:
            log.error("auto-post failed\n%s", traceback.format_exc())
            warnings.append("posting failed: %s" % str(e)[:200])
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
            STOP.pop(jid, None)
        except Stopped as e:
            STOP.pop(jid, None)
            if str(e) == "delete":
                shutil.rmtree(os.path.join(JOBS, jid), ignore_errors=True)
                log.info("%s stopped and deleted", jid)
            else:
                save(job, state="paused", step="paused: resume continues where it stopped")
                log.info("%s paused", jid)
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


# Storage and CDN hosts say nothing about who took a picture (a Google Photos "copy link" gives
# lh3.googleusercontent.com): such links get no automatic credit, and the editor asks for one instead.
CDN_HOSTS = re.compile(r"(^|\.)(googleusercontent\.com|ggpht\.com|gstatic\.com|googleapis\.com|photos\.app\.goo\.gl|"
                       r"photos\.google\.com|drive\.google\.com|fbcdn\.net|cdninstagram\.com|twimg\.com|pinimg\.com|"
                       r"imgur\.com|cloudfront\.net|amazonaws\.com|akamaized\.net|wp\.com|blogspot\.com|bp\.blogspot\.com|"
                       r"dropboxusercontent\.com|dropbox\.com|1drv\.ms|sharepoint\.com|icloud\.com|cloudinary\.com|"
                       r"imagekit\.io|githubusercontent\.com|discordapp\.(com|net)|wixstatic\.com|squarespace-cdn\.com|"
                       r"shopify\.com|unsplash\.com)$", re.I)


def _site_name(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return "" if CDN_HOSTS.search(host) else host.replace("www.", "")


def _full_size(url):
    """Google Photos / Blogger links carry a size suffix (=w400-h300): ask for the original instead."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host.endswith(("googleusercontent.com", "ggpht.com")):
        return re.sub(r"=[\w-]*$", "", url) + "=s0"
    return url


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
    ctype, final = _download(_full_size(url), dest)
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
    _download(_full_size(urllib.parse.urljoin(final, media)), dest)
    site = pg.meta.get("og:site_name") or ""
    if not site or CDN_HOSTS.search((urllib.parse.urlparse(final).hostname or "").lower()):
        site = _site_name(final)                        # a Google Photos album page names no photographer
    return site, final


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
                 "kind": "link" if url else "upload", "credit": credit, "needs_credit": bool(url) and not credit}
        os.remove(tmp)
    except Exception:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                "-of", "csv=p=0", tmp], capture_output=True, text=True)
        if probe.returncode != 0 or "," not in probe.stdout:
            os.remove(tmp)
            raise ValueError("that file is not a picture or a video ffmpeg can read")
        entry = MO.make_clip(tmp, os.path.join(pdir, "clip_%02d" % k), 0, 10)
        os.remove(tmp)
        entry.update(caption="", alt=name, url=source, kind="video", credit=credit, needs_credit=bool(url) and not credit)
    photos.append(entry)
    K.write_json(os.path.join(d, "photos.json"), photos)
    return {"index": k, "kind": entry["kind"], "photos": public(load(jid), full=True)["photos"]}


# ====================================================================== brand kit
BRAND_DIR = os.path.join(WORK, "brand")
BRAND_FILE = os.path.join(BRAND_DIR, "brand.json")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def brand():
    """The channel's brand: name, handle, colour, logo; seeded once from the notebook settings (BRAND_*)."""
    b = K.read_json(BRAND_FILE)
    if b is None:
        b = {"name": os.environ.get("BRAND_NAME", ""), "handle": os.environ.get("BRAND_HANDLE", ""),
             "color": os.environ.get("BRAND_COLOR", "") if HEX.match(os.environ.get("BRAND_COLOR", "")) else "",
             "corner": True, "outro": True, "logo": ""}
        os.makedirs(BRAND_DIR, exist_ok=True)
        K.write_json(BRAND_FILE, b)
        src = os.environ.get("BRAND_LOGO", "")
        if src:
            try:
                set_logo(url=src) if re.match(r"^https?://", src) else set_logo(data=open(src, "rb").read())
                b = K.read_json(BRAND_FILE)
            except Exception as e:
                log.warning("brand logo %s not loaded: %s", src, e)
    return b


def brand_public():
    b = dict(brand())
    b["logo_url"] = "/api/brand/logo?v=%d" % int(os.path.getmtime(b["logo"])) if b.get("logo") and os.path.exists(b["logo"]) else None
    b.pop("logo", None)
    return b


def set_brand(**kw):
    b = brand()
    for k in ("name", "handle"):
        if kw.get(k) is not None:
            b[k] = re.sub(r"\s+", " ", str(kw[k])).strip()[:60]
    if kw.get("color") is not None:
        c = str(kw["color"]).strip()
        if c and not HEX.match(c):
            raise ValueError("color must look like #e1261c (or be empty to use the style's colour)")
        b["color"] = c
    for k in ("corner", "outro"):
        if kw.get(k) is not None:
            b[k] = kw[k] if isinstance(kw[k], bool) else str(kw[k]).lower() in ("1", "true", "yes", "on")
    K.write_json(BRAND_FILE, b)
    return brand_public()


def set_logo(data=None, url=None, remove=False):
    """Stores the logo as a transparent PNG (any picture format; SVG is not supported)."""
    from PIL import Image, ImageOps
    b = K.read_json(BRAND_FILE) or brand()
    os.makedirs(BRAND_DIR, exist_ok=True)
    if remove:
        b["logo"] = ""
    else:
        if url:
            req = urllib.request.Request(_full_size(url), headers={"User-Agent": SRC.UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(20 << 20)
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")
        except Exception:
            raise ValueError("the logo must be a picture (PNG with transparency works best; SVG is not supported)")
        bbox = im.getchannel("A").getbbox()
        if bbox:
            im = im.crop(bbox)                          # trim empty margins so the logo sits tight
        im.thumbnail((900, 900))
        path = os.path.join(BRAND_DIR, "logo_%d.png" % int(time.time()))
        im.save(path)
        b["logo"] = path
    K.write_json(BRAND_FILE, b)
    return brand_public()


def brand_for_render(o):
    if not o.get("brand", True):
        return None
    b = brand()
    if not (b.get("name") or b.get("logo")):
        return None
    return {"name": b.get("name", ""), "handle": b.get("handle", ""), "color": b.get("color", ""),
            "logo": b["logo"] if b.get("logo") and os.path.exists(b["logo"]) else "",
            "corner": b.get("corner", True), "outro": b.get("outro", True)}


# ====================================================================== posting (Postiz)
def parse_when(v):
    """'' = now; otherwise an ISO date-time ('2026-10-04T18:30', with or without a zone; no zone = UTC)."""
    v = str(v or "").strip()
    if not v:
        return ""
    import datetime
    try:
        d = datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("post_at must be a date-time like 2026-10-04T18:30 (UTC) or 2026-10-04T18:30+05:45")
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    d = d.astimezone(datetime.timezone.utc)
    if d.timestamp() < time.time() - 60:
        raise ValueError("that time is in the past")
    return d.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def job_caption(jid):
    d = jdir(jid)
    sb = K.read_json(os.path.join(d, "storyboard.json")) or {}
    src = K.read_json(os.path.join(d, "source.json")) or {}
    return PZ.caption(dict(sb, language=load(jid).get("language") or sb.get("language")), src, brand())


def post_job(jid, targets, text=None, when="", privacy="public"):
    """Posts (or schedules) the finished video to Postiz channels: reels to TikTok / Instagram / YouTube Shorts,
    landscape elsewhere, unless a target names its format."""
    job = load(jid)
    d = jdir(jid)
    files = {f: os.path.join(d, f + ".mp4") for f in FORMATS if os.path.exists(os.path.join(d, f + ".mp4"))}
    if not files:
        raise ValueError("the video is not rendered yet")
    known = {c["id"]: c for c in PZ.channels()}
    picked = []
    for t in targets or []:
        t = t if isinstance(t, dict) else {"id": str(t)}
        c = known.get(t.get("id"))
        if not c:
            raise ValueError("unknown Postiz channel %r (list_channels shows the connected ones)" % t.get("id"))
        picked.append(dict(c, format=t.get("format") or c["format"]))
    sb = K.read_json(os.path.join(d, "storyboard.json")) or {}
    cache_path = os.path.join(d, "postiz_media.json")
    cache = {k: v for k, v in (K.read_json(cache_path, {}) or {}).items()
             if k in files and v.get("mtime") == int(os.path.getmtime(files[k]))}
    media = {k: {"id": v["id"], "path": v["path"]} for k, v in cache.items()}
    res, media = PZ.post(files, picked, text if text is not None else job_caption(jid), sb.get("title") or job["title"],
                         parse_when(when) if when else None, privacy, media)
    K.write_json(cache_path, {k: dict(v, mtime=int(os.path.getmtime(files[k]))) for k, v in media.items()})
    entry = {"at": time.time(), "when": when or "now", "channels": [{"name": c["name"], "identifier": c["identifier"],
                                                                     "format": c["format"]} for c in picked],
             "result": res if isinstance(res, (list, dict)) else str(res)}
    save(job, posts=(job.get("posts") or []) + [entry])
    log.info("%s posted to %s", jid, ", ".join(c["name"] for c in picked))
    return entry


def set_photo(jid, k, credit=None):
    """Changes what is credited on screen for one of the job's pictures ("" = no credit)."""
    d = jdir(jid)
    photos = K.read_json(os.path.join(d, "photos.json"), []) or []
    k = int(k)
    if not 0 <= k < len(photos):
        raise ValueError("there is no photo %d" % k)
    if credit is not None:
        photos[k]["credit"] = re.sub(r"\s+", " ", str(credit)).strip()[:80]
        photos[k].pop("needs_credit", None)
    K.write_json(os.path.join(d, "photos.json"), photos)
    return {"index": k, "photos": public(load(jid), full=True)["photos"]}


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
        return control(jid, "delete")
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
            "postiz": PZ.configured(), "brand": brand_public(),
            "motion": {"choices": list(MOTION_CHOICES), "parallax": DEPTH_ON, "cutout": CUTOUT_ON, "maps": MAPS_ON,
                       "ai": I2V_MODEL or None},
            "queue": dict({s: sum(1 for j in jobs if j["state"] == s) for s in ("queued", "running", "review", "done", "error", "paused")},
                          order=list(QUEUE))}


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

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/photos/(?P<k>\d+)")
    def _photo(req):
        return set_photo(req.params["j"], req.params["k"], req.json().get("credit"))

    @app.route("GET", "/api/postiz/channels")
    def _channels(req):
        if not PZ.configured():
            return {"configured": False, "channels": []}
        return {"configured": True, "channels": PZ.channels(bool(req.arg("refresh")))}

    @app.route("GET", r"/api/jobs/(?P<j>[\w-]+)/caption")
    def _caption(req):
        return {"caption": job_caption(req.params["j"])}

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/post")
    def _post(req):
        b = req.json()
        return post_job(req.params["j"], b.get("channels"), b.get("caption"), b.get("when") or "", b.get("privacy") or "public")

    @app.route("GET", "/api/brand")
    def _brand(req):
        return brand_public()

    @app.route("POST", "/api/brand")
    def _brand_set(req):
        return set_brand(**req.json())

    @app.route("GET", "/api/brand/logo")
    def _logo(req):
        b = brand()
        if not b.get("logo") or not os.path.exists(b["logo"]):
            raise K.HTTPError(404, "no logo yet")
        return K.FileResponse(b["logo"])

    @app.route("POST", "/api/brand/logo")
    def _logo_set(req):
        """The raw logo picture as the body, ?url=<link> to fetch it, or ?remove=1."""
        if req.arg("remove"):
            return set_logo(remove=True)
        if req.arg("url"):
            return set_logo(url=req.arg("url"))
        return set_logo(data=req.body())

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/(?P<a>pause|resume|move)")
    def _control(req):
        return control(req.params["j"], req.params["a"], req.json().get("to"))

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
        "brand": {"type": "boolean", "default": True, "description": "the channel brand (set_brand) on the corner and outro"},
        "music_level": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
        "clip_sound": {"type": "boolean", "default": True, "description": "keep video clips' own sound under the narration"},
        "footage": {"type": "boolean", "default": True, "description": "for a video link as the source, cut its real footage into the scenes"},
        "post_to": {"type": "array", "description": "Postiz channel ids to post the finished video to automatically"},
        "post_at": {"type": "string", "description": "schedule the auto-post (ISO date-time, no zone = UTC); empty = now"},
        "post_privacy": {"type": "string", "enum": ["public", "unlisted", "private"], "default": "public"},
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
              "name). Storage links (Google Photos, Drive, CDNs) carry no credit unless you pass one. Then use the returned index "
              "as a scene's photo in update_storyboard. Files from disk: POST the raw file to /api/jobs/<id>/assets.",
              {"job_id": {"type": "string"}, "url": {"type": "string"},
               "credit": {"type": "string", "description": "override the on-screen credit"}}, ["job_id", "url"])
    def t_asset(job_id, url, credit=None):
        out = add_asset(job_id, url=url)
        return set_photo(job_id, out["index"], credit) if credit is not None else out

    @app.tool("pause_job", "Pause a waiting or running video at once (renderers are stopped). Everything made so far is "
              "kept; resume_job continues from there.", {"job_id": {"type": "string"}}, ["job_id"])
    def t_pause(job_id):
        return control(job_id, "pause")

    @app.tool("resume_job", "Put a paused (or failed) video back in the queue; finished steps and scenes are reused.",
              {"job_id": {"type": "string"}}, ["job_id"])
    def t_resume(job_id):
        return control(job_id, "resume")

    @app.tool("move_job", "Move a waiting video in the queue.", {"job_id": {"type": "string"},
              "to": {"type": "string", "enum": ["top", "up", "down", "bottom"]}}, ["job_id", "to"])
    def t_move(job_id, to):
        return control(job_id, "move", to)

    @app.tool("list_channels", "Social channels connected in Postiz (needs the POSTIZ_API_KEY secret): id, name, "
              "identifier (youtube, tiktok, instagram, facebook, x, linkedin...) and the format posted there by default.", {}, [])
    def t_channels():
        return {"configured": PZ.configured(), "channels": PZ.channels() if PZ.configured() else []}

    @app.tool("post_video", "Post a finished video through Postiz now, or schedule it. channels: ids from list_channels "
              "(or objects {id, format: reel|landscape}); reels go to TikTok / Instagram / YouTube Shorts by default. "
              "caption defaults to title, summary, source credit, brand handle and hashtags. when: ISO date-time to "
              "schedule (no zone = UTC), empty = now. YouTube/TikTok privacy: public, unlisted or private.",
              {"job_id": {"type": "string"}, "channels": {"type": "array"}, "caption": {"type": "string"},
               "when": {"type": "string"}, "privacy": {"type": "string", "enum": ["public", "unlisted", "private"]}},
              ["job_id", "channels"])
    def t_post(job_id, channels, caption=None, when="", privacy="public"):
        return post_job(job_id, channels, caption, when, privacy)

    @app.tool("get_brand", "The channel brand used on every video (name, handle, colour, logo URL, corner mark, "
              "outro card).", {}, [])
    def t_get_brand():
        return brand_public()

    @app.tool("set_brand", "Set the channel brand: name and handle/website shown on the outro card and the corner mark, "
              "an accent colour (#rrggbb, empty = the style's colour), a logo from a URL, and whether to show the "
              "corner mark and the outro card. Applies to the next render.",
              {"name": {"type": "string"}, "handle": {"type": "string"}, "color": {"type": "string"},
               "logo_url": {"type": "string"}, "corner": {"type": "boolean"}, "outro": {"type": "boolean"}}, [])
    def t_set_brand(logo_url=None, **kw):
        if logo_url:
            set_logo(url=logo_url)
        return set_brand(**kw)

    @app.tool("set_photo_credit", "Set what is credited on screen for one of a job's photos (\"\" for none), e.g. when a "
              "Google Photos or CDN link could not be attributed automatically.",
              {"job_id": {"type": "string"}, "index": {"type": "integer"}, "credit": {"type": "string"}}, ["job_id", "index", "credit"])
    def t_credit(job_id, index, credit):
        return set_photo(job_id, index, credit)

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
