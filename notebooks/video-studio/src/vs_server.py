"""Video Studio: any link or idea -> narrated video with music, as landscape (16:9) and reel (9:16) MP4s.

Runs as its own process (started by the notebook with studio_http.spawn):
    APP_PASSWORD=... python vs_server.py --port 7860

Pipeline per job (one job at a time, files in WORK_DIR/jobs/<id>/):
    source (vs_source) -> storyboard (local LLM via Ollama, vs_story) -> [optional review/edit]
    -> scene images (SDXL) -> narration (Kokoro) -> music (MusicGen) -> mix (vs_media)
    -> render landscape + reel in parallel (vs_render, headless Chromium + ffmpeg)
Images and narration are cached by content, so editing one scene and re-rendering only redoes that scene.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

import studio_http as K
import vs_media as M
import vs_scenes as V
import vs_source as SRC
import vs_story as ST

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR", "/kaggle/working/video_studio")
JOBS = os.path.join(WORK, "jobs")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "10m")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "25"))
MUSIC_MODEL = os.environ.get("MUSIC_MODEL", "facebook/musicgen-small")
TTS_ENABLED = os.environ.get("TTS", "1") != "0"
MEDIA_DEVICE = os.environ.get("MEDIA_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cpu")
UNLOAD_AFTER = os.environ.get("UNLOAD_AFTER", "0") == "1"
DEFAULT_QUALITY = os.environ.get("RENDER_QUALITY", "1080p")
FPS = int(os.environ.get("RENDER_FPS", "30"))
FORMATS = ("landscape", "reel")
MAX_WAIT = 85
log = K.file_logger(os.environ.get("APP_LOG", "/kaggle/working/logs/video_studio.log"), "video")

LOCK = threading.RLock()
WAKE = threading.Condition(LOCK)
QUEUE = []
MEDIA = M.Media(IMAGE_MODEL, IMAGE_STEPS, MEDIA_DEVICE, MUSIC_MODEL, MEDIA_DEVICE, TTS_DEVICE, log)


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
    out = {}
    for f in ("landscape.mp4", "reel.mp4", "landscape.jpg", "reel.jpg"):
        if os.path.exists(os.path.join(d, f)):
            out[f] = "/api/jobs/%s/files/%s" % (jid, f)
    return out


def public(job, full=False):
    keys = ("id", "title", "state", "step", "progress", "created", "updated", "error", "warnings", "options",
            "duration", "render", "seconds")
    out = {k: job.get(k) for k in keys if k in job}
    out["files"] = files_of(job["id"])
    if full:
        sb = K.read_json(os.path.join(JOBS, job["id"], "storyboard.json"))
        if sb:
            out["storyboard"] = sb
        out["images"] = {str(i): "/api/jobs/%s/files/images/%s" % (job["id"], os.path.basename(p))
                         for i, p in (job.get("scene_images") or {}).items() if p}
    return out


def parse_options(b, base=None):
    o = dict(base or {"style": "midnight", "length": 60, "voice": "af_heart", "music": bool(MUSIC_MODEL),
                      "captions": True, "formats": list(FORMATS), "quality": DEFAULT_QUALITY, "review": False})
    if "style" in b and b["style"] is not None:
        if b["style"] not in V.STYLES:
            raise ValueError("style must be one of " + ", ".join(V.STYLES))
        o["style"] = b["style"]
    if b.get("length") not in (None, ""):
        L = int(b["length"])
        if L not in ST.LENGTHS:
            raise ValueError("length must be 30, 60 or 90 seconds")
        o["length"] = L
    if b.get("voice"):
        if b["voice"] not in M.VOICES and b["voice"] != "none":
            raise ValueError("voice must be one of %s, or none" % ", ".join(M.VOICES))
        o["voice"] = b["voice"]
    for k in ("music", "captions", "review"):
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
        raise ValueError("give a link (GitHub, article, YouTube, PDF...) or describe the video")
    if len(source) > 4000:
        raise ValueError("the description is longer than 4000 characters")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
    d = os.path.join(JOBS, jid)
    os.makedirs(os.path.join(d, "images"))
    os.makedirs(os.path.join(d, "voice"))
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


def scene_duration(sc, narr_dur):
    base = {"title": 3.4, "stats": 4.2, "outro": 3.6}.get(sc["layout"], 3.2)
    if sc["layout"] == "code":
        base = max(base, 2.6 + len(sc.get("code") or "") / 28.0)
    if sc["layout"] == "bullets":
        base = max(base, 1.4 + 0.45 * len(sc.get("bullets") or []))
    return round(max(base, narr_dur + 1.1), 2)


def step(job, name, frac):
    save(job, state="running", step=name, progress=round(frac, 3))
    log.info("%s: %s", job["id"], name)


def run_job(job):
    d = os.path.join(JOBS, job["id"])
    o = job["options"]
    t0 = time.time()
    warnings = []
    # 1. source
    src = K.read_json(os.path.join(d, "source.json"))
    if not src:
        step(job, "reading the source", 0.02)
        src = SRC.fetch_source(job["input"])
        K.write_json(os.path.join(d, "source.json"), src)
        save(job, title=src["title"][:100])
    # 2. storyboard
    sb = K.read_json(os.path.join(d, "storyboard.json"))
    if not sb:
        step(job, "writing the script (%s)" % LLM_MODEL, 0.06)
        try:
            sb, warn = ST.write_storyboard(src, o["length"], OLLAMA_URL, LLM_MODEL, LLM_KEEP_ALIVE, log)
        except Exception as e:
            sb, warn = ST.fallback(src, o["length"]), "script model unavailable (%s); used a plain script" % e
        if warn:
            warnings.append(warn)
        K.write_json(os.path.join(d, "storyboard.json"), sb)
        save(job, title=sb["title"][:100])
    if not job.get("approved"):
        save(job, state="review", step="review the script, then render", progress=0.15, warnings=warnings)
        return
    scenes = sb["scenes"]
    n = len(scenes)
    # 3. images
    images = {}
    if IMAGE_MODEL:
        for i, sc in enumerate(scenes):
            if sc["layout"] not in ("title", "bullets", "image", "outro") or not sc.get("image_prompt"):
                continue
            if sc["layout"] == "outro" and scenes[0].get("image_prompt"):
                sc = dict(sc, image_prompt=scenes[0]["image_prompt"])
            path = os.path.join(d, "images", h(sc["image_prompt"], o["style"], IMAGE_MODEL) + ".png")
            if not os.path.exists(path):
                step(job, "drawing image %d/%d" % (i + 1, n), 0.15 + 0.35 * i / n)
                try:
                    MEDIA.image(sc["image_prompt"], o["style"], path, seed=int(h(sc["image_prompt"])[:6], 16))
                except Exception as e:
                    log.error("image failed\n%s", traceback.format_exc())
                    warnings.append("image for scene %d failed: %s" % (i + 1, str(e)[:200]))
                    continue
            images[i] = path
    # 4. narration
    narr = {}
    if TTS_ENABLED and o["voice"] != "none":
        for i, sc in enumerate(scenes):
            text = (sc.get("narration") or "").strip()
            if not text:
                continue
            path = os.path.join(d, "voice", h(text, o["voice"]) + ".wav")
            if not os.path.exists(path):
                step(job, "recording narration %d/%d" % (i + 1, n), 0.5 + 0.12 * i / n)
                try:
                    MEDIA.tts(text, o["voice"], path)
                except Exception as e:
                    log.error("tts failed\n%s", traceback.format_exc())
                    warnings.append("narration failed (%s); the video has captions only" % str(e)[:200])
                    break
            narr[i] = (path, M.duration(path))
    # 5. timing
    plan_scenes, t = [], 0.0
    for i, sc in enumerate(scenes):
        nd = narr[i][1] if i in narr else len((sc.get("narration") or "").split()) / ST.WORDS_PER_SECOND
        dur = scene_duration(sc, nd)
        plan_scenes.append(dict(sc, dur=dur, narr_start=0.5, narr_dur=nd, image=images.get(i), start=t))
        t += dur
    total = round(t, 2)
    # 6. music
    music = None
    if o["music"] and MUSIC_MODEL:
        music = os.path.join(d, "music_%s.wav" % h(sb.get("music_prompt"), MUSIC_MODEL))
        if not os.path.exists(music):
            step(job, "composing music", 0.64)
            try:
                MEDIA.music(sb.get("music_prompt") or "light upbeat electronic", music, 30)
            except Exception as e:
                log.error("music failed\n%s", traceback.format_exc())
                warnings.append("music failed: %s" % str(e)[:200])
                music = None
    if UNLOAD_AFTER:
        MEDIA.unload()
    # 7. audio mix
    step(job, "mixing audio", 0.68)
    audio = os.path.join(d, "audio.wav")
    if narr or music:
        M.mix(total, [(ps["start"] + ps["narr_start"], narr[i][0]) for i, ps in enumerate(plan_scenes) if i in narr],
              music, audio)
    elif os.path.exists(audio):
        os.remove(audio)
    # 8. render
    K.write_json(os.path.join(d, "plan.json"), {"story": sb, "style": o["style"], "captions": o["captions"], "fps": FPS,
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
        save(job, state="running", step="rendering %s (about %s left)" % (" + ".join(procs), "%d:%02d" % divmod(eta, 60)),
             progress=round(0.7 + 0.3 * frac, 3), render=prog)
        time.sleep(2)
    errors = []
    for f, p in procs.items():
        pr = K.read_json(os.path.join(d, f, "progress.json"), {}) or {}
        if p.returncode != 0 or pr.get("state") != "done":
            errors.append("%s: %s" % (f, pr.get("error") or K.tail(os.path.join(d, "render_%s.log" % f), 5)))
    if errors:
        raise RuntimeError("rendering failed; " + " | ".join(errors))
    save(job, state="done", step="done", progress=1.0, duration=total, warnings=warnings,
         seconds=round(time.time() - t0), scene_images={str(i): p for i, p in images.items()},
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


def set_storyboard(jid, sb, render=False):
    d = jdir(jid)
    job = load(jid)
    if job["state"] in ("queued", "running"):
        raise ValueError("the job is busy (%s); wait for it to finish" % job["state"])
    src = K.read_json(os.path.join(d, "source.json")) or {"kind": "prompt", "title": job["title"], "text": "", "facts": {}}
    if isinstance(sb, str):
        sb = json.loads(sb)
    clean = ST.validate_user_storyboard(sb, src, job["options"]["length"])
    K.write_json(os.path.join(d, "storyboard.json"), clean)
    save(job, title=clean["title"][:100])
    if render:
        enqueue(job, approved=True)
    return clean


def rerender(jid, b):
    job = load(jid)
    if job["state"] in ("queued", "running"):
        raise ValueError("the job is already %s" % job["state"])
    opts = parse_options(b or {}, job["options"])
    if b and b.get("rewrite"):
        p = os.path.join(jdir(jid), "storyboard.json")
        if os.path.exists(p):
            os.remove(p)
    return enqueue(job, options=opts, approved=not (b or {}).get("rewrite") or not opts["review"])


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
    return {"styles": list(V.STYLES), "voices": M.VOICES, "lengths": list(ST.LENGTHS), "formats": list(FORMATS),
            "qualities": ["1080p", "720p"], "llm": LLM_MODEL, "image_model": IMAGE_MODEL or None,
            "music_model": MUSIC_MODEL or None, "tts": TTS_ENABLED, "gpus": K.gpu_info(),
            "queue": {s: sum(1 for j in jobs if j["state"] == s) for s in ("queued", "running", "review", "done", "error")}}


# ====================================================================== app
def build_app():
    app = K.App("Video Studio", password=os.environ.get("APP_PASSWORD"), log=log, max_body=8 << 20, instructions=(
        "Makes narrated explainer videos with music from any link (GitHub repo, article, YouTube, PDF) or an idea, "
        "rendered as landscape (16:9) and reel (9:16) MP4s. make_video queues a job; poll get_job until state is "
        "'done' (rendering takes a few minutes), then download files.landscape.mp4 / files.reel.mp4 with the same "
        "password. With review=true the job stops at state 'review': read get_storyboard, edit it with "
        "update_storyboard(render=true) or call render to continue."))
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

    @app.route("POST", r"/api/jobs/(?P<j>[\w-]+)/delete")
    def _delete(req):
        return delete(req.params["j"])

    @app.route("GET", r"/api/jobs/(?P<j>[\w-]+)/files/(?P<f>(?:images/)?[\w.-]+)")
    def _file(req):
        d = jdir(req.params["j"])
        f = req.params["f"]
        job = load(req.params["j"])
        name = "%s-%s" % (K.safe_name(job.get("title"), 50) or job["id"], f) if f.endswith(".mp4") else None
        return K.FileResponse(os.path.join(d, f), name=name if req.arg("download") else None)

    # ---------------------------------------------------------------- MCP tools
    opt_schema = {
        "style": {"type": "string", "enum": list(V.STYLES), "default": "midnight"},
        "length": {"type": "integer", "enum": list(ST.LENGTHS), "default": 60, "description": "seconds"},
        "voice": {"type": "string", "enum": list(M.VOICES) + ["none"], "default": "af_heart"},
        "music": {"type": "boolean", "default": True}, "captions": {"type": "boolean", "default": True},
        "formats": {"type": "array", "items": {"type": "string", "enum": list(FORMATS)},
                    "default": list(FORMATS)},
        "quality": {"type": "string", "enum": ["1080p", "720p"], "default": DEFAULT_QUALITY}}

    @app.tool("make_video", "Make a narrated video with music from a link (GitHub repo, article, YouTube, PDF) or a "
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

    @app.tool("get_storyboard", "The job's script: title, tagline, music prompt and scenes (layout, heading, bullets, "
              "narration, image_prompt, code, quote, stats).", {"job_id": {"type": "string"}}, ["job_id"])
    def t_sb(job_id):
        sb = K.read_json(os.path.join(jdir(job_id), "storyboard.json"))
        if not sb:
            raise ValueError("the script is not written yet (state: %s)" % load(job_id)["state"])
        return sb

    @app.tool("update_storyboard", "Replace the job's script (same shape as get_storyboard) and optionally render it. "
              "Unchanged scenes reuse their images and narration.", {
                  "job_id": {"type": "string"}, "storyboard": {"type": "object"},
                  "render": {"type": "boolean", "default": True}}, ["job_id", "storyboard"])
    def t_update(job_id, storyboard, render=True):
        return {"storyboard": set_storyboard(job_id, storyboard, render), "state": load(job_id)["state"]}

    @app.tool("render", "Render (or re-render) a job, optionally with new options. rewrite=true writes a new script first.",
              dict({"job_id": {"type": "string"}, "rewrite": {"type": "boolean", "default": False}}, **opt_schema),
              ["job_id"])
    def t_render(job_id, **b):
        return public(rerender(job_id, b))

    @app.tool("list_jobs", "Recent jobs, newest first.", {"limit": {"type": "integer", "default": 20}})
    def t_list(limit=20):
        return [public(j) for j in all_jobs()[:max(1, int(limit))]]

    @app.tool("delete_job", "Delete a job and its files.", {"job_id": {"type": "string"}}, ["job_id"])
    def t_delete(job_id):
        return delete(job_id)

    @app.tool("list_options", "Styles, voices, lengths, formats and the models in use.")
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
