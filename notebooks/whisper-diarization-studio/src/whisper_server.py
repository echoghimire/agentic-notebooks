"""Whisper Diarization Studio: the meeting & audio OS on port 7860.

Runs as its own process (started by the notebook with studio_http.spawn):
    APP_PASSWORD=... HF_TOKEN=... python whisper_server.py --port 7860

Audio and video in (upload, URL, or a file under /kaggle/input) -> faster-whisper transcript ->
pyannote speaker diarization -> optional Ollama summary with decisions and action items ->
txt / md / srt / vtt / json. One job runs at a time; jobs live in WORK_DIR/jobs/<id>/ and survive
a server restart (an interrupted job is queued again).
"""
import argparse
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import wave

import studio_http as K
import whisper_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR", "/kaggle/working/whisper_studio")
JOBS_DIR = os.path.join(WORK, "jobs")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
DIAR_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
INPUT_ROOTS = [r for r in ("/kaggle/input", "/kaggle/working") if os.path.isdir(r)] or [WORK]
MEDIA_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".webm", ".mp4", ".mkv",
             ".mov", ".avi", ".m4v", ".amr", ".3gp")
MAX_UPLOAD = 4 << 30
MAX_WAIT = 85
log = K.file_logger(os.environ.get("APP_LOG", "/kaggle/working/logs/whisper.log"), "whisper")

LOCK = threading.RLock()
WAKE = threading.Condition(LOCK)
MODELS = {"whisper": None, "diar": None, "diar_error": None, "devices": None}
SUMMARIES = {}       # job id -> thread


# ====================================================================== job storage
def jdir(jid):
    jid = K.safe_name(jid, 40)
    d = os.path.join(JOBS_DIR, jid)
    if not jid or not os.path.isfile(os.path.join(d, "job.json")):
        raise K.HTTPError(404, "unknown job %r" % jid)
    return d


def load_job(jid):
    return K.read_json(os.path.join(jdir(jid), "job.json"))


def save_job(job, **changes):
    with LOCK:
        cur = K.read_json(os.path.join(JOBS_DIR, job["id"], "job.json"), job) or job
        cur.update(changes)
        cur["updated"] = time.time()
        K.write_json(os.path.join(JOBS_DIR, job["id"], "job.json"), cur)
        job.clear()
        job.update(cur)
    return job


def all_jobs():
    out = []
    if os.path.isdir(JOBS_DIR):
        for name in os.listdir(JOBS_DIR):
            j = K.read_json(os.path.join(JOBS_DIR, name, "job.json"))
            if j:
                out.append(j)
    return sorted(out, key=lambda j: -j["created"])


def public(job):
    keys = ("id", "title", "state", "step", "progress", "created", "updated", "error", "warnings", "duration",
            "language", "speakers", "options", "source", "summary_state", "summary_error", "seconds")
    return {k: job.get(k) for k in keys if k in job}


def parse_options(b):
    def opt_int(k, lo=1, hi=30):
        v = b.get(k)
        if v in (None, "", 0, "0"):
            return None
        v = int(v)
        if not lo <= v <= hi:
            raise ValueError("%s must be between %d and %d" % (k, lo, hi))
        return v

    def flag(k, default):
        v = b.get(k)
        if v in (None, ""):
            return default
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
    task = b.get("task") or "transcribe"
    if task not in ("transcribe", "translate"):
        raise ValueError("task must be 'transcribe' or 'translate' (to English)")
    lang = (b.get("language") or "").strip().lower() or None
    if lang and not re.fullmatch(r"[a-z]{2,3}", lang):
        raise ValueError("language must be a code like 'en', 'ne', 'hi' (or empty to detect)")
    return {"language": lang, "task": task, "diarize": flag("diarize", True),
            "num_speakers": opt_int("num_speakers"), "min_speakers": opt_int("min_speakers"),
            "max_speakers": opt_int("max_speakers"), "summarize": flag("summarize", bool(SUMMARY_MODEL)),
            "initial_prompt": str(b.get("initial_prompt") or "")[:800] or None, "vad": flag("vad", True)}


def new_job(source, options, title=None):
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = os.path.join(JOBS_DIR, jid)
    os.makedirs(d)
    job = {"id": jid, "title": (title or source.get("name") or source.get("value") or jid)[:120],
           "source": source, "options": options, "state": "new", "step": "receiving", "progress": 0.0,
           "created": time.time(), "warnings": []}
    K.write_json(os.path.join(d, "job.json"), job)
    return d, job


def enqueue(job):
    with WAKE:
        save_job(job, state="queued", step="waiting", progress=0.0, error=None)
        WAKE.notify_all()
    log.info("queued %s (%s)", job["id"], job["title"])
    return job


def safe_input_path(path):
    full = os.path.realpath(str(path or ""))
    if not any(full == r or full.startswith(os.path.realpath(r) + os.sep) for r in INPUT_ROOTS):
        raise ValueError("path must be under " + " or ".join(INPUT_ROOTS))
    if not os.path.isfile(full):
        raise ValueError("no such file: %s" % path)
    return full


def list_inputs():
    out = []
    for root in INPUT_ROOTS:
        for dp, dn, fn in os.walk(root):
            if dp.startswith(os.path.realpath(WORK)):
                dn[:] = []
                continue
            for f in sorted(fn):
                if f.lower().endswith(MEDIA_EXT):
                    p = os.path.join(dp, f)
                    out.append({"path": p, "size": os.path.getsize(p)})
                    if len(out) >= 300:
                        return out
    return out


# ====================================================================== models (loaded on first use)
def devices():
    if MODELS["devices"] is None:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        # whisper on GPU 0; diarization on GPU 1 when there is one
        MODELS["devices"] = {"n": n, "whisper": 0 if n else None, "diar": (1 if n > 1 else 0) if n else None}
    return MODELS["devices"]


def whisper_model():
    if MODELS["whisper"] is None:
        from faster_whisper import WhisperModel
        dv = devices()
        if dv["whisper"] is None:
            MODELS["whisper"] = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        else:
            MODELS["whisper"] = WhisperModel(WHISPER_MODEL, device="cuda", device_index=dv["whisper"],
                                             compute_type="float16")
        log.info("loaded whisper %s", WHISPER_MODEL)
    return MODELS["whisper"]


def diarization_status():
    if not HF_TOKEN:
        return False, "add an HF_TOKEN secret (and accept the terms of %s and pyannote/segmentation-3.0)" % DIAR_MODEL
    if MODELS["diar_error"]:
        return False, MODELS["diar_error"]
    try:
        import importlib.util
        if importlib.util.find_spec("pyannote.audio") is None:
            return False, "pyannote.audio is not installed"
    except Exception:
        return False, "pyannote.audio is not installed"
    return True, "ready" if MODELS["diar"] else "loads on first use"


def diar_pipeline():
    if MODELS["diar"] is None:
        import torch
        from pyannote.audio import Pipeline
        try:
            pipe = Pipeline.from_pretrained(DIAR_MODEL, token=HF_TOKEN)              # pyannote.audio 4.x
        except TypeError:
            pipe = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)     # pyannote.audio 3.x
        if pipe is None:
            raise RuntimeError("could not load %s: accept its terms (and pyannote/segmentation-3.0) on "
                               "huggingface.co with the account of your HF_TOKEN" % DIAR_MODEL)
        dv = devices()
        if dv["diar"] is not None:
            pipe.to(torch.device("cuda:%d" % dv["diar"]))
        MODELS["diar"] = pipe
        log.info("loaded diarization %s", DIAR_MODEL)
    return MODELS["diar"]


# ====================================================================== pipeline steps
def run(cmd, timeout=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("%s failed: %s" % (os.path.basename(cmd[0]), (p.stderr or p.stdout)[-1200:]))
    return p.stdout


def fetch_input(job, d):
    src = job["source"]
    if src["type"] in ("upload", "path"):
        return src["file"] if src["type"] == "upload" else safe_input_path(src["value"])
    save_job(job, step="downloading")
    url = src["value"]
    if shutil.which("yt-dlp"):
        run(["yt-dlp", "-q", "--no-playlist", "-f", "bestaudio/best", "--max-filesize", "4G",
             "-o", os.path.join(d, "input.%(ext)s"), url], timeout=3600)
        files = [f for f in os.listdir(d) if f.startswith("input.") and not f.endswith(".part")]
        if not files:
            raise RuntimeError("download produced no file")
        return os.path.join(d, files[0])
    import urllib.request
    path = os.path.join(d, "input" + (os.path.splitext(url.split("?")[0])[1][:6] or ".bin"))
    urllib.request.urlretrieve(url, path)
    return path


def convert(job, d, path):
    save_job(job, step="converting audio")
    wav = os.path.join(d, "audio.wav")
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", wav], timeout=3600)
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", wav, "-c:a", "libmp3lame",
         "-b:a", "64k", os.path.join(d, "audio.mp3")], timeout=3600)
    with wave.open(wav) as w:
        duration = w.getnframes() / float(w.getframerate())
    if duration < 0.5:
        raise RuntimeError("the file has no usable audio")
    return wav, duration


def transcribe(job, wav, duration):
    o = job["options"]
    save_job(job, step="loading the speech model" if MODELS["whisper"] is None else "transcribing")
    model = whisper_model()
    save_job(job, step="transcribing")
    segs, info = model.transcribe(wav, language=o["language"], task=o["task"], word_timestamps=True,
                                  vad_filter=o["vad"], initial_prompt=o["initial_prompt"], beam_size=5)
    out, last = [], 0.0
    for s in segs:
        out.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
                    "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                               "p": round(w.probability, 3)} for w in (s.words or [])]})
        if time.time() - last > 2:
            save_job(job, progress=round(min(0.99, s.end / duration), 3) * (0.75 if o["diarize"] else 1.0))
            last = time.time()
    return out, info.language


def diarize(job, wav):
    import numpy as np
    import torch
    o = job["options"]
    save_job(job, step="loading the diarization model" if MODELS["diar"] is None else "finding speakers",
             progress=0.75)
    pipe = diar_pipeline()
    save_job(job, step="finding speakers")
    with wave.open(wav) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        sr = w.getframerate()
    kw = {k: o[k] for k in ("num_speakers", "min_speakers", "max_speakers") if o.get(k)}
    out = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr}, **kw)
    ann = getattr(out, "speaker_diarization", out)       # 4.x returns DiarizeOutput, 3.x an Annotation
    return [(round(t.start, 3), round(t.end, 3), spk) for t, _, spk in ann.itertracks(yield_label=True)]


def process(job):
    d = os.path.join(JOBS_DIR, job["id"])
    t0 = time.time()
    save_job(job, state="running", progress=0.0, error=None, warnings=[])
    path = fetch_input(job, d)
    wav, duration = convert(job, d, path)
    save_job(job, duration=round(duration, 1))
    segments, language = transcribe(job, wav, duration)
    turns, warnings = [], []
    if job["options"]["diarize"]:
        ok, why = diarization_status()
        if ok:
            try:
                turns = diarize(job, wav)
            except Exception as e:
                log.error("diarization failed for %s\n%s", job["id"], traceback.format_exc())
                MODELS["diar_error"] = None if MODELS["diar"] else str(e)[:300]
                warnings.append("speaker diarization failed: %s" % str(e)[:300])
        else:
            warnings.append("no speaker labels: " + why)
    utterances = C.assign_speakers(segments, turns)
    data = {"language": language, "duration": round(duration, 1), "segments": segments, "utterances": utterances,
            "diarized": bool(turns), "whisper_model": WHISPER_MODEL, "options": job["options"]}
    K.write_json(os.path.join(d, "transcript.json"), data)
    if os.path.exists(wav) and os.path.exists(os.path.join(d, "audio.mp3")):
        os.remove(wav)                                  # keep the small mp3 for playback
    stats = C.speaker_stats(utterances)
    save_job(job, state="done", step="done", progress=1.0, language=language, warnings=warnings,
             speakers=sorted(stats), seconds=round(time.time() - t0, 1))
    log.info("done %s in %.0fs", job["id"], time.time() - t0)
    if job["options"]["summarize"] and SUMMARY_MODEL:
        start_summary(job["id"])


def worker():
    while True:
        with WAKE:
            queued = [j for j in all_jobs() if j["state"] == "queued"]
            if not queued:
                WAKE.wait(timeout=30)
                continue
            job = min(queued, key=lambda j: j["created"])
        try:
            process(job)
        except Exception as e:
            log.error("job %s failed\n%s", job["id"], traceback.format_exc())
            save_job(job, state="error", step="failed", error=str(e)[:1000])


# ====================================================================== summaries
def summary_text(jid, names=None):
    d = jdir(jid)
    data = K.read_json(os.path.join(d, "transcript.json"))
    if not data:
        raise ValueError("the transcript is not ready yet")
    return C.render("txt", data, names if names is not None else K.read_json(os.path.join(d, "speakers.json"), {}))


def start_summary(jid, instructions=""):
    if not SUMMARY_MODEL:
        raise ValueError("summaries are off: set SUMMARY_MODEL in the notebook's settings cell and re-run it")
    job = load_job(jid)
    if job["state"] != "done":
        raise ValueError("the transcript is not ready yet (state: %s)" % job["state"])
    t = SUMMARIES.get(jid)
    if t and t.is_alive():
        return job
    text = summary_text(jid)
    save_job(job, summary_state="running", summary_error=None, summary_step="starting")

    def go():
        try:
            md = C.summarize(text, OLLAMA_URL, SUMMARY_MODEL, instructions,
                             progress=lambda s: save_job(job, summary_step=s))
            with open(os.path.join(JOBS_DIR, jid, "summary.md"), "w", encoding="utf-8") as f:
                f.write(md + "\n")
            save_job(job, summary_state="done", summary_step="done")
        except Exception as e:
            log.error("summary %s failed\n%s", jid, traceback.format_exc())
            save_job(job, summary_state="error", summary_error=str(e)[:500])
    t = threading.Thread(target=go, daemon=True)
    SUMMARIES[jid] = t
    t.start()
    return job


def transcript_doc(jid, fmt):
    d = jdir(jid)
    job = load_job(jid)
    data = K.read_json(os.path.join(d, "transcript.json"))
    if not data:
        raise ValueError("the transcript is not ready yet (state: %s)" % job["state"])
    if os.path.exists(os.path.join(d, "summary.md")):
        data["summary"] = open(os.path.join(d, "summary.md"), encoding="utf-8").read()
    return C.render(fmt, data, K.read_json(os.path.join(d, "speakers.json"), {}), job["title"])


def rename_speakers(jid, names):
    if not isinstance(names, dict):
        raise ValueError("names must be an object like {\"SPEAKER_00\": \"Asha\"}")
    d = jdir(jid)
    cur = K.read_json(os.path.join(d, "speakers.json"), {})
    for k, v in names.items():
        v = str(v or "").strip()[:60]
        if v:
            cur[str(k)] = v
        else:
            cur.pop(str(k), None)
    K.write_json(os.path.join(d, "speakers.json"), cur)
    return cur


def wait_job(jid, seconds):
    seconds = max(0.0, min(MAX_WAIT, float(seconds or 0)))
    t = time.time()
    while True:
        job = load_job(jid)
        busy = job["state"] in ("queued", "running") or job.get("summary_state") == "running"
        if not busy or time.time() - t >= seconds:
            return job
        time.sleep(1.5)


def delete_job(jid):
    job = load_job(jid)
    if job["state"] == "running":
        raise ValueError("this job is running; wait for it to finish")
    shutil.rmtree(jdir(jid))
    return {"deleted": jid}


def info():
    ok, why = diarization_status()
    jobs = all_jobs()
    return {"whisper_model": WHISPER_MODEL, "diarization": {"available": ok, "detail": why, "model": DIAR_MODEL},
            "summary_model": SUMMARY_MODEL or None, "gpus": K.gpu_info(),
            "queue": {s: sum(1 for j in jobs if j["state"] == s) for s in ("queued", "running", "done", "error")}}


# ====================================================================== app
def build_app():
    app = K.App("Whisper Diarization Studio", password=os.environ.get("APP_PASSWORD"), log=log,
                max_body=MAX_UPLOAD, instructions=(
                    "Transcribes audio/video with speaker labels and meeting summaries. transcribe() queues a job "
                    "from a URL or a file path under /kaggle/input (see list_input_files) and returns a job_id; poll "
                    "get_job (wait_seconds up to 85) until state is 'done', then read get_transcript (format md "
                    "includes the summary). Rename speakers with rename_speakers. Uploads go through REST: "
                    "POST /api/jobs/upload?name=<file> with the raw file body."))
    app.page("/", os.path.join(HERE, "whisper_ui.html"))
    app.static("/static/", HERE)

    @app.route("GET", "/api/info")
    def _info(req):
        return info()

    @app.route("GET", "/api/jobs")
    def _jobs(req):
        return {"jobs": [public(j) for j in all_jobs()]}

    @app.route("POST", "/api/jobs/upload")
    def _upload(req):
        name = K.safe_name(req.arg("name", "upload.bin"), 100) or "upload.bin"
        if not name.lower().endswith(MEDIA_EXT):
            raise ValueError("unsupported file type; send audio or video (%s)" % " ".join(MEDIA_EXT))
        opts = parse_options(req.query)
        d, job = new_job({"type": "upload", "name": name}, opts, req.arg("title"))
        try:
            path = os.path.join(d, "input_" + name)
            size = req.save_body(path, MAX_UPLOAD)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        save_job(job, source=dict(job["source"], file=path, size=size))
        return public(enqueue(job))

    @app.route("POST", "/api/jobs")
    def _create(req):
        b = req.json()
        return public(create_from(b))

    @app.route("GET", r"/api/jobs/(?P<jid>[\w-]+)")
    def _job(req):
        jid = req.params["jid"]
        job = wait_job(jid, req.arg("wait", 0, float))
        d = jdir(jid)
        out = public(job)
        data = K.read_json(os.path.join(d, "transcript.json"))
        if data:
            out["utterances"] = data["utterances"]
            out["diarized"] = data.get("diarized")
            out["speaker_stats"] = C.speaker_stats(data["utterances"])
        out["speaker_names"] = K.read_json(os.path.join(d, "speakers.json"), {})
        if os.path.exists(os.path.join(d, "summary.md")):
            out["summary"] = open(os.path.join(d, "summary.md"), encoding="utf-8").read()
        out["has_audio"] = os.path.exists(os.path.join(d, "audio.mp3"))
        return out

    @app.route("GET", r"/api/jobs/(?P<jid>[\w-]+)/download/(?P<fmt>\w+)")
    def _download(req):
        fmt = req.params["fmt"]
        if fmt not in C.FORMATS:
            raise ValueError("format must be one of " + ", ".join(C.FORMATS))
        job = load_job(req.params["jid"])
        name = "%s.%s" % (K.safe_name(job["title"], 60) or job["id"], fmt)
        return K.Response(transcript_doc(job["id"], fmt), 200, C.FORMATS[fmt] + "; charset=utf-8",
                          {"Content-Disposition": K._disposition(name)})

    @app.route("GET", r"/api/jobs/(?P<jid>[\w-]+)/audio")
    def _audio(req):
        return K.FileResponse(os.path.join(jdir(req.params["jid"]), "audio.mp3"), ctype="audio/mpeg")

    @app.route("POST", r"/api/jobs/(?P<jid>[\w-]+)/speakers")
    def _speakers(req):
        return {"speaker_names": rename_speakers(req.params["jid"], req.json().get("names"))}

    @app.route("POST", r"/api/jobs/(?P<jid>[\w-]+)/summarize")
    def _summarize(req):
        return public(start_summary(req.params["jid"], str(req.json().get("instructions") or "")[:1000]))

    @app.route("POST", r"/api/jobs/(?P<jid>[\w-]+)/retry")
    def _retry(req):
        job = load_job(req.params["jid"])
        if job["state"] in ("queued", "running"):
            raise ValueError("job is already %s" % job["state"])
        return public(enqueue(job))

    @app.route("POST", r"/api/jobs/(?P<jid>[\w-]+)/delete")
    def _delete(req):
        return delete_job(req.params["jid"])

    @app.route("GET", "/api/sources")
    def _sources(req):
        return {"files": list_inputs()}

    # ---------------------------------------------------------------- MCP tools
    opts_schema = {
        "title": {"type": "string"},
        "language": {"type": "string", "description": "e.g. 'en', 'ne', 'hi'; omit to detect"},
        "task": {"type": "string", "enum": ["transcribe", "translate"], "description": "translate = into English"},
        "diarize": {"type": "boolean", "default": True},
        "num_speakers": {"type": "integer", "description": "exact number of speakers, if known"},
        "min_speakers": {"type": "integer"}, "max_speakers": {"type": "integer"},
        "summarize": {"type": "boolean", "description": "default: on when a summary model is configured"},
        "initial_prompt": {"type": "string", "description": "names and jargon to help spelling"},
    }

    @app.tool("transcribe", "Queue a transcription job from a URL (direct file or any yt-dlp site) or a file path "
              "under /kaggle/input. Returns the job; poll get_job until state is 'done'.",
              dict({"url": {"type": "string"}, "path": {"type": "string"}}, **opts_schema))
    def t_transcribe(url=None, path=None, **opts):
        return public(create_from(dict(opts, url=url, path=path)))

    @app.tool("get_job", "Job status, progress and speakers. wait_seconds (max 85) waits for it to finish.", {
        "job_id": {"type": "string"}, "wait_seconds": {"type": "number", "default": 0}}, ["job_id"])
    def t_get_job(job_id, wait_seconds=0):
        return public(wait_job(job_id, wait_seconds))

    @app.tool("get_transcript", "The transcript as txt (timestamps + speakers), md (with summary and talk time), "
              "srt, vtt or json. Long transcripts are paged: pass offset=next_offset.", {
                  "job_id": {"type": "string"}, "format": {"type": "string", "enum": list(C.FORMATS), "default": "md"},
                  "offset": {"type": "integer", "default": 0}, "max_chars": {"type": "integer", "default": 60000}},
              ["job_id"])
    def t_transcript(job_id, format="md", offset=0, max_chars=60000):
        if format not in C.FORMATS:
            raise ValueError("format must be one of " + ", ".join(C.FORMATS))
        text = transcript_doc(job_id, format)
        offset, max_chars = max(0, int(offset)), max(1000, min(200000, int(max_chars)))
        part = text[offset:offset + max_chars]
        nxt = offset + len(part)
        return part + ("\n\n[... %d more characters: call again with offset=%d]" % (len(text) - nxt, nxt)
                       if nxt < len(text) else "")

    @app.tool("list_jobs", "Recent jobs, newest first.", {"limit": {"type": "integer", "default": 20}})
    def t_list(limit=20):
        return [public(j) for j in all_jobs()[:max(1, int(limit))]]

    @app.tool("rename_speakers", "Give speakers real names, e.g. {\"SPEAKER_00\": \"Asha\"}. An empty name "
              "resets one.", {"job_id": {"type": "string"}, "names": {"type": "object"}}, ["job_id", "names"])
    def t_rename(job_id, names):
        return {"speaker_names": rename_speakers(job_id, names)}

    @app.tool("summarize", "(Re)write the meeting summary: key points, decisions, action items, open questions. "
              "Uses the current speaker names.", {
                  "job_id": {"type": "string"}, "instructions": {"type": "string"},
                  "wait_seconds": {"type": "number", "default": 60}}, ["job_id"])
    def t_summarize(job_id, instructions="", wait_seconds=60):
        start_summary(job_id, str(instructions or "")[:1000])
        job = wait_job(job_id, wait_seconds)
        if job.get("summary_state") == "done":
            return open(os.path.join(jdir(job_id), "summary.md"), encoding="utf-8").read()
        return public(job)

    @app.tool("delete_job", "Delete a job and its files.", {"job_id": {"type": "string"}}, ["job_id"])
    def t_delete(job_id):
        return delete_job(job_id)

    @app.tool("list_input_files", "Audio and video files attached to the notebook (under /kaggle/input).")
    def t_inputs():
        return list_inputs()

    @app.tool("server_status", "Models, diarization availability, GPU memory and queue counts.")
    def t_status():
        return info()

    return app


def create_from(b):
    url, path = (b.get("url") or "").strip(), (b.get("path") or "").strip()
    if bool(url) == bool(path):
        raise ValueError("give exactly one of url or path")
    opts = parse_options(b)
    if url:
        if not re.match(r"^https?://", url):
            raise ValueError("url must start with http:// or https://")
        source = {"type": "url", "value": url}
    else:
        source = {"type": "path", "value": safe_input_path(path), "name": os.path.basename(path)}
    d, job = new_job(source, opts, b.get("title"))
    return enqueue(job)


def resume_interrupted():
    for j in all_jobs():
        if j["state"] == "running":
            save_job(j, state="queued", step="waiting (restarted)")
        if j.get("summary_state") == "running":
            save_job(j, summary_state="error", summary_error="interrupted by a restart; run it again")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    os.makedirs(JOBS_DIR, exist_ok=True)
    resume_interrupted()
    threading.Thread(target=worker, daemon=True).start()
    build_app().serve_forever(ap.parse_args().port)
