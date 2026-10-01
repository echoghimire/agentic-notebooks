"""B-roll for OpenShorts clips, all local and free.

clip.mp4 -> transcribe (faster-whisper) -> pick moments + image prompts (Ollama)
         -> images (Stable Diffusion, default SDXL 1.0 base) -> slow zoom (FFmpeg zoompan)
         -> cut the b-roll into the clip, original audio kept -> clip_broll.mp4

Mounted by the gateway at /broll:
  GET  /broll/                 page
  GET  /broll/api/clips        clips OpenShorts has made
  POST /broll/api/upload       add your own clip (mp4)
  GET  /broll/media/<path>     clips, images and results
  POST /broll/api/jobs         {"clip": "<job>/<file.mp4>", "count": 3, "mode": "upper"|"full", "style": "..."}
  GET  /broll/api/jobs/<id>    progress and result
"""
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Route

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "output_dir": "output",                       # OpenShorts output dir (served at /videos)
    "ollama": "http://127.0.0.1:11434/v1",
    "llm_model": "qwen2.5:7b",
    "sd_model": "stabilityai/stable-diffusion-xl-base-1.0",   # OpenRAIL++: commercial use allowed
    "sd_steps": 20,
    "sd_device": "cuda:1",
    "whisper_model": "small",
    "whisper_device": "cuda",
    "whisper_index": 1,
    "fps": 30,
    "unload_after": False,
}
JOBS = {}
QUEUE_LOCK = threading.Lock()
_models = {"sd": None, "whisper": None}
STYLE = "photorealistic, cinematic lighting, shallow depth of field, high detail, vertical composition, no text, no watermark"
NEGATIVE = "text, letters, watermark, logo, blurry, distorted, deformed hands, extra fingers"


# ------------------------------------------------------------------ helpers
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + (p.stderr or p.stdout)[-1500:])
    return p.stdout


def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate:format=duration", "-of", "json", path])
    d = json.loads(out)
    st, fmt = d["streams"][0], d["format"]
    num, _, den = st.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    has_audio = bool(json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                     "stream=index", "-of", "json", path])).get("streams"))
    return {"w": int(st["width"]), "h": int(st["height"]), "fps": round(fps, 3) or 30.0,
            "duration": float(fmt["duration"]), "audio": has_audio}


def safe_clip_path(rel):
    root = os.path.realpath(CFG["output_dir"])
    full = os.path.realpath(os.path.join(root, rel))
    if not full.startswith(root + os.sep) or not full.endswith(".mp4") or not os.path.isfile(full):
        raise ValueError("unknown clip")
    return full


def list_clips():
    root = CFG["output_dir"]
    out = []
    for job_dir in sorted(glob.glob(os.path.join(root, "*")), key=os.path.getmtime, reverse=True):
        if not os.path.isdir(job_dir):
            continue
        job = os.path.basename(job_dir)
        title, source = job, None
        metas = glob.glob(os.path.join(job_dir, "*_metadata.json"))
        if metas:
            title = os.path.basename(metas[0])[:-len("_metadata.json")]
            try:
                source = json.load(open(metas[0])).get("source_video")
            except Exception:
                pass
        for f in sorted(glob.glob(os.path.join(job_dir, "*.mp4"))):
            name = os.path.basename(f)
            if name == source or name.endswith("_broll.mp4") or name.startswith(("temp_", "tmp_", ".")):
                continue
            broll = f[:-4] + "_broll.mp4"
            out.append({"clip": "%s/%s" % (job, name), "job": job, "title": title, "name": name,
                        "url": "/broll/media/%s/%s" % (job, name), "mtime": os.path.getmtime(f),
                        "broll_url": "/broll/media/%s/%s" % (job, os.path.basename(broll)) if os.path.exists(broll) else None})
    return out


# ------------------------------------------------------------------ 1. transcribe
def transcribe(path):
    if _models["whisper"] is None:
        from faster_whisper import WhisperModel
        dev = CFG["whisper_device"]
        _models["whisper"] = WhisperModel(CFG["whisper_model"], device=dev, device_index=CFG["whisper_index"],
                                          compute_type="float16" if dev == "cuda" else "int8")
    segs, _ = _models["whisper"].transcribe(path, vad_filter=True)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs if s.text.strip()]


# ------------------------------------------------------------------ 2. plan with Ollama
PLAN_PROMPT = """You are a short-form video editor. Pick moments in this vertical clip where a b-roll image would help the viewer picture what is being said.

Clip length: {duration:.1f} seconds.
Transcript (start-end seconds: text):
{lines}

Return JSON only, in this exact shape:
{{"inserts": [{{"start": 4.2, "duration": 2.5, "prompt": "a detailed visual description for an image generator"}}]}}

Rules:
- Exactly {count} inserts, in time order, at least 1 second apart.
- Each duration between 1.5 and 3.5 seconds.
- Nothing before {lead:.1f}s or after {tail:.1f}s.
- Prompts describe a concrete scene that matches what is said at that moment. {style_hint}
- No text, captions, logos, brand names or real people's names in prompts.
"""


def plan(segments, duration, count, style):
    lead, tail = min(1.5, duration * .1), max(0.0, duration - 1.0)
    lines = "\n".join("%.1f-%.1f: %s" % (s["start"], s["end"], s["text"]) for s in segments) or "(no speech)"
    prompt = PLAN_PROMPT.format(duration=duration, lines=lines[:12000], count=count, lead=lead, tail=tail,
                                style_hint=("Visual style: " + style + ".") if style else "")
    inserts = []
    try:
        r = httpx.post(CFG["ollama"] + "/chat/completions", timeout=300, json={
            "model": CFG["llm_model"], "temperature": 0.4, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        inserts = json.loads(m.group(0) if m else text).get("inserts", [])
    except Exception as e:
        print("[broll] LLM plan failed, using even spacing:", e)
    return clean_plan(inserts, segments, duration, count, lead, tail)


def clean_plan(inserts, segments, duration, count, lead, tail):
    out, last_end = [], -1e9
    for it in sorted((i for i in inserts if isinstance(i, dict)), key=lambda i: float(i.get("start", 0) or 0)):
        try:
            s, d = float(it["start"]), float(it.get("duration", 2.5))
        except (KeyError, TypeError, ValueError):
            continue
        d = min(3.5, max(1.5, d))
        s = max(lead, s)
        if s < last_end + 1.0 <= s + 1.5:      # too close to the previous insert: nudge it later a little
            s = last_end + 1.0
        if s + d > tail:
            s = tail - d
        p = str(it.get("prompt") or "").strip()
        if s < lead or not p or s < last_end + 1.0 - 1e-6:
            continue
        out.append({"start": round(s, 2), "duration": round(d, 2), "prompt": p[:400]})
        last_end = s + d
        if len(out) == count:
            break
    if not out and tail - lead >= 2:            # fallback: evenly spaced, prompt = what is said there
        n = max(1, min(count, int((tail - lead) // 4)))
        step = (tail - lead) / n
        for k in range(n):
            s = lead + k * step + max(0, (step - 2.5) / 2)
            said = " ".join(x["text"] for x in segments if x["end"] > s and x["start"] < s + 2.5) or "the topic of the video"
            out.append({"start": round(s, 2), "duration": 2.5, "prompt": "a scene showing " + said[:200]})
    return out


# ------------------------------------------------------------------ 3. images with Stable Diffusion
def image_size(info, mode):
    """Target aspect at the model's native size (~1 MP for SDXL, ~0.5 MP for Turbo/SD 1.x), multiples of 64."""
    w, h = info["w"], info["h"] if mode == "full" else int(info["h"] * 0.6)
    ratio = w / h
    m = CFG["sd_model"].lower()
    area = 768 * 1344 if ("xl" in m and "turbo" not in m) else 512 * 896
    gh = int(round((area / ratio) ** 0.5 / 64)) * 64
    gw = int(round(gh * ratio / 64)) * 64
    return max(256, min(1536, gw)), max(256, min(1536, gh))


def generate_image(prompt, style, size, out_path, seed):
    if _models["sd"] is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(CFG["sd_model"], torch_dtype=torch.float16, variant="fp16")
        except Exception:                      # model without an fp16 variant
            pipe = AutoPipelineForText2Image.from_pretrained(CFG["sd_model"], torch_dtype=torch.float16)
        pipe = pipe.to(CFG["sd_device"])
        if hasattr(pipe, "enable_vae_tiling"):  # keeps the VAE decode within the T4's memory next to Ollama
            pipe.enable_vae_tiling()
        _models["sd"] = pipe
    import torch
    pipe = _models["sd"]
    g = torch.Generator(device=CFG["sd_device"]).manual_seed(seed)
    full_prompt = ", ".join(x for x in (prompt, style, STYLE) if x)
    turbo = "turbo" in CFG["sd_model"].lower()
    img = pipe(prompt=full_prompt, negative_prompt=None if turbo else NEGATIVE, width=size[0], height=size[1],
               num_inference_steps=CFG["sd_steps"], guidance_scale=0.0 if turbo else 6.0, generator=g).images[0]
    img.save(out_path)
    return out_path


# ------------------------------------------------------------------ 4. slow zoom (Ken Burns)
def ken_burns(image, out_path, w, h, duration, fps, k):
    frames = max(2, int(round(duration * fps)))
    zoom_in = k % 2 == 0
    z = "min(1+0.12*on/%d,1.12)" % frames if zoom_in else "max(1.12-0.12*on/%d,1)" % frames
    vf = ("scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,"
          "zoompan=z='%s':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=%d:s=%dx%d:fps=%s,format=yuv420p"
          % (w * 2, h * 2, w * 2, h * 2, z, frames, w, h, fps))
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", image, "-vf", vf, "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out_path])
    return out_path


# ------------------------------------------------------------------ 5. cut into the clip
def composite(clip, inserts, info, mode, out_path):
    """Overlay each zoom clip during its window; 'upper' keeps the bottom 40% (burned-in captions) visible."""
    w, h = info["w"], info["h"]
    bh = h if mode == "full" else int(h * 0.6) // 2 * 2
    args = ["ffmpeg", "-y", "-v", "error", "-i", clip]
    for it in inserts:
        args += ["-i", it["video"]]
    chain, last = [], "0:v"
    for i, it in enumerate(inserts, 1):
        s, d = it["start"], it["duration"]
        f = min(0.25, d / 4)
        chain.append("[%d:v]scale=%d:%d,format=yuva420p,fade=t=in:st=0:d=%.2f:alpha=1,fade=t=out:st=%.2f:d=%.2f:alpha=1,"
                     "setpts=PTS-STARTPTS+%.3f/TB[b%d]" % (i, w, bh, f, d - f, f, s, i))
        chain.append("[%s][b%d]overlay=x=0:y=0:eof_action=pass:enable='between(t,%.3f,%.3f)'[v%d]" % (last, i, s, s + d, i))
        last = "v%d" % i
    args += ["-filter_complex", ";".join(chain), "-map", "[%s]" % last]
    if info["audio"]:
        args += ["-map", "0:a", "-c:a", "copy"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    run(args)
    return out_path


# ------------------------------------------------------------------ job runner
def _set(job, **kw):
    job.update(kw)
    job["updated"] = time.time()


def run_job(job):
    try:
        with QUEUE_LOCK:                      # one job at a time: the GPU is shared
            clip = safe_clip_path(job["clip"])
            info = probe(clip)
            work = os.path.join(os.path.dirname(clip), "broll_" + os.path.basename(clip)[:-4])
            os.makedirs(work, exist_ok=True)
            _set(job, state="running", step="Transcribing the clip", progress=0.05, info=info)
            segments = transcribe(clip)
            _set(job, step="Choosing moments with %s" % CFG["llm_model"], progress=0.2, transcript=segments)
            inserts = plan(segments, info["duration"], job["count"], job["style"])
            if not inserts:
                raise RuntimeError("clip is too short for b-roll")
            _set(job, inserts=inserts, progress=0.3)
            size = image_size(info, job["mode"])
            bh = info["h"] if job["mode"] == "full" else int(info["h"] * 0.6) // 2 * 2
            for k, it in enumerate(inserts):
                _set(job, step="Drawing image %d of %d" % (k + 1, len(inserts)), progress=0.3 + 0.5 * k / len(inserts))
                img = generate_image(it["prompt"], job["style"], size, os.path.join(work, "img_%d.png" % k), seed=job["seed"] + k)
                it["image_url"] = "/broll/media/%s/%s/img_%d.png" % (os.path.basename(os.path.dirname(clip)), os.path.basename(work), k)
                it["video"] = ken_burns(img, os.path.join(work, "kb_%d.mp4" % k), info["w"], bh, it["duration"], info["fps"], k)
                _set(job, inserts=inserts)
            _set(job, step="Cutting b-roll into the clip", progress=0.85)
            out = clip[:-4] + "_broll.mp4"
            composite(clip, inserts, info, job["mode"], out)
            rel = os.path.relpath(out, CFG["output_dir"]).replace(os.sep, "/")
            _set(job, state="done", step="Done", progress=1.0, result_url="/broll/media/" + rel)
    except Exception as e:
        traceback.print_exc()
        _set(job, state="error", step="Failed", error=str(e)[-1500:])
    finally:
        if CFG.get("unload_after"):            # single-GPU sessions: give the VRAM back to OpenShorts/Ollama
            unload_models()


def create_job(clip, count=3, mode="upper", style="", seed=None):
    safe_clip_path(clip)
    if mode not in ("upper", "full"):
        raise ValueError("mode must be 'upper' or 'full'")
    count = max(1, min(6, int(count)))
    job = {"id": uuid.uuid4().hex[:12], "clip": clip, "count": count, "mode": mode, "style": str(style or "")[:200],
           "seed": int(seed) if seed is not None else int(time.time()) % 100000, "state": "queued",
           "step": "Waiting for the GPU", "progress": 0.0, "created": time.time(), "updated": time.time()}
    JOBS[job["id"]] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job


def public(job):
    j = {k: v for k, v in job.items() if k not in ("transcript",)}
    j["inserts"] = [{k: v for k, v in it.items() if k != "video"} for it in job.get("inserts", [])]
    return j


# ------------------------------------------------------------------ HTTP
async def page(request):
    return FileResponse(os.path.join(HERE, "broll_ui.html"))


async def media(request):
    root = os.path.realpath(CFG["output_dir"])
    full = os.path.realpath(os.path.join(root, request.path_params["path"]))
    if not full.startswith(root + os.sep) or not os.path.isfile(full):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(full)


async def api_clips(request):
    return JSONResponse({"clips": list_clips(), "model": CFG["llm_model"], "sd_model": CFG["sd_model"]})


async def api_jobs(request: Request):
    if request.method == "GET":
        return JSONResponse({"jobs": [public(j) for j in sorted(JOBS.values(), key=lambda j: -j["created"])][:30]})
    try:
        b = await request.json()
        job = create_job(b.get("clip", ""), b.get("count", 3), b.get("mode", "upper"), b.get("style", ""), b.get("seed"))
    except (ValueError, TypeError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(public(job))


async def api_job(request):
    job = JOBS.get(request.path_params["jid"])
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(public(job))


async def api_upload(request: Request):
    """Raw mp4 body (Cloudflare's free plan caps a request at 100 MB)."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.query_params.get("name", "clip.mp4"))[:80] or "clip.mp4"
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    d = os.path.join(CFG["output_dir"], "uploads-broll")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "%s_%s" % (uuid.uuid4().hex[:6], name))
    size = 0
    with open(path, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > 100 * 1024 * 1024:
                f.close()
                os.remove(path)
                return JSONResponse({"error": "file over 100 MB"}, status_code=413)
            f.write(chunk)
    try:
        probe(path)
    except Exception:
        os.remove(path)
        return JSONResponse({"error": "not a readable video"}, status_code=400)
    return JSONResponse({"clip": "uploads-broll/" + os.path.basename(path)})


def build_app(output_dir=None, **cfg):
    if output_dir:
        CFG["output_dir"] = output_dir
    CFG.update({k: v for k, v in cfg.items() if v is not None})
    return Starlette(routes=[
        Route("/", page),
        Route("/media/{path:path}", media),
        Route("/api/clips", api_clips),
        Route("/api/jobs", api_jobs, methods=["GET", "POST"]),
        Route("/api/jobs/{jid}", api_job),
        Route("/api/upload", api_upload, methods=["POST"]),
    ])


def unload_models():
    """Free GPU memory (e.g. before a long OpenShorts job)."""
    _models.update(sd=None, whisper=None)
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
