"""ComfyUI + FLUX gateway on port 7860, the only port the Cloudflare Tunnel exposes.

Runs as its own process, started by the notebook with studio_http.spawn():
    APP_PASSWORD=... COMFY_URL=http://127.0.0.1:8188 python comfy_gateway.py --port 7860

- Everything ComfyUI serves (its node editor at /, its HTTP API and the /ws websocket) is proxied
  behind the password.
- /flux is a one-box prompt -> image page.
- /flux/api/* is a small JSON API, and /mcp offers the same features to agents as MCP tools.
"""
import argparse
import io
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import quote, urlencode

import studio_http as K

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
VARIANT = os.environ.get("FLUX_VARIANT", "schnell")
CKPT = os.environ.get("FLUX_CKPT", "flux1-schnell-fp8.safetensors")
DEFAULT_STEPS = {"schnell": 4, "dev": 20}.get(VARIANT, 20)
CLIENT_ID = "flux-gateway-" + uuid.uuid4().hex[:8]
MAX_WAIT = 85                     # Cloudflare ends requests after 100 s
log = K.file_logger(os.environ.get("APP_LOG", "/kaggle/working/logs/gateway.log"), "gateway")
JOBS = {}                         # prompt_id -> what we submitted (newest last)


# ====================================================================== ComfyUI client
def _comfy_error(detail):
    if not isinstance(detail, dict):
        return str(detail)[:500]
    parts = []
    err = detail.get("error")
    if isinstance(err, dict):
        parts.append(err.get("message") or err.get("type") or "error")
        if err.get("details"):
            parts.append(str(err["details"]))
    for node, info in (detail.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append("node %s (%s): %s %s" % (node, info.get("class_type", "?"), e.get("message", ""),
                                                  e.get("details", "")))
    return "; ".join(p for p in parts if p) or json.dumps(detail)[:500]


def comfy(path, data=None, timeout=30, raw=False):
    req = urllib.request.Request(COMFY + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            detail = json.loads(body)
        except ValueError:
            detail = body.decode("utf-8", "replace")
        raise K.HTTPError(400 if e.code < 500 else 502, "ComfyUI: " + _comfy_error(detail))
    except OSError as e:
        raise K.HTTPError(502, "ComfyUI is not reachable (%s). Check the ComfyUI cell and "
                               "/kaggle/working/logs/comfyui.log" % type(e).__name__)
    if raw:
        return body
    return json.loads(body) if body.strip() else {}


def view_url(img):
    return "/view?" + urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                 "type": img.get("type", "output")})


# ====================================================================== workflows
def _int(v, default, lo, hi, name):
    v = default if v in (None, "") else int(v)
    if not lo <= v <= hi:
        raise ValueError("%s must be between %d and %d" % (name, lo, hi))
    return v


def flux_workflow(prompt, width=1024, height=1024, steps=None, seed=None, guidance=3.5, batch_size=1,
                  prefix="flux/api"):
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")
    if len(prompt) > 4000:
        raise ValueError("prompt is longer than 4000 characters")
    width = _int(width, 1024, 256, 2048, "width") // 16 * 16
    height = _int(height, 1024, 256, 2048, "height") // 16 * 16
    steps = _int(steps, DEFAULT_STEPS, 1, 60, "steps")
    batch_size = _int(batch_size, 1, 1, 4, "batch_size")
    seed = random.randint(0, 2 ** 48) if seed in (None, "", -1) else int(seed)
    positive = ["6", 0]
    wf = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "33": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "27": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height,
                                                               "batch_size": batch_size}},
        "31": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0, "model": ["4", 0], "positive": positive, "negative": ["33", 0],
            "latent_image": ["27", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    if VARIANT == "dev":   # FLUX.1-dev uses distilled guidance instead of CFG
        wf["35"] = {"class_type": "FluxGuidance", "inputs": {"guidance": float(guidance or 3.5),
                                                             "conditioning": ["6", 0]}}
        wf["31"]["inputs"]["positive"] = ["35", 0]
    params = {"prompt": prompt, "width": width, "height": height, "steps": steps, "seed": seed,
              "batch_size": batch_size, "guidance": float(guidance or 3.5) if VARIANT == "dev" else None}
    return wf, params


def submit(workflow, kind, params=None):
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("workflow must be a ComfyUI API-format object: {node_id: {class_type, inputs}}")
    if "nodes" in workflow and "links" in workflow:
        raise ValueError("this is a UI-format workflow; in ComfyUI use Workflow -> Export (API) instead")
    r = comfy("/prompt", {"prompt": workflow, "client_id": CLIENT_ID})
    pid = r.get("prompt_id")
    if not pid:
        raise K.HTTPError(502, "ComfyUI did not return a prompt_id: %s" % _comfy_error(r))
    JOBS[pid] = {"job_id": pid, "kind": kind, "params": params or {}, "created": time.time()}
    while len(JOBS) > 200:
        JOBS.pop(next(iter(JOBS)))
    log.info("queued %s job %s", kind, pid)
    return pid


def _queue():
    q = comfy("/queue")
    running = [x[1] for x in q.get("queue_running", [])]
    pending = [x[1] for x in sorted(q.get("queue_pending", []), key=lambda x: x[0])]
    return running, pending


def job_status(pid, history=None, queue=None):
    out = dict(JOBS.get(pid) or {"job_id": pid})
    h = (history if history is not None else comfy("/history/" + quote(pid))).get(pid)
    if h:
        st = h.get("status") or {}
        images = []
        for node_out in (h.get("outputs") or {}).values():
            for img in node_out.get("images", []):
                if img.get("type") == "output":
                    images.append({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                   "url": view_url(img)})
        if st.get("status_str") == "error":
            msg = "failed"
            for name, data in st.get("messages", []):
                if name == "execution_error":
                    msg = "%s (node %s %s)" % (data.get("exception_message", "").strip(), data.get("node_id"),
                                               data.get("node_type"))
                elif name == "execution_interrupted":
                    msg = "cancelled"
            out.update(state="error", error=msg, images=images)
        else:
            out.update(state="done", images=images)
        times = [m[1].get("timestamp") for m in st.get("messages", []) if m[0] in ("execution_start",
                                                                                   "execution_success")]
        if len(times) == 2 and all(times):
            out["seconds"] = round((times[1] - times[0]) / 1000, 1)
        return out
    running, pending = queue or _queue()
    if pid in running:
        out.update(state="running")
    elif pid in pending:
        out.update(state="queued", queue_position=pending.index(pid) + 1)
    else:
        out.update(state="unknown", error="ComfyUI has no job with this id (it may have restarted)")
    return out


def wait_job(pid, seconds):
    seconds = max(0, min(MAX_WAIT, float(seconds or 0)))
    t = time.time()
    while True:
        st = job_status(pid)
        if st["state"] in ("done", "error", "unknown") or time.time() - t >= seconds:
            return st
        time.sleep(1.0)


def image_bytes(img, max_side=1024):
    """Image as JPEG for MCP replies (PNGs from FLUX are often 1.5 MB+)."""
    data = comfy(view_url(dict(img, type="output")), raw=True, timeout=60)
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, "image/png"


def tool_reply(st, return_image=True):
    blocks = [K.text_block(json.dumps(st, indent=1, default=str))]
    if st.get("state") == "done" and return_image:
        for img in st.get("images", [])[:4]:
            try:
                data, mime = image_bytes(img)
                blocks.append(K.image_block(data, mime))
            except Exception as e:
                blocks.append(K.text_block("could not attach %s: %s" % (img["filename"], e)))
    elif st.get("state") in ("queued", "running"):
        blocks.append(K.text_block("Still working. Call get_job with this job_id (wait_seconds up to 85)."))
    return K.ToolContent(blocks)


def models():
    def options(node, field):
        try:
            info = comfy("/object_info/" + node)[node]["input"]["required"][field][0]
            return info if isinstance(info, list) else []
        except (K.HTTPError, KeyError, IndexError, TypeError):
            return []
    return {"checkpoints": options("CheckpointLoaderSimple", "ckpt_name"),
            "loras": options("LoraLoader", "lora_name"),
            "default_checkpoint": CKPT, "variant": VARIANT}


def info():
    out = {"variant": VARIANT, "checkpoint": CKPT, "default_steps": DEFAULT_STEPS,
           "guidance": VARIANT == "dev"}
    try:
        running, pending = _queue()
        out["queue"] = {"running": len(running), "pending": len(pending)}
        stats = comfy("/system_stats")
        out["devices"] = [{"name": d.get("name"), "vram_free_gb": round(d.get("vram_free", 0) / 2 ** 30, 1),
                           "vram_total_gb": round(d.get("vram_total", 0) / 2 ** 30, 1)}
                          for d in stats.get("devices", [])]
        out["comfyui"] = (stats.get("system") or {}).get("comfyui_version")
        out["ready"] = True
    except K.HTTPError as e:
        out.update(ready=False, error=str(e))
    return out


def recent_jobs(limit=30):
    hist = comfy("/history?max_items=%d" % limit)
    queue = _queue()
    ids = list(dict.fromkeys(list(reversed(list(JOBS))) + queue[0] + queue[1] + list(reversed(list(hist)))))
    return [job_status(pid, hist, queue) for pid in ids[:limit]]


def cancel(pid):
    running, pending = _queue()
    if pid in running:
        comfy("/interrupt", {})
        return {"job_id": pid, "cancelled": "interrupted the running job"}
    if pid in pending:
        comfy("/queue", {"delete": [pid]})
        return {"job_id": pid, "cancelled": "removed from the queue"}
    return {"job_id": pid, "cancelled": False, "reason": "job is not queued or running"}


# ====================================================================== app
def build_app():
    app = K.App("ComfyUI FLUX Studio", password=os.environ.get("APP_PASSWORD"), proxy=COMFY, log=log,
                max_body=200 << 20, instructions=(
                    "Text-to-image with FLUX.1-%s on ComfyUI. generate_image queues a job and waits up to "
                    "wait_seconds (max 85) for it; if the reply says queued/running, call get_job with the "
                    "job_id. run_workflow accepts any ComfyUI API-format workflow. Images are also downloadable "
                    "from the url fields (relative to this server, same password)." % VARIANT))

    app.page("/flux", os.path.join(HERE, "flux_ui.html"))
    app.page("/flux/", os.path.join(HERE, "flux_ui.html"))
    app.static("/flux/static/", HERE)

    @app.route("GET", "/flux/api/info")
    def _info(req):
        return info()

    @app.route("GET", "/flux/api/models")
    def _models(req):
        return models()

    @app.route("POST", "/flux/api/generate")
    def _generate(req):
        b = req.json()
        wf, params = flux_workflow(b.get("prompt"), b.get("width"), b.get("height"), b.get("steps"), b.get("seed"),
                                   b.get("guidance"), b.get("batch_size"), prefix="flux/web")
        return {"job_id": submit(wf, "generate", params), "params": params}

    @app.route("POST", "/flux/api/workflow")
    def _workflow(req):
        b = req.json()
        return {"job_id": submit(b.get("workflow", b), "workflow")}

    @app.route("GET", "/flux/api/jobs")
    def _jobs(req):
        return {"jobs": recent_jobs(req.arg("limit", 30, int))}

    @app.route("GET", r"/flux/api/jobs/(?P<pid>[\w-]+)")
    def _job(req):
        return wait_job(req.params["pid"], req.arg("wait", 0, float))

    @app.route("POST", r"/flux/api/jobs/(?P<pid>[\w-]+)/cancel")
    def _cancel(req):
        return cancel(req.params["pid"])

    # ---------------------------------------------------------------- MCP tools
    @app.tool("generate_image", "Generate images from a text prompt with FLUX.1-%s. Returns the job status, and "
              "the images inline once done (JPEG previews; full PNGs at the url fields)." % VARIANT, {
                  "prompt": {"type": "string", "description": "What to draw. FLUX follows long, literal prompts well."},
                  "width": {"type": "integer", "default": 1024, "minimum": 256, "maximum": 2048},
                  "height": {"type": "integer", "default": 1024, "minimum": 256, "maximum": 2048},
                  "steps": {"type": "integer", "description": "Default %d" % DEFAULT_STEPS},
                  "seed": {"type": "integer", "description": "Omit for random"},
                  "guidance": {"type": "number", "default": 3.5, "description": "FLUX.1-dev only"},
                  "batch_size": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
                  "wait_seconds": {"type": "number", "default": 75, "maximum": 85},
                  "return_image": {"type": "boolean", "default": True}}, ["prompt"])
    def t_generate(prompt, width=1024, height=1024, steps=None, seed=None, guidance=3.5, batch_size=1,
                   wait_seconds=75, return_image=True):
        wf, params = flux_workflow(prompt, width, height, steps, seed, guidance, batch_size, prefix="flux/mcp")
        return tool_reply(wait_job(submit(wf, "generate", params), wait_seconds), return_image)

    @app.tool("get_job", "Status of a job; returns its images once done.", {
        "job_id": {"type": "string"},
        "wait_seconds": {"type": "number", "default": 0, "maximum": 85},
        "return_image": {"type": "boolean", "default": True}}, ["job_id"])
    def t_get_job(job_id, wait_seconds=0, return_image=True):
        return tool_reply(wait_job(str(job_id), wait_seconds), return_image)

    @app.tool("run_workflow", "Queue any ComfyUI workflow in API format ({node_id: {class_type, inputs}}; in "
              "ComfyUI: Workflow -> Export (API)). Use list_models for valid checkpoint and LoRA names.", {
                  "workflow": {"type": "object"},
                  "wait_seconds": {"type": "number", "default": 75, "maximum": 85},
                  "return_image": {"type": "boolean", "default": True}}, ["workflow"])
    def t_workflow(workflow, wait_seconds=75, return_image=True):
        return tool_reply(wait_job(submit(workflow, "workflow"), wait_seconds), return_image)

    @app.tool("list_jobs", "Recent jobs, newest first.", {"limit": {"type": "integer", "default": 20}})
    def t_jobs(limit=20):
        return [{k: v for k, v in j.items() if k != "params"} for j in recent_jobs(min(int(limit), 100))]

    @app.tool("cancel_job", "Cancel a queued or running job.", {"job_id": {"type": "string"}}, ["job_id"])
    def t_cancel(job_id):
        return cancel(str(job_id))

    @app.tool("list_models", "Checkpoints and LoRAs ComfyUI can load.")
    def t_models():
        return models()

    @app.tool("server_status", "Queue length, GPU memory and the loaded FLUX variant.")
    def t_status():
        return info()

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    build_app().serve_forever(ap.parse_args().port)
