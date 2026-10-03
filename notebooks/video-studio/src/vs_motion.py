"""Motion for still photos, location maps, sound effects and (optional) AI image-to-video for Video Studio.

Everything here is best effort: when a model or service is missing the video still renders, just with less motion.

- depth(): Depth Anything V2 Small (Apache 2.0) -> a depth map per photo. The scene page then moves near and far
  parts of the real photo at different speeds (2.5D parallax, a WebGL shader in vs_scenes.py). Nothing is invented.
- cutout(): BiRefNet (MIT) separates the main subject so it can lift off the background in headline scenes.
- place_map(): geocodes the story's place (OpenStreetMap Nominatim) and snapshots a country view and a local view
  with MapLibre GL + OpenFreeMap tiles in headless Chromium; the map scene zooms between them with a pin.
- whoosh() / hit(): transition sounds synthesised in code (no sample libraries, no licences).
- animate(): image-to-video (LTX-Video or Wan 2.2 TI2V via diffusers) for non-news scenes only, off by default.
"""
import json
import os
import threading
import urllib.parse
import urllib.request

import numpy as np

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
CUTOUT_MODEL = "ZhengPeng7/BiRefNet_lite"
UA = "agentic-notebooks-video-studio/1.0 (https://github.com/echoghimire/agentic-notebooks)"
MAPLIBRE = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl"
MAP_STYLES = {"dark": "https://tiles.openfreemap.org/styles/dark", "light": "https://tiles.openfreemap.org/styles/positron",
              "bright": "https://tiles.openfreemap.org/styles/liberty"}
I2V_MODELS = {"ltx": "Lightricks/LTX-Video", "wan": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"}


class Motion:
    def __init__(self, device="cuda:0", log=None):
        self.device, self.log = device, log
        self.lock = threading.Lock()
        self.m = {}

    def _info(self, *a):
        if self.log:
            self.log.info(*a)

    # ------------------------------------------------------------------ depth parallax
    def depth(self, photo_path, out_path):
        """Writes a smoothed 8-bit depth map (bright = near) next to the photo; returns out_path."""
        from PIL import Image, ImageFilter
        with self.lock:
            if "depth" not in self.m:
                import torch
                from transformers import pipeline
                dev = 0 if self.device.startswith("cuda") else -1
                if self.device.startswith("cuda:"):
                    dev = int(self.device.split(":")[1])
                self.m["depth"] = pipeline("depth-estimation", model=DEPTH_MODEL, device=dev,
                                           torch_dtype=torch.float16 if dev >= 0 else torch.float32)
                self._info("loaded %s", DEPTH_MODEL)
            im = Image.open(photo_path).convert("RGB")
            im.thumbnail((1024, 1024))
            d = self.m["depth"](im)["depth"].convert("L")
        d = d.resize(im.size).filter(ImageFilter.GaussianBlur(max(2, im.size[0] // 300)))   # soft edges: no tearing
        a = np.asarray(d, np.float32)
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        a = np.clip((a - lo) / max(1.0, hi - lo), 0, 1) * 255
        Image.fromarray(a.astype(np.uint8)).save(out_path)
        return out_path

    # ------------------------------------------------------------------ subject cut-out
    def cutout(self, photo_path, out_path):
        """Writes an RGBA PNG of the main subject; returns (out_path, covered fraction) or (None, frac) when the
        photo has no clear single subject (a crowd, a landscape...)."""
        import torch
        from PIL import Image
        with self.lock:
            if "cut" not in self.m:
                from transformers import AutoModelForImageSegmentation
                model = AutoModelForImageSegmentation.from_pretrained(CUTOUT_MODEL, trust_remote_code=True)
                model = model.to(self.device).eval()
                if self.device.startswith("cuda"):
                    model = model.half()
                self.m["cut"] = model
                self._info("loaded %s", CUTOUT_MODEL)
            model = self.m["cut"]
            im = Image.open(photo_path).convert("RGB")
            x = np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255.0
            x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            t = t.half() if self.device.startswith("cuda") else t.float()
            with torch.no_grad():
                pred = model(t)[-1].sigmoid().float().cpu()[0, 0].numpy()
        mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
        m = np.asarray(mask, np.float32) / 255
        frac = float((m > 0.5).mean())
        ys, xs = np.where(m > 0.5)
        touches = 0 if not len(xs) else sum([xs.min() < 3, ys.min() < 3, xs.max() > im.size[0] - 4, ys.max() > im.size[1] - 4])
        if not 0.06 <= frac <= 0.55 or touches >= 3:      # no clear subject: leave the photo whole
            return None, frac
        rgba = im.copy()
        rgba.putalpha(mask)
        rgba.save(out_path)
        return out_path, frac

    def unload(self):
        self.m.clear()
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ image-to-video (opt-in, non-news)
    def animate(self, image_path, prompt, out_dir, model="ltx", seconds=3.0, log=None):
        """Generates a short clip from a still and writes its frames as JPEGs into out_dir (24 fps).
        Returns the number of frames. Heavy: minutes per clip on a T4."""
        import torch
        from PIL import Image
        repo = I2V_MODELS.get(model, model)
        with self.lock:
            key = "i2v:" + repo
            if key not in self.m:
                self.m = {k: v for k, v in self.m.items() if not k.startswith("i2v:")}
                if "wan" in repo.lower():
                    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
                    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
                    pipe = WanImageToVideoPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.float16)
                else:
                    from diffusers import LTXImageToVideoPipeline
                    pipe = LTXImageToVideoPipeline.from_pretrained(repo, torch_dtype=torch.float16)
                gpu = int(self.device.split(":")[1]) if ":" in self.device else 0
                pipe.enable_model_cpu_offload(gpu_id=gpu)
                if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                pipe.set_progress_bar_config(disable=True)
                self.m[key] = pipe
                self._info("loaded %s", repo)
            pipe = self.m[key]
            im = Image.open(image_path).convert("RGB")
            wan = "wan" in repo.lower()
            w, h = ((832, 480) if wan else (768, 512))
            if im.size[0] < im.size[1]:
                w, h = h, w
            im = im.resize((w, h), Image.LANCZOS)
            fps = 24
            frames = int(seconds * fps) // 8 * 8 + 1         # both models want 8k+1 frames
            neg = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark"
            gen = torch.Generator("cpu").manual_seed(7)
            if wan:
                out = pipe(image=im, prompt=prompt, negative_prompt=neg, width=w, height=h, num_frames=frames,
                           num_inference_steps=30, guidance_scale=5.0, generator=gen).frames[0]
            else:
                out = pipe(image=im, prompt=prompt, negative_prompt=neg, width=w, height=h, num_frames=frames,
                           num_inference_steps=30, guidance_scale=3.0, generator=gen).frames[0]
        os.makedirs(out_dir, exist_ok=True)
        for i, f in enumerate(out):
            (f if hasattr(f, "save") else Image.fromarray((np.asarray(f) * 255).clip(0, 255).astype(np.uint8))).save(
                os.path.join(out_dir, "f%03d.jpg" % i), quality=90)
        return len(out)


# ---------------------------------------------------------------------- location map
def geocode(place, cache_dir=None):
    """(lat, lon, display name) for a place name via OpenStreetMap Nominatim (one request, cached)."""
    key = urllib.parse.quote(place.strip().lower())
    path = os.path.join(cache_dir, "geo_%s.json" % key[:80]) if cache_dir else None
    if path and os.path.exists(path):
        return tuple(json.load(open(path)))
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": place, "format": "json", "limit": 1, "accept-language": "en"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.loads(r.read())
    if not res:
        return None
    out = (float(res[0]["lat"]), float(res[0]["lon"]), res[0].get("display_name", place))
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(out, open(path, "w"))
    return out


MAP_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="%(lib)s.css"><script src="%(lib)s.js"></script>
<style>html,body,#m{margin:0;width:100%%;height:100%%;background:#111}.maplibregl-ctrl{display:none}</style></head>
<body><div id="m"></div><script>
window.__idle = false;
const map = new maplibregl.Map({container: "m", style: "%(style)s", center: [%(lon)f, %(lat)f], zoom: %(zoom)f,
  interactive: false, attributionControl: false, preserveDrawingBuffer: true, fadeDuration: 0});
map.on("idle", () => { window.__idle = true; });
</script></body></html>"""


MAP_ZOOMS = (("wide", 5.0), ("mid", 6.6), ("close", 8.2))     # each step is 2**1.6 = 3x closer (matches vs_scenes)


def place_map(lat, lon, out_dir, theme="dark", size=(1920, 1920)):
    """Snapshots three views centred on the place, each 3x closer than the last; the map scene zooms through
    them with cross-fades. Returns {"wide", "mid", "close"} paths."""
    from playwright.sync_api import sync_playwright
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None,
                               args=["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                     "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": size[0], "height": size[1]})
        for name, zoom in MAP_ZOOMS:
            html = MAP_PAGE % {"lib": MAPLIBRE, "style": MAP_STYLES.get(theme, MAP_STYLES["dark"]), "lon": lon, "lat": lat,
                               "zoom": zoom}
            f = os.path.join(out_dir, "map_%s.html" % name)
            open(f, "w", encoding="utf-8").write(html)
            page.goto("file://" + f, wait_until="load", timeout=60000)
            page.wait_for_function("window.__idle === true", timeout=90000)
            page.wait_for_timeout(400)
            p = os.path.join(out_dir, "map_%s.jpg" % name)
            page.screenshot(path=p, type="jpeg", quality=90)
            out[name] = p
        b.close()
    return out


# ---------------------------------------------------------------------- transition sounds
def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def whoosh(rate=44100, dur=0.7, seed=1):
    """A filtered-noise sweep that rises then falls: a soft transition swoosh."""
    n = int(rate * dur)
    x = _noise(n, seed)
    t = np.linspace(0, 1, n, dtype=np.float32)
    cut = 0.02 + 0.25 * np.sin(np.pi * t) ** 2               # one-pole low-pass whose cutoff sweeps up and down
    y = np.zeros(n, np.float32)
    acc = 0.0
    for i in range(n):
        acc += cut[i] * (x[i] - acc)
        y[i] = acc
    env = np.clip(np.sin(np.pi * t), 0, 1) ** 1.5
    y = y * env
    return y / (np.abs(y).max() + 1e-6)


def hit(rate=44100, dur=0.9, seed=2):
    """A low, soft impact (sine drop plus a little noise) for headline reveals."""
    n = int(rate * dur)
    t = np.arange(n, dtype=np.float32) / rate
    f = 90 * np.exp(-t * 5) + 45
    y = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-t * 4.5)
    y += 0.15 * _noise(n, seed) * np.exp(-t * 30)
    return (y / (np.abs(y).max() + 1e-6)).astype(np.float32)


def sfx_track(total, cuts, rate=44100, tone="neutral"):
    """Whooshes centred on each scene change (seconds) and a soft hit at the start, as one track."""
    level = {"tragic": 0.10, "serious": 0.14}.get(tone, 0.22)
    out = np.zeros(int(total * rate), np.float32)
    w = whoosh(rate)
    for k, c in enumerate(cuts):
        i = int((c - 0.35) * rate)
        seg = w if k % 2 == 0 else w[::-1]                 # alternate so repeated cuts do not sound identical
        if 0 <= i < len(out):
            out[i:i + len(seg)] += seg[:len(out) - i] * level
    hh = hit(rate)
    out[:len(hh)] += hh[:len(out)] * level * 0.8
    return out


# ---------------------------------------------------------------------- video clips (uploads, links, source footage)
def make_clip(src, base, start=0.0, seconds=10.0, fps=24):
    """Cuts src[start:start+seconds] into base.mp4 (max 1600 px), 24 fps frames in base_frames/, its sound in base.wav
    (when it has any) and a poster base.jpg. Returns a photos.json entry (path, w, h, clip, frames_dir, fps, audio)."""
    import subprocess
    from PIL import Image
    clip, fdir, wav, poster = base + ".mp4", base + "_frames", base + ".wav", base + ".jpg"
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    sz = "scale='if(gt(iw,ih),min(1600,iw),-2)':'if(gt(iw,ih),-2,min(1600,ih))'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.2f" % start, "-i", src, "-t", "%.2f" % seconds, "-vf", sz,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    clip], check=True, capture_output=True, timeout=600)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", "fps=%d" % fps, "-q:v", "3", os.path.join(fdir, "f%03d.jpg")],
                   check=True, capture_output=True, timeout=600)
    frames = frames_in(fdir)
    if not frames:
        raise ValueError("could not read frames from that video")
    a = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vn", "-ac", "1", "-ar", "44100", "-sample_fmt", "s16", wav],
                       capture_output=True, timeout=300)
    has_audio = a.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) > 4096
    if not has_audio and os.path.exists(wav):
        os.remove(wav)
    import shutil
    shutil.copy(frames[min(len(frames) - 1, 12)], poster)
    w, h = Image.open(poster).size
    return {"path": poster, "w": w, "h": h, "clip": clip, "frames_dir": fdir, "fps": fps, "audio": wav if has_audio else None}


def shot_starts(video, max_seconds=None):
    """Times (s) where the picture changes shot (ffmpeg scene detection), for picking distinct clips."""
    import subprocess
    cmd = ["ffmpeg", "-v", "info", "-hide_banner", "-i", video]
    if max_seconds:
        cmd[3:3] = ["-t", str(max_seconds)]
    cmd += ["-vf", "select='gt(scene,0.32)',showinfo,scale=160:-2", "-an", "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    import re
    return [float(x) for x in re.findall(r"pts_time:([\d.]+)", p.stderr)]


def video_seconds(video):
    import subprocess
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def pick_segments(video, n, seconds=7.0):
    """n non-overlapping clip starts: the video (minus the first 4% of intros and the last 6% of end screens) is cut
    into n equal windows and each clip starts on a shot change inside its window when there is one. Clips get
    shorter when the video is too short for n of them. Returns (starts, clip_seconds)."""
    total = video_seconds(video)
    if total <= 0 or n <= 0:
        return [], seconds
    lo, hi = total * 0.04, total * 0.94
    win = (hi - lo) / n
    seconds = max(1.5, min(seconds, win))
    cuts = shot_starts(video)
    out = []
    for k in range(n):
        a, b = lo + k * win, lo + (k + 1) * win
        inside = [c + 0.15 for c in cuts if a <= c + 0.15 and c + 0.15 + seconds <= b + 0.01]
        out.append(round(inside[0] if inside else a + (win - seconds) / 2, 2))
    return out, round(seconds, 2)


# ---------------------------------------------------------------------- helpers
def frames_in(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")) if os.path.isdir(d) else []
