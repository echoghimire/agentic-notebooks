"""Photos, images, narration and music for Video Studio, plus the audio mix. Heavy libraries load on first use.

- Photos: downloaded from the source page, checked, de-duplicated and lightly edited (auto-contrast, sharpen);
  optionally described by a vision LLM (Gemma 3 via Ollama) so the script can match photos to scenes.
- Images: Stable Diffusion XL (diffusers), only for non-news topics without usable photos.
- Narration: Kokoro-82M here; Nepali / Hindi voices live in vs_tts.py.
- Music: MusicGen (transformers), ~30 s generated once, looped under the whole video.
- mix(): narration track + music ducked under the voice -> one 44.1 kHz WAV for both formats.
"""
import os
import threading
import wave

import numpy as np

STYLE_SUFFIX = {
    "midnight": "cinematic digital illustration, deep blue and magenta lighting, soft glow, high detail",
    "paper": "warm editorial illustration, paper texture, muted earthy colors, gouache",
    "neon": "cyberpunk illustration, glowing cyan and magenta neon, dark background, high contrast",
    "swiss": "minimal flat geometric illustration, bold red black and white, clean shapes, bauhaus poster",
}
NEGATIVE = "text, letters, words, watermark, logo, signature, caption, ui, blurry, lowres, deformed, extra fingers"
OUT_RATE = 44100
MUSIC_LEVELS = {"tragic": (0.10, 0.035), "serious": (0.14, 0.05)}     # (music level, level under the voice)


def read_wav(path):
    with wave.open(path) as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, rate


def write_wav(path, data, rate):
    data = np.clip(np.asarray(data, dtype=np.float32), -1, 1)
    tmp = path + ".tmp.wav"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((data * 32767).astype(np.int16).tobytes())
    os.replace(tmp, path)


def resample(x, src, dst):
    if src == dst or not len(x):
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def duration(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


class Media:
    def __init__(self, image_model, image_steps=25, image_device="cuda:0", music_model="", music_device="cuda:0",
                 tts_device="cpu", log=None):
        self.image_model, self.image_steps, self.image_device = image_model, int(image_steps), image_device
        self.music_model, self.music_device, self.tts_device = music_model, music_device, tts_device
        self.log = log
        self.lock = threading.Lock()
        self.m = {}

    # ------------------------------------------------------------------ images
    def _pipe(self):
        if "sd" not in self.m:
            import torch
            from diffusers import AutoPipelineForText2Image
            kw = {"torch_dtype": torch.float16}
            if "xl" in self.image_model.lower() and "turbo" not in self.image_model.lower():
                from diffusers import AutoencoderKL
                kw["vae"] = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(self.image_model, variant="fp16", **kw)
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(self.image_model, **kw)
            pipe.set_progress_bar_config(disable=True)
            self.m["sd"] = pipe.to(self.image_device)
            if self.log:
                self.log.info("loaded %s on %s", self.image_model, self.image_device)
        return self.m["sd"]

    def image(self, prompt, style, path, seed=0):
        import torch
        with self.lock:
            pipe = self._pipe()
            full = "%s, %s" % (prompt, STYLE_SUFFIX.get(style, ""))
            gen = torch.Generator(self.image_device).manual_seed(int(seed))
            turbo = "turbo" in self.image_model.lower() or "schnell" in self.image_model.lower()
            out = pipe(prompt=full, negative_prompt=None if turbo else NEGATIVE, num_inference_steps=self.image_steps,
                       guidance_scale=0.0 if turbo else 6.0, width=1024, height=1024, generator=gen).images[0]
        out.save(path + ".tmp.png")
        os.replace(path + ".tmp.png", path)
        return path

    # ------------------------------------------------------------------ narration
    def tts(self, text, voice, path, speed=1.0):
        """Writes a 24 kHz WAV and returns its duration in seconds."""
        lang = voice[0]
        with self.lock:
            key = "tts_" + lang
            if key not in self.m:
                from kokoro import KPipeline
                try:
                    self.m[key] = KPipeline(lang_code=lang, device=self.tts_device)
                except TypeError:
                    self.m[key] = KPipeline(lang_code=lang)
            parts = []
            for r in self.m[key](text, voice=voice, speed=speed):
                audio = r.audio if hasattr(r, "audio") else r[2]
                if audio is not None:
                    parts.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        if not parts:
            raise RuntimeError("no audio produced")
        data = np.concatenate(parts).astype(np.float32)
        write_wav(path, data, 24000)
        return len(data) / 24000.0

    # ------------------------------------------------------------------ music
    def music(self, prompt, path, seconds=30):
        name = self.music_model
        if name.lower().startswith("ace"):
            try:
                return self._ace(prompt, path, seconds)
            except Exception as e:
                name = os.environ.get("MUSIC_FALLBACK", "")
                if not name:
                    raise
                if self.log:
                    self.log.warning("ACE-Step failed (%s); using %s", e, name)
        with self.lock:
            if "mg" not in self.m:
                import torch
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
                proc = AutoProcessor.from_pretrained(name)
                model = MusicgenForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16
                                                                         if self.music_device.startswith("cuda") else torch.float32)
                self.m["mg"] = (proc, model.to(self.music_device))
            proc, model = self.m["mg"]
            inputs = proc(text=[prompt + ", instrumental, no vocals, loopable"], padding=True, return_tensors="pt").to(self.music_device)
            out = model.generate(**inputs, max_new_tokens=int(min(30, seconds) * 50), do_sample=True, guidance_scale=3.0)
            rate = model.config.audio_encoder.sampling_rate
            data = out[0, 0].float().cpu().numpy()
        write_wav(path, data / (np.abs(data).max() + 1e-6) * 0.9, rate)
        return path

    def _ace(self, prompt, path, seconds):
        """ACE-Step (Apache 2.0, commercial use OK) in its own virtualenv; see vs_ace_worker.py."""
        import subprocess
        py = os.environ.get("ACE_PYTHON", "")
        if not py or not os.path.exists(py):
            raise RuntimeError("ACE-Step is not installed (see the notebook's install cell)")
        dev = int(self.music_device.split(":")[1]) if ":" in self.music_device else 0
        log = os.environ.get("ACE_LOG", "/kaggle/working/logs/ace_step.log")
        with self.lock, open(log, "a") as lf:
            p = subprocess.run([py, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vs_ace_worker.py"), "--out", path,
                                "--prompt", prompt, "--seconds", str(int(seconds)), "--device", str(dev)],
                               stdout=lf, stderr=subprocess.STDOUT, timeout=1800)
        if p.returncode != 0 or not os.path.exists(path):
            raise RuntimeError("ACE-Step failed; see " + log)
        return path

    def unload(self, *names):
        for n in names or list(self.m):
            self.m.pop(n, None)
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def tempo(src, dst, factor):
    """Speeds speech up by factor without changing its pitch (ffmpeg atempo); returns the new duration."""
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter:a", "atempo=%.3f" % factor, dst],
                   check=True, capture_output=True, timeout=120)
    return duration(dst)


def mix(total, narrations, music_path, out_path, tone="neutral", cuts=None):
    """narrations: [(start_seconds, wav_path)]. Music loops with crossfades and dips under the voice.
    cuts: scene-change times for soft transition whooshes (None = no sound effects)."""
    n = int(total * OUT_RATE)
    voice = np.zeros(n, np.float32)
    for start, p in narrations:
        x, r = read_wav(p)
        x = resample(x, r, OUT_RATE)
        i = int(start * OUT_RATE)
        x = x[:max(0, n - i)]
        voice[i:i + len(x)] += x
    music_gain, duck_gain = MUSIC_LEVELS.get(tone, (0.22, 0.07))
    out = voice.copy()
    if music_path and os.path.exists(music_path):
        m, r = read_wav(music_path)
        m = resample(m, r, OUT_RATE)
        xf = min(int(1.5 * OUT_RATE), len(m) // 4)
        if len(m) > 2 * xf:
            loop = m.copy()
            while len(loop) < n:                         # append with a crossfade so the loop seam is smooth
                fade = np.linspace(0, 1, xf, dtype=np.float32)
                loop[-xf:] = loop[-xf:] * (1 - fade) + m[:xf] * fade
                loop = np.concatenate([loop, m[xf:]])
            m = loop[:n]
        else:
            m = np.resize(m, n)
        env = np.abs(voice)
        win = int(0.35 * OUT_RATE)
        env = np.convolve(env, np.ones(win, np.float32) / win, mode="same")
        talking = np.clip(env / 0.02, 0, 1)
        gain = music_gain - (music_gain - duck_gain) * talking
        fade_in, fade_out = int(1.0 * OUT_RATE), int(2.5 * OUT_RATE)
        gain[:fade_in] *= np.linspace(0, 1, len(gain[:fade_in]))
        gain[-fade_out:] *= np.linspace(1, 0, len(gain[-fade_out:]))
        out = out + m * gain
    if cuts is not None:
        import vs_motion
        out = out + vs_motion.sfx_track(total, cuts, OUT_RATE, tone)[:len(out)]
    peak = np.abs(out).max()
    if peak > 0.97:
        out = out * (0.97 / peak)
    write_wav(out_path, out, OUT_RATE)
    return out_path


# ====================================================================== photos from the source
def _ahash(im):
    g = im.convert("L").resize((9, 8))
    px = list(g.getdata())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def fetch_photos(images, out_dir, limit=8, referer=None, log=None):
    """Downloads candidate photos, keeps real photos (>= 480 px, sane shape, not duplicates), edits them lightly.
    Returns [{"path", "w", "h", "caption", "alt", "url"}] in the source's order of importance."""
    import io
    import urllib.request
    from PIL import Image, ImageFilter, ImageOps
    os.makedirs(out_dir, exist_ok=True)
    kept, hashes = [], []
    for c in images:
        if len(kept) >= limit:
            break
        try:
            if c.get("bytes"):
                data = c["bytes"]
            else:
                req = urllib.request.Request(c["url"], headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/126 Safari/537.36",
                                                                "Referer": referer or c["url"], "Accept": "image/*,*/*;q=0.5"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    data = r.read(20 << 20)
            im = Image.open(io.BytesIO(data))
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            if log:
                log.info("photo skipped (%s): %s", str(c.get("url"))[:120], e)
            continue
        w, h = im.size
        if min(w, h) < 480 or max(w, h) / min(w, h) > 3.2:
            continue
        hsh = _ahash(im)
        if any(bin(hsh ^ x).count("1") <= 6 for x in hashes):
            if log:
                log.info("duplicate photo skipped: %s", str(c.get("url"))[:120])
            continue                                    # same picture at another size or crop
        hashes.append(hsh)
        if max(w, h) > 2400:
            im.thumbnail((2400, 2400), Image.LANCZOS)
        im = ImageOps.autocontrast(im, cutoff=0.5, preserve_tone=True) if "preserve_tone" in ImageOps.autocontrast.__code__.co_varnames \
            else ImageOps.autocontrast(im, cutoff=0.5)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))
        path = os.path.join(out_dir, "photo_%02d.jpg" % len(kept))
        im.save(path, "JPEG", quality=92)
        kept.append({"path": path, "w": im.size[0], "h": im.size[1], "caption": c.get("caption") or "",
                     "alt": c.get("alt") or "", "url": c.get("url")})
    return kept


SCREEN_PROMPT = """This picture was found on a web page about: {topic}

Return ONLY a JSON object:
{{"description": "one short factual English sentence: who or what is shown and where; no guessing names",
  "kind": one of "news photo", "advertisement", "product", "stock graphic", "logo or text", "portrait", "other",
  "relevance": 0 to 10, how well it fits the page's topic (10 = clearly shows this story)}}

Advertisements and products include food, kitchens, appliances, phones, fashion, banks, offers and prices."""
DROP_KINDS = {"advertisement", "product", "logo or text"}


def describe_photos(photos, ollama_url, model, keep_alive="10m", log=None, topic="", keep=8):
    """Asks a vision model (Gemma 3) to describe each photo and to spot ads and off-topic pictures, which are
    dropped. Returns the photos to use; best effort (without a vision model every photo is kept)."""
    import base64
    import io
    import json as _json
    import re as _re
    import urllib.request
    from PIL import Image
    kept = []
    for i, p in enumerate(photos):
        try:
            im = Image.open(p["path"])
            im.thumbnail((640, 640))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            body = _json.dumps({"model": model, "stream": False, "keep_alive": keep_alive, "format": "json",
                                "options": {"temperature": 0.1},
                                "messages": [{"role": "user", "images": [base64.b64encode(buf.getvalue()).decode()],
                                              "content": SCREEN_PROMPT.format(topic=topic[:300] or "(unknown)")}]}).encode()
            req = urllib.request.Request(ollama_url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                content = _json.loads(r.read())["message"]["content"]
            m = _re.search(r"\{.*\}", content, _re.S)
            try:
                v = _json.loads(m.group(0)) if m else {}
            except ValueError:
                v = {}
            if not isinstance(v, dict) or not v:
                v = {"description": content}            # a plain-text answer: keep it as the description
        except Exception as e:
            if log:
                log.info("photo check skipped: %s", e)
            return (kept + photos[i:])[:keep]          # the model cannot see images: keep the rest unchecked
        kind = str(v.get("kind") or "").lower().strip()
        try:
            rel = float(v.get("relevance", 5))
        except (TypeError, ValueError):
            rel = 5.0
        p["description"] = str(v.get("description") or "").strip()[:240]
        p["kind"], p["relevance"] = kind, rel
        if kind in DROP_KINDS or rel < 3:
            if log:
                log.info("photo dropped (%s, relevance %s): %s", kind or "?", rel, str(p.get("url"))[:120])
            continue
        kept.append(p)
        if len(kept) >= keep:
            break
    return sorted(kept, key=lambda p: p.get("kind") == "stock graphic")   # real photos lead (the headline uses the first)
