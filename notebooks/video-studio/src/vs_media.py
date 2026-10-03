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
# Music is first brought to one loudness (RMS 0.2), then: (level alone, level under the voice). The voice stays
# about 12 dB above the music while it speaks, a broadcast-style balance.
MUSIC_LEVELS = {"low": (0.38, 0.12), "medium": (0.55, 0.18), "high": (0.75, 0.26)}
TONE_SCALE = {"tragic": 0.75, "serious": 0.85}


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
    def music(self, prompt, path, seconds=30, tone="neutral"):
        name = self.music_model
        if name.lower() == "synth":
            return synth_music(path, tone, seconds)
        if name.lower().startswith("ace"):
            try:
                return self._ace(prompt, path, seconds)
            except Exception as e:
                name = os.environ.get("MUSIC_FALLBACK", "synth")
                if self.log:
                    self.log.warning("ACE-Step failed (%s); using %s", e, name or "no music")
                if not name:
                    raise
                if name == "synth":
                    return synth_music(path, tone, seconds)
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


# ---------------------------------------------------------------------- built-in music (always commercial-safe)
_PROG = {   # chord roots (semitones from A) and qualities, bpm, pulse
    "tragic": ([(0, "m"), (8, "M"), (3, "M"), (10, "M")], 58, False),       # Am F C G, slow, no beat
    "serious": ([(0, "m"), (5, "m"), (8, "M"), (7, "M")], 72, False),       # Am Dm F E
    "neutral": ([(3, "M7"), (0, "m7"), (8, "M7"), (10, "6")], 88, True),     # Cmaj7 Am7 Fmaj7 G6
    "upbeat": ([(3, "M"), (10, "M"), (0, "m"), (8, "M")], 112, True),       # C G Am F
    "inspiring": ([(8, "M"), (3, "M"), (10, "M"), (0, "m")], 96, True),     # F C G Am
}
_CHORD = {"m": (0, 3, 7), "M": (0, 4, 7), "m7": (0, 3, 7, 10), "M7": (0, 4, 7, 11), "6": (0, 4, 7, 9)}


def synth_music(path, tone="neutral", seconds=30, rate=OUT_RATE, seed=3):
    """A soft, loopable bed made in code: warm pads, a sub bass, a plucked arpeggio and (for lighter tones) a gentle
    pulse, with a simple reverb. Nothing sampled, so it is free to use in monetised videos."""
    rng = np.random.default_rng(seed)
    prog, bpm, pulse = _PROG.get(tone, _PROG["neutral"])
    beat = 60.0 / bpm
    bar = 4 * beat
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    out = np.zeros(n, np.float32)
    hz = lambda semi, octave: 220.0 * 2 ** ((semi + 12 * octave) / 12.0)
    bars = int(np.ceil(seconds / bar))
    for b in range(bars):
        root, q = prog[b % len(prog)]
        s0, s1 = int(b * bar * rate), min(n, int((b + 1) * bar * rate + 0.6 * rate))
        if s0 >= n:
            break
        tt = t[s0:s1] - t[s0]
        env = np.minimum(1, tt / 0.8) * np.exp(-np.maximum(0, tt - bar) * 3)          # slow swell, overlapping tails
        for iv in _CHORD[q]:                            # pad: detuned soft saws (a few harmonics)
            f = hz(root + iv, -1)
            for det in (-0.12, 0.12):
                ph = 2 * np.pi * f * (1 + det / 100) * tt
                out[s0:s1] += 0.045 * env * (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph))
        out[s0:s1] += 0.10 * env * np.sin(2 * np.pi * hz(root, -2) * tt)               # sub bass
        notes = [root + iv for iv in _CHORD[q]] + [root + 12]
        step = beat / (2 if pulse else 1)
        for k in range(int(bar / step)):                # plucked arpeggio
            a0 = s0 + int(k * step * rate)
            if a0 >= n:
                break
            a1 = min(n, a0 + int(1.2 * rate))
            ta = t[a0:a1] - t[a0]
            f = hz(notes[(k * 2 + b) % len(notes)], 1)
            out[a0:a1] += (0.05 if pulse else 0.035) * np.exp(-ta * 4.5) * (np.sin(2 * np.pi * f * ta) + 0.2 * np.sin(4 * np.pi * f * ta))
        if pulse:                                       # soft kick + shaker
            for k in range(4):
                k0 = s0 + int(k * beat * rate)
                if k0 >= n:
                    break
                k1 = min(n, k0 + int(0.35 * rate))
                tk = t[k0:k1] - t[k0]
                out[k0:k1] += 0.16 * np.sin(2 * np.pi * (55 + 60 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 9)
                h0 = k0 + int(beat / 2 * rate)
                if h0 < n:
                    h1 = min(n, h0 + int(0.06 * rate))
                    out[h0:h1] += 0.03 * rng.standard_normal(h1 - h0).astype(np.float32) * np.exp(-(t[h0:h1] - t[h0]) * 60)
    ir_n = int(1.8 * rate)                              # reverb: decaying noise impulse response (FFT convolution)
    ir = rng.standard_normal(ir_n).astype(np.float32) * np.exp(-np.arange(ir_n) / (0.45 * rate))
    ir /= np.abs(ir).sum() / 6
    m = 1 << int(np.ceil(np.log2(n + ir_n)))
    wet = np.fft.irfft(np.fft.rfft(out, m) * np.fft.rfft(ir, m), m)[:n].astype(np.float32)
    out = 0.75 * out + 0.25 * wet / (np.abs(wet).max() + 1e-6) * np.abs(out).max()
    fade = int(1.5 * rate)
    out[-fade:] *= np.linspace(1, 0, fade)
    write_wav(path, out / (np.abs(out).max() + 1e-6) * 0.9, rate)
    return path


def tempo(src, dst, factor):
    """Speeds speech up by factor without changing its pitch (ffmpeg atempo); returns the new duration."""
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter:a", "atempo=%.3f" % factor, dst],
                   check=True, capture_output=True, timeout=120)
    return duration(dst)


def mix(total, narrations, music_path, out_path, tone="neutral", cuts=None, level="medium", ambience=None):
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
    music_gain, duck_gain = (x * TONE_SCALE.get(tone, 1.0) for x in MUSIC_LEVELS.get(level, MUSIC_LEVELS["medium"]))
    # clips' own sound: natural sound under the narration, full when nobody speaks in that scene
    amb = np.zeros(n, np.float32)
    for start, dur, p, spoken in ambience or []:
        try:
            x, r = read_wav(p)
        except Exception:
            continue
        x = resample(x, r, OUT_RATE)[:int(dur * OUT_RATE)]
        rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0
        if rms < 1e-4:
            continue
        x = x * (0.15 / rms) * (0.35 if spoken else 0.9)
        f = min(len(x) // 2, int(0.3 * OUT_RATE))
        if f:
            x[:f] *= np.linspace(0, 1, f)
            x[-f:] *= np.linspace(1, 0, f)
        i = int(start * OUT_RATE)
        x = x[:max(0, n - i)]
        amb[i:i + len(x)] += x
    if amb.any():
        venv = np.convolve(np.abs(voice), np.ones(int(0.3 * OUT_RATE), np.float32) / int(0.3 * OUT_RATE), mode="same")
        amb *= 1 - 0.55 * np.clip(venv / 0.02, 0, 1)    # and it steps back further while the voice is speaking
    out = voice + amb
    if music_path and os.path.exists(music_path):
        m, r = read_wav(music_path)
        m = resample(m, r, OUT_RATE)
        rms = float(np.sqrt(np.mean(m ** 2))) if len(m) else 0
        if rms > 1e-4:                                  # one loudness for every track, whatever the model made
            m = np.clip(m * (0.2 / rms), -0.98, 0.98)
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
        env = np.abs(voice) + 0.5 * np.abs(amb)        # music also gives way to a clip's own sound
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
        fx = vs_motion.sfx_track(total, cuts, OUT_RATE, tone)[:len(out)]
        env = np.convolve(np.abs(voice), np.ones(int(0.25 * OUT_RATE), np.float32) / int(0.25 * OUT_RATE), mode="same")
        out = out + fx * (1 - 0.85 * np.clip(env / 0.02, 0, 1))     # sounds step back whenever someone speaks
    peak = np.abs(out).max()
    if peak > 1e-4:
        out = out * (0.95 / peak)                       # full level: platforms play quiet uploads quietly
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
        p["seen_as"], p["relevance"] = kind, rel         # what the vision model saw ("kind" says where it came from)
        if (kind in DROP_KINDS or rel < 3) and p.get("kind") != "video":   # footage is never screened out
            if log:
                log.info("photo dropped (%s, relevance %s): %s", kind or "?", rel, str(p.get("url"))[:120])
            continue
        kept.append(p)
        if len(kept) >= keep:
            break
    return sorted(kept, key=lambda p: p.get("seen_as") == "stock graphic")   # real photos lead (the headline uses the first)
