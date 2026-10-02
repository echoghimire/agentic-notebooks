"""Images, narration and music for Video Studio, plus the audio mix. Heavy libraries load on first use.

- Images: Stable Diffusion XL (diffusers), one square image per scene, cropped by the templates for 16:9 and 9:16.
- Narration: Kokoro-82M (Apache 2.0), 24 kHz.
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
VOICES = {
    "af_heart": "US English, female (warm)", "af_bella": "US English, female (bright)",
    "am_michael": "US English, male", "am_fenrir": "US English, male (deep)",
    "bf_emma": "UK English, female", "bm_george": "UK English, male",
    "ef_dora": "Spanish, female", "ff_siwis": "French, female", "hf_alpha": "Hindi, female",
    "if_sara": "Italian, female", "pf_dora": "Portuguese (BR), female",
}
OUT_RATE = 44100


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
        with self.lock:
            if "mg" not in self.m:
                import torch
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
                proc = AutoProcessor.from_pretrained(self.music_model)
                model = MusicgenForConditionalGeneration.from_pretrained(self.music_model, torch_dtype=torch.float16
                                                                         if self.music_device.startswith("cuda") else torch.float32)
                self.m["mg"] = (proc, model.to(self.music_device))
            proc, model = self.m["mg"]
            inputs = proc(text=[prompt + ", instrumental, no vocals, loopable"], padding=True, return_tensors="pt").to(self.music_device)
            out = model.generate(**inputs, max_new_tokens=int(min(30, seconds) * 50), do_sample=True, guidance_scale=3.0)
            rate = model.config.audio_encoder.sampling_rate
            data = out[0, 0].float().cpu().numpy()
        write_wav(path, data / (np.abs(data).max() + 1e-6) * 0.9, rate)
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


def mix(total, narrations, music_path, out_path, music_gain=0.22, duck_gain=0.07):
    """narrations: [(start_seconds, wav_path)]. Music loops with crossfades and dips under the voice."""
    n = int(total * OUT_RATE)
    voice = np.zeros(n, np.float32)
    for start, p in narrations:
        x, r = read_wav(p)
        x = resample(x, r, OUT_RATE)
        i = int(start * OUT_RATE)
        x = x[:max(0, n - i)]
        voice[i:i + len(x)] += x
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
    peak = np.abs(out).max()
    if peak > 0.97:
        out = out * (0.97 / peak)
    write_wav(out_path, out, OUT_RATE)
    return out_path
