"""Narration voices for Video Studio, by language.

- Kokoro-82M (Apache 2.0): English and a few European languages, in the server process (CPU is fine).
- Indic Parler-TTS (Apache 2.0): Nepali, Hindi and other Indian languages, in vs_parler_worker.py (its own
  virtualenv and process, on the media GPU), with a named speaker so the voice stays the same in every scene.
- Piper (MIT; voice from the OpenSLR Nepali corpus): a light CPU Nepali voice, used when Parler is unavailable.
"auto" picks the best installed voice for the video's language.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import studio_http as K

PARLER_PYTHON = os.environ.get("PARLER_PYTHON", "")
PARLER_PORT = int(os.environ.get("PARLER_PORT", "7862"))
PIPER_DIR = os.environ.get("PIPER_DIR", "/kaggle/working/piper")
HERE = os.path.dirname(os.path.abspath(__file__))

VOICES = {
    "ne_amrita": {"engine": "parler", "lang": "ne", "speaker": "Amrita", "label": "नेपाली · Amrita (female)"},
    "ne_piper": {"engine": "piper", "lang": "ne", "model": "ne_NP-google-medium", "label": "नेपाली · Piper (light)"},
    "hi_divya": {"engine": "parler", "lang": "hi", "speaker": "Divya", "label": "हिन्दी · Divya (female)"},
    "hi_rohit": {"engine": "parler", "lang": "hi", "speaker": "Rohit", "label": "हिन्दी · Rohit (male)"},
    "af_heart": {"engine": "kokoro", "lang": "en", "label": "English (US) · female, warm"},
    "af_bella": {"engine": "kokoro", "lang": "en", "label": "English (US) · female, bright"},
    "am_michael": {"engine": "kokoro", "lang": "en", "label": "English (US) · male"},
    "am_fenrir": {"engine": "kokoro", "lang": "en", "label": "English (US) · male, deep"},
    "bf_emma": {"engine": "kokoro", "lang": "en", "label": "English (UK) · female"},
    "bm_george": {"engine": "kokoro", "lang": "en", "label": "English (UK) · male"},
    "ef_dora": {"engine": "kokoro", "lang": "es", "label": "Español · female"},
    "ff_siwis": {"engine": "kokoro", "lang": "fr", "label": "Français · female"},
    "if_sara": {"engine": "kokoro", "lang": "it", "label": "Italiano · female"},
    "pf_dora": {"engine": "kokoro", "lang": "pt", "label": "Português (BR) · female"},
}
AUTO = {"en": ["af_heart"], "ne": ["ne_amrita", "ne_piper"], "hi": ["hi_divya"], "es": ["ef_dora"], "fr": ["ff_siwis"],
        "it": ["if_sara"], "pt": ["pf_dora"]}
PARLER_LANGS = {"as", "bn", "brx", "doi", "gu", "kn", "kok", "mai", "ml", "mni", "mr", "or", "sa", "sat", "sd", "ta", "te", "ur"}
STYLE = {"tragic": "in a calm, gentle and serious tone, slowly", "serious": "in a calm, serious and measured tone",
         "upbeat": "in a bright, friendly and energetic tone", "inspiring": "in a warm, confident tone"}


def parler_ok():
    return bool(PARLER_PYTHON) and os.path.exists(PARLER_PYTHON)


def piper_ok(model="ne_NP-google-medium"):
    return os.path.exists(os.path.join(PIPER_DIR, model + ".onnx"))


def available(voice):
    v = VOICES.get(voice)
    if not v:
        return False
    return {"parler": parler_ok, "piper": lambda: piper_ok(v.get("model")), "kokoro": lambda: True}[v["engine"]]()


def pick(voice, lang):
    """Resolves "auto" (or an unavailable voice) to a usable voice for the language; None means captions only."""
    if voice and voice != "auto" and voice in VOICES and available(voice):
        return voice
    for v in AUTO.get(lang, []):
        if available(v):
            return v
    if lang in PARLER_LANGS and parler_ok():
        return "indic:" + lang
    return None


def voice_options():
    out = {"auto": "Auto (match the video's language)"}
    for k, v in VOICES.items():
        if available(k):
            out[k] = v["label"]
    return out


# ---------------------------------------------------------------------- Parler worker
def _parler_health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PARLER_PORT, timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def ensure_parler(device, log=None, wait=900):
    h = _parler_health()
    if h and h.get("state") == "ready":
        return
    if not h or h.get("state") == "error":
        if not parler_ok():
            raise RuntimeError("Indic Parler-TTS is not installed (see the notebook's install cell)")
        K.spawn("vs-parler", [PARLER_PYTHON, os.path.join(HERE, "vs_parler_worker.py"), "--port", str(PARLER_PORT)],
                os.environ.get("PARLER_LOG", "/kaggle/working/logs/parler.log"),
                env=dict(os.environ, PARLER_DEVICE=device, PYTHONUNBUFFERED="1"), cwd=HERE)
        if log:
            log.info("starting the Parler TTS worker on %s", device)
    t = time.time()
    while time.time() - t < wait:
        h = _parler_health()
        if h and h.get("state") == "ready":
            return
        if h and h.get("state") == "error":
            raise RuntimeError("Parler TTS failed to load: %s" % h.get("error"))
        if not K.process_alive("vs-parler"):
            raise RuntimeError("the Parler TTS worker stopped; see /kaggle/working/logs/parler.log")
        time.sleep(3)
    raise RuntimeError("Parler TTS took too long to load")


def _parler(text, speaker, tone, path, device, log, lang=None):
    ensure_parler(device, log)
    who = ("%s speaks" % speaker) if speaker else ("A clear female speaker speaks %s" % ("in " + lang if lang else ""))
    desc = "%s %s, at a moderate pace, with a very clear, close recording and no background noise." % (
        who, STYLE.get(tone, "in a clear, natural, expressive tone"))
    req = urllib.request.Request("http://127.0.0.1:%d/tts" % PARLER_PORT, headers={"Content-Type": "application/json"},
                                 data=json.dumps({"text": text, "description": desc, "path": path, "seed": 11}).encode())
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["duration"]


def _piper(text, model, path):
    onnx = os.path.join(PIPER_DIR, model + ".onnx")
    cmds = [[sys.executable, "-m", "piper", "-m", onnx, "-f", path]]
    if shutil.which("piper"):
        cmds.append(["piper", "-m", onnx, "-f", path])
    err = ""
    for cmd in cmds:
        p = subprocess.run(cmd, input=re.sub(r"\s+", " ", text), capture_output=True, text=True, timeout=300)
        if p.returncode == 0 and os.path.exists(path):
            import vs_media
            return vs_media.duration(path)
        err = (p.stderr or p.stdout)[-400:]
    raise RuntimeError("piper failed: " + err)


def speak(text, voice, path, media, tone="neutral", device="cuda:0", log=None):
    """Writes narration to path and returns its duration in seconds."""
    if voice.startswith("indic:"):
        return _parler(text, None, tone, path, device, log, voice.split(":", 1)[1])
    v = VOICES[voice]
    if v["engine"] == "parler":
        try:
            return _parler(text, v["speaker"], tone, path, device, log)
        except Exception as e:
            if v["lang"] == "ne" and piper_ok():
                if log:
                    log.warning("Parler failed (%s); using Piper for Nepali", e)
                return _piper(text, "ne_NP-google-medium", path)
            raise
    if v["engine"] == "piper":
        return _piper(text, v["model"], path)
    return media.tts(text, voice, path, speed=0.95 if tone in ("tragic", "serious") else 1.0)
