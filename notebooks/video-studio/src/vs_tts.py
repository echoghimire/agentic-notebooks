"""Narration voices for Video Studio, by language.

- Kokoro-82M (Apache 2.0): English and a few European languages, in the server process (CPU is fine).
- Indic Parler-TTS (Apache 2.0): Nepali, Hindi and other Indian languages, in vs_parler_worker.py (its own
  virtualenv and process, on the media GPU), with a named speaker so the voice stays the same in every scene.
- Piper (MIT; voice from the OpenSLR Nepali corpus): a light CPU Nepali voice, used when Parler is unavailable.
- Svara-TTS v1 (Apache 2.0, Kenpath): male and female voices for 19 Indic languages including Nepali; a Llama-3.2-3B
  speech model whose SNAC codes are decoded to 24 kHz audio, run here with plain transformers on the media GPU.
- Microsoft Edge neural voices (online, opt-in with ONLINE_VOICES=1): very natural Nepali / Hindi / English voices
  through the edge-tts client. Not open source, and text is sent to Microsoft.
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
SVARA_MODEL = os.environ.get("SVARA_MODEL", "")                 # e.g. kenpath/svara-tts-v1 ("" = off)
ONLINE_VOICES = os.environ.get("ONLINE_VOICES", "0") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))

VOICES = {
    "ne_amrita": {"engine": "parler", "lang": "ne", "speaker": "Amrita", "label": "नेपाली · Amrita (female)"},
    "ne_piper": {"engine": "piper", "lang": "ne", "model": "ne_NP-google-medium", "label": "नेपाली · Piper (light)"},
    "ne_svara_f": {"engine": "svara", "lang": "ne", "speaker": "Nepali (Female)", "label": "नेपाली · Svara (female)"},
    "ne_svara_m": {"engine": "svara", "lang": "ne", "speaker": "Nepali (Male)", "label": "नेपाली · Svara (male)"},
    "ne_hemkala": {"engine": "edge", "lang": "ne", "speaker": "ne-NP-HemkalaNeural", "label": "नेपाली · Hemkala (female, online)"},
    "ne_sagar": {"engine": "edge", "lang": "ne", "speaker": "ne-NP-SagarNeural", "label": "नेपाली · Sagar (male, online)"},
    "hi_divya": {"engine": "parler", "lang": "hi", "speaker": "Divya", "label": "हिन्दी · Divya (female)"},
    "hi_svara_f": {"engine": "svara", "lang": "hi", "speaker": "Hindi (Female)", "label": "हिन्दी · Svara (female)"},
    "hi_svara_m": {"engine": "svara", "lang": "hi", "speaker": "Hindi (Male)", "label": "हिन्दी · Svara (male)"},
    "hi_swara": {"engine": "edge", "lang": "hi", "speaker": "hi-IN-SwaraNeural", "label": "हिन्दी · Swara (female, online)"},
    "hi_madhur": {"engine": "edge", "lang": "hi", "speaker": "hi-IN-MadhurNeural", "label": "हिन्दी · Madhur (male, online)"},
    "en_aria": {"engine": "edge", "lang": "en", "speaker": "en-US-AriaNeural", "label": "English (US) · Aria (female, online)"},
    "en_guy": {"engine": "edge", "lang": "en", "speaker": "en-US-GuyNeural", "label": "English (US) · Guy (male, online)"},
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
AUTO = {"en": ["af_heart", "en_aria"], "ne": ["ne_amrita", "ne_svara_f", "ne_hemkala", "ne_piper"],
        "hi": ["hi_divya", "hi_svara_f", "hi_swara"], "es": ["ef_dora"], "fr": ["ff_siwis"],
        "it": ["if_sara"], "pt": ["pf_dora"]}
PARLER_LANGS = {"as", "bn", "brx", "doi", "gu", "kn", "kok", "mai", "ml", "mni", "mr", "or", "sa", "sat", "sd", "ta", "te", "ur"}
STYLE = {"tragic": "in a calm, gentle and serious tone, slowly", "serious": "in a calm, serious and measured tone",
         "upbeat": "in a bright, friendly and energetic tone", "inspiring": "in a warm, confident tone"}


NE_0_99 = ("शून्य एक दुई तीन चार पाँच छ सात आठ नौ दस एघार बाह्र तेह्र चौध पन्ध्र सोह्र सत्र अठार उन्नाइस बीस एक्काइस बाइस "
           "तेइस चौबीस पच्चीस छब्बीस सत्ताइस अट्ठाइस उनन्तीस तीस एकतीस बत्तीस तेत्तीस चौंतीस पैंतीस छत्तीस सैंतीस अठतीस "
           "उनन्चालीस चालीस एकचालीस बयालीस त्रिचालीस चवालीस पैंतालीस छयालीस सतचालीस अठचालीस उनन्चास पचास एकाउन्न बाउन्न "
           "त्रिपन्न चउन्न पचपन्न छपन्न सन्ताउन्न अन्ठाउन्न उनन्साठी साठी एकसट्ठी बयसट्ठी त्रिसट्ठी चौसट्ठी पैंसट्ठी छयसट्ठी "
           "सतसट्ठी अठसट्ठी उनन्सत्तरी सत्तरी एकहत्तर बहत्तर त्रिहत्तर चौहत्तर पचहत्तर छयहत्तर सतहत्तर अठहत्तर उनासी असी "
           "एकासी बयासी त्रियासी चौरासी पचासी छयासी सतासी अठासी उनान्नब्बे नब्बे एकानब्बे बयानब्बे त्रियानब्बे चौरानब्बे "
           "पन्चानब्बे छयानब्बे सन्तानब्बे अन्ठानब्बे उनान्सय").split()
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def ne_number(n):
    """Integer -> Nepali words, in the Nepali system (सय, हजार, लाख, करोड): 2082 -> दुई हजार बयासी."""
    if n < 100:
        return NE_0_99[n]
    out = []
    for size, name in ((10 ** 7, "करोड"), (10 ** 5, "लाख"), (1000, "हजार"), (100, "सय")):
        if n >= size:
            q, n = divmod(n, size)
            out.append((ne_number(q) if q >= 100 else NE_0_99[q]) + " " + name)
    if n:
        out.append(NE_0_99[n])
    return " ".join(out)


def speakable(text, lang):
    """Text as it should be read aloud: Nepali numbers, percentages and decimals written out as words, so the
    voice reads them instead of guessing (५ जना -> पाँच जना, ३.५% -> तीन दशमलव पाँच प्रतिशत)."""
    if lang != "ne":
        return text
    t = text.translate(DEVA_DIGITS)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)                    # 1,25,000 -> 125000

    def num(m):
        whole, frac, pct = m.group(1), m.group(2), m.group(3)
        if len(whole) > 12:
            return m.group(0)
        w = ne_number(int(whole))
        if frac:
            w += " दशमलव " + " ".join(NE_0_99[int(c)] for c in frac)
        return w + (" प्रतिशत" if pct else "")
    t = re.sub(r"(\d+)(?:\.(\d+))?(?:\s*(%))?", num, t)
    return re.sub(r"\s+", " ", t.replace("&", " र ")).strip()


def parler_ok():
    return bool(PARLER_PYTHON) and os.path.exists(PARLER_PYTHON)


def piper_ok(model="ne_NP-google-medium"):
    import importlib.util
    return (importlib.util.find_spec("piper") is not None or bool(shutil.which("piper"))) and \
        os.path.exists(os.path.join(PIPER_DIR, model + ".onnx"))


def svara_ok():
    import importlib.util
    return bool(SVARA_MODEL) and importlib.util.find_spec("snac") is not None


def edge_ok():
    return ONLINE_VOICES and bool(shutil.which("edge-tts"))


def kokoro_ok():
    import importlib.util
    return importlib.util.find_spec("kokoro") is not None


def available(voice):
    v = VOICES.get(voice)
    if not v:
        return False
    return {"parler": parler_ok, "piper": lambda: piper_ok(v.get("model")), "kokoro": kokoro_ok, "svara": svara_ok,
            "edge": edge_ok}[v["engine"]]()


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


# ---------------------------------------------------------------------- Svara (transformers, in process)
_SVARA = {}
SOS, EOS_SPEECH, SOH, EOH, SOAI, EOT, AUDIO, BOS = 128257, 128258, 128259, 128260, 128261, 128009, 156939, 128000
CODE_BASE = 128266


def _svara_load(device, log):
    if "m" not in _SVARA:
        import torch
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SVARA_MODEL)
        model = AutoModelForCausalLM.from_pretrained(SVARA_MODEL, torch_dtype=torch.float16).to(device).eval()
        snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        _SVARA.update(m=model, tok=tok, snac=snac)
        if log:
            log.info("loaded %s on %s", SVARA_MODEL, device)
    return _SVARA["m"], _SVARA["tok"], _SVARA["snac"]


def svara_unload():
    _SVARA.clear()


def _svara_codes(ids):
    """Generated token ids -> the three SNAC code streams (7 tokens per frame: c0, c1, c2, c2, c1, c2, c2)."""
    codes = [t - CODE_BASE for t in ids if t >= CODE_BASE]
    codes = [c - (k % 7) * 4096 for k, c in enumerate(codes)]
    n = len(codes) // 7
    if not n or any(c < 0 or c > 4095 for c in codes[:n * 7]):
        raise RuntimeError("Svara produced invalid audio codes")
    f = [codes[7 * i:7 * i + 7] for i in range(n)]
    return ([x[0] for x in f], [y for x in f for y in (x[1], x[4])], [y for x in f for y in (x[2], x[3], x[5], x[6])])


def _svara(text, speaker, path, device, log):
    import numpy as np
    import torch
    import vs_media
    model, tok, snac = _svara_load(device, log)
    parts = []
    for sent in [x for x in re.split(r"(?<=[.!?।॥])\s+", text.strip()) if x.strip()]:
        ids = tok("%s: %s" % (speaker, sent), add_special_tokens=False).input_ids
        prompt = [BOS, SOH, AUDIO] + ids + [EOH, EOT, SOAI, SOS]
        x = torch.tensor([prompt], device=device)
        with torch.no_grad():
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=min(4000, 120 + 26 * len(sent)),
                                 do_sample=True, temperature=0.75, top_p=0.9, repetition_penalty=1.1,
                                 eos_token_id=[EOS_SPEECH], pad_token_id=EOS_SPEECH)
        c0, c1, c2 = _svara_codes(out[0, len(prompt):].tolist())
        with torch.no_grad():
            audio = snac.decode([torch.tensor([c], device=device) for c in (c0, c1, c2)])
        parts += [audio[0, 0].float().cpu().numpy(), np.zeros(int(0.14 * 24000), np.float32)]
    if not parts:
        raise RuntimeError("no text to speak")
    data = np.concatenate(parts)
    vs_media.write_wav(path, data / (np.abs(data).max() + 1e-6) * 0.9, 24000)
    return len(data) / 24000.0


# ---------------------------------------------------------------------- Edge (online, opt-in)
def _edge(text, speaker, path, tone):
    import vs_media
    mp3 = path[:-4] + ".mp3"
    rate = "-6%" if tone in ("tragic", "serious") else "+0%"
    p = subprocess.run(["edge-tts", "--voice", speaker, "--rate=" + rate, "--text", re.sub(r"\s+", " ", text),
                        "--write-media", mp3], capture_output=True, text=True, timeout=180)
    if p.returncode != 0 or not os.path.exists(mp3):
        raise RuntimeError("online voice failed: " + (p.stderr or p.stdout)[-300:])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "24000", path], check=True, timeout=120)
    os.remove(mp3)
    return vs_media.duration(path)


def speak(text, voice, path, media, tone="neutral", device="cuda:0", log=None):
    """Writes narration to path and returns its duration in seconds."""
    text = speakable(text, voice.split(":", 1)[1] if voice.startswith("indic:") else VOICES.get(voice, {}).get("lang", ""))
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
    if v["engine"] in ("svara", "edge"):
        try:
            return _svara(text, v["speaker"], path, device, log) if v["engine"] == "svara" else _edge(text, v["speaker"], path, tone)
        except Exception as e:
            fb = next((x for x in AUTO.get(v["lang"], []) if VOICES[x]["engine"] not in ("svara", "edge") and available(x)), None)
            if not fb:
                raise
            if log:
                log.warning("%s failed (%s); using %s", voice, e, fb)
            return speak(text, fb, path, media, tone, device, log)
    return media.tts(text, voice, path, speed=0.95 if tone in ("tragic", "serious") else 1.0)
