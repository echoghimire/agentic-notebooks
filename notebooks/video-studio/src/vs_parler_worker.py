"""Indic Parler-TTS worker (Nepali, Hindi and 19 more Indian languages; Apache 2.0).

parler-tts pins its own transformers version, so this runs in a separate virtualenv that shares Kaggle's
PyTorch (created by the notebook's install cell) and is started by vs_tts.py:
    <venv>/bin/python vs_parler_worker.py --port 7862
Listens on 127.0.0.1 only. GET /health -> {"state": "loading"|"ready"|"error"};
POST /tts {"text", "description", "path", "seed"} -> {"duration"} (writes a 16-bit WAV to path).
Long text is spoken sentence by sentence (Parler is best below ~25 s per call) with short pauses between.
"""
import argparse
import json
import os
import re
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("PARLER_MODEL", "ai4bharat/indic-parler-tts")
DEVICE = os.environ.get("PARLER_DEVICE", "cuda:0")
STATE = {"state": "loading", "error": None, "model": MODEL}
M = {}
LOCK = threading.Lock()


def load():
    try:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        dev = DEVICE if torch.cuda.is_available() else "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.float16 if dev.startswith("cuda") else torch.float32).to(dev)
        M.update(model=model, tok=AutoTokenizer.from_pretrained(MODEL),
                 dtok=AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path), dev=dev,
                 rate=model.config.sampling_rate)
        STATE.update(state="ready", device=dev)
        print("parler ready on", dev, flush=True)
    except Exception as e:
        traceback.print_exc()
        STATE.update(state="error", error="%s: %s" % (type(e).__name__, str(e)[:400]))


def split(text, limit=180):
    parts = [p.strip() for p in re.split(r"(?<=[.!?।॥])\s+", text) if p.strip()]
    out = []
    for p in parts:
        while len(p) > limit:
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out or [text]


def tts(b):
    import numpy as np
    import torch
    text, desc, path = str(b["text"]), str(b["description"]), b["path"]
    model, tok, dtok, dev = M["model"], M["tok"], M["dtok"], M["dev"]
    d = dtok(desc, return_tensors="pt").to(dev)
    pieces = []
    with LOCK:
        for i, sent in enumerate(split(text)):
            torch.manual_seed(int(b.get("seed", 7)) + i)          # same voice and pacing every time
            p = tok(sent, return_tensors="pt").to(dev)
            with torch.inference_mode():
                gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                     prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            a = gen.to(torch.float32).cpu().numpy().squeeze()
            pieces += [a, np.zeros(int(0.18 * M["rate"]), np.float32)]
    audio = np.concatenate(pieces[:-1]) if pieces else np.zeros(1, np.float32)
    audio = audio / (np.abs(audio).max() + 1e-6) * 0.9
    tmp = path + ".tmp.wav"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(M["rate"])
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    os.replace(tmp, path)
    return {"duration": len(audio) / float(M["rate"])}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, STATE)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if STATE["state"] != "ready":
            return self.send(409, STATE)
        try:
            t = time.time()
            out = tts(body)
            out["seconds"] = round(time.time() - t, 2)
            self.send(200, out)
        except Exception as e:
            traceback.print_exc()
            self.send(500, {"error": "%s: %s" % (type(e).__name__, str(e)[:400])})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    srv = ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H)
    threading.Thread(target=load, daemon=True).start()
    srv.serve_forever()
