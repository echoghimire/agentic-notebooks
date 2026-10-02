"""Chat with a base model or a fine-tuned run. Started by lab_server.py as its own process:
    MODEL_PATH=<hub id or run adapter dir> MODEL_KEY=<label> python infer_worker.py --port 7861
Listens on 127.0.0.1 only. GET /health -> {"state": "loading"|"ready"|"error", "model": key};
POST /generate {"messages": [...], "max_new_tokens", "temperature", "top_p"} -> {"text", "tokens", "seconds"}.
"""
import argparse
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"state": "loading", "model": os.environ.get("MODEL_KEY", os.environ.get("MODEL_PATH")), "error": None}
M = {}
GEN_LOCK = threading.Lock()


def load():
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model, tok = FastLanguageModel.from_pretrained(model_name=os.environ["MODEL_PATH"],
                                                       max_seq_length=int(os.environ.get("MAX_SEQ", "4096")),
                                                       load_in_4bit=True, dtype=None, token=os.environ.get("HF_TOKEN"))
        if getattr(tok, "chat_template", None) is None:
            tok = get_chat_template(tok, chat_template="chatml")
        FastLanguageModel.for_inference(model)
        M.update(model=model, tok=tok)
        STATE.update(state="ready", loaded=time.time())
        print("ready:", STATE["model"], flush=True)
    except Exception as e:
        traceback.print_exc()
        STATE.update(state="error", error="%s: %s" % (type(e).__name__, str(e)[:500]))


def generate(b):
    msgs = b.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages must be a non-empty list")
    tok, model = M["tok"], M["model"]
    max_new = max(1, min(2048, int(b.get("max_new_tokens") or 512)))
    temp = float(b.get("temperature", 0.7))
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                return_dict=True)["input_ids"].to(model.device)
    t = time.time()
    with GEN_LOCK:
        kw = dict(input_ids=ids, max_new_tokens=max_new, do_sample=temp > 0)
        if temp > 0:
            kw.update(temperature=temp, top_p=float(b.get("top_p", 0.9)))
        out = model.generate(**kw)
    new = out[0][ids.shape[1]:]
    return {"text": tok.decode(new, skip_special_tokens=True).strip(), "tokens": int(new.shape[0]),
            "prompt_tokens": int(ids.shape[1]), "seconds": round(time.time() - t, 2), "model": STATE["model"]}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, STATE) if self.path == "/health" else self.send(404, {"error": "not found"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path != "/generate":
            return self.send(404, {"error": "not found"})
        if STATE["state"] != "ready":
            return self.send(409, {"error": "model is %s" % STATE["state"], **STATE})
        try:
            self.send(200, generate(json.loads(body or b"{}")))
        except ValueError as e:
            self.send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self.send(500, {"error": "%s: %s" % (type(e).__name__, e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    srv = ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H)
    threading.Thread(target=load, daemon=True).start()
    srv.serve_forever()
