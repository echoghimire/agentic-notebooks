"""Laya Studio: dataset, labelling, training and testing UI for Laya on port 7860.

Standard library HTTP server (no Gradio). Start it from the notebook with
    import laya_studio_server as S; S.start(work_dir=..., port=7860, password=..., hf_token=...)
"""
import base64
import hmac
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import laya_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODELS = [
    {"id": "convaiinnovations/laya", "label": "laya (English, 421M)"},
    {"id": "convaiinnovations/laya-typed-decisions", "label": "laya-typed-decisions (English, 421M, pre-tuned)"},
    {"id": "convaiinnovations/laya-multilingual", "label": "laya-multilingual (100+ languages incl. Nepali, 322M)"},
]
IMPORT_ROOTS = ["/kaggle/input"]
MAX_BODY = 200 * 1024 * 1024

CFG = {"work": None, "password": None, "hf_token": None}
LOCK = threading.RLock()
DATA = {"records": [], "templates": {}}
TRAIN = {"proc": None, "run": None}
AGENTS = {"key": None, "agent": None, "loading": None, "error": None}
PUSH = {}          # run name -> {"state": ..., "message": ...}
SERVERS = []


# ================================================================ storage
def p(*parts):
    return os.path.join(CFG["work"], *parts)


def _load_state():
    os.makedirs(p("runs"), exist_ok=True)
    if os.path.exists(p("dataset.jsonl")):
        recs, errs = D.parse_jsonl(open(p("dataset.jsonl"), encoding="utf-8").read())
        DATA["records"] = recs
        if errs:
            print("dataset.jsonl: skipped %d bad lines" % len(errs))
    try:
        DATA["templates"] = D.validate_templates(json.load(open(p("templates.json"))))
    except Exception:
        DATA["templates"] = D.default_templates()


def _save_records():
    tmp = p("dataset.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in DATA["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p("dataset.jsonl"))


def _safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s or "")).strip("-.")
    return s[:60]


def _run_dir(name):
    name = _safe_name(name)
    if not name or not os.path.isdir(p("runs", name)):
        raise ValueError("unknown run %r" % name)
    return p("runs", name)


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


# ================================================================ GPU / training
def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        gpus = []
        for line in out.strip().splitlines():
            n, u, t, g = [x.strip() for x in line.split(",")]
            gpus.append({"name": n, "mem_used": int(u), "mem_total": int(t), "util": int(g)})
        return gpus
    except Exception:
        return []


def training_running():
    proc = TRAIN["proc"]
    return proc is not None and proc.poll() is None


def train_status():
    run = TRAIN["run"]
    if not run:
        return {"state": "idle"}
    rd = p("runs", run)
    st = _read_json(os.path.join(rd, "status.json"), {}) or {}
    running = training_running()
    phase = st.get("phase", "starting")
    if not running and phase not in ("done", "error"):
        phase = "stopped" if TRAIN.get("stopped") else "error"
        if phase == "error" and not st.get("error"):
            st["error"] = "Training process exited early - see the log below."
    log = ""
    try:
        with open(os.path.join(rd, "train.log"), "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 12000))
            log = f.read().decode("utf-8", "replace")
    except Exception:
        pass
    st.update(run=run, state="running" if running else phase, phase=phase, log=log)
    return st


def start_training(opts):
    with LOCK:
        if training_running():
            raise ValueError("a training run is already in progress")
        if len(DATA["records"]) < 10:
            raise ValueError("add at least 10 labelled records first (you have %d)" % len(DATA["records"]))
        name = _safe_name(opts.get("name")) or time.strftime("run-%Y%m%d-%H%M%S")
        rd = p("runs", name)
        if os.path.exists(rd):
            raise ValueError("a run named %r already exists" % name)
        base = str(opts.get("base") or BASE_MODELS[0]["id"])
        if base.startswith("run:"):
            base = os.path.join(_run_dir(base[4:]), "model")
            if not os.path.exists(os.path.join(base, "model.safetensors")):
                raise ValueError("that run has no saved model")
        elif base not in [b["id"] for b in BASE_MODELS]:
            raise ValueError("unknown base model")
        os.makedirs(rd)
        shutil.copy(p("dataset.jsonl"), os.path.join(rd, "dataset.jsonl"))   # snapshot of the data used

        def num(key, default, cast, lo, hi):
            v = opts.get(key)
            v = default if v in (None, "") else cast(v)
            if not lo <= v <= hi:
                raise ValueError("%s must be between %s and %s" % (key, lo, hi))
            return v

        job = {"name": name, "run_dir": rd, "dataset": os.path.join(rd, "dataset.jsonl"), "base": base,
               "epochs": num("epochs", 4, int, 1, 50), "micro_batch": num("micro_batch", 8, int, 1, 64),
               "grad_accum": num("grad_accum", 4, int, 1, 64), "lr_encoder": num("lr_encoder", 2.5e-5, float, 1e-7, 1e-3),
               "lr_head": num("lr_head", 1e-4, float, 1e-7, 1e-2), "val_frac": num("val_frac", 0.15, float, 0.0, 0.5),
               "max_len": num("max_len", 0, int, 0, 8192), "seed": 42}
        with open(os.path.join(rd, "job.json"), "w") as f:
            json.dump(job, f, indent=2)

        _unload_agent()                       # free GPU memory for training
        n_gpu = len(gpu_info())
        want = num("gpus", n_gpu, int, 0, 16)
        n = min(n_gpu, want) if n_gpu else 0
        script = os.path.join(HERE, "train_laya.py")
        if n >= 1:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=%d" % n,
                   script, os.path.join(rd, "job.json")]
        else:
            cmd = [sys.executable, script, os.path.join(rd, "job.json")]
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
                   TOKENIZERS_PARALLELISM="false")
        env.pop("OLLAMA_HOST", None)
        logf = open(os.path.join(rd, "train.log"), "w")
        logf.write("$ %s\n" % " ".join(cmd))
        logf.flush()
        TRAIN.update(proc=subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=HERE, env=env,
                                           start_new_session=True), run=name, stopped=False)
        return {"run": name, "gpus": n}


def stop_training():
    proc = TRAIN["proc"]
    if not training_running():
        return {"stopped": False}
    TRAIN["stopped"] = True
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    return {"stopped": True}


def list_runs():
    runs = []
    for name in sorted(os.listdir(p("runs")), reverse=True):
        rd = p("runs", name)
        if not os.path.isdir(rd):
            continue
        job = _read_json(os.path.join(rd, "job.json"), {}) or {}
        st = _read_json(os.path.join(rd, "status.json"), {}) or {}
        m = _read_json(os.path.join(rd, "metrics.json"))
        has_model = os.path.exists(os.path.join(rd, "model", "model.safetensors"))
        phase = st.get("phase", "?")
        if TRAIN["run"] == name and training_running():
            phase = "running"
        elif phase not in ("done", "error"):
            phase = "stopped"
        runs.append({"name": name, "base": job.get("base"), "phase": phase, "has_model": has_model,
                     "created": os.path.getmtime(rd), "metrics": m, "push": PUSH.get(name)})
    return runs


# ================================================================ inference
def _unload_agent():
    AGENTS.update(key=None, agent=None, loading=None, error=None)
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_agent(path, device):         # replaced in tests
    import laya
    return laya.Agent(path, device=device)


def _resolve_model(key):
    if key.startswith("run:"):
        path = os.path.join(_run_dir(key[4:]), "model")
        if not os.path.exists(os.path.join(path, "model.safetensors")):
            raise ValueError("that run has no saved model yet")
        return path
    if key not in [b["id"] for b in BASE_MODELS]:
        raise ValueError("unknown model")
    return key


def get_agent(key):
    """Return a loaded agent, or None while it loads in the background (loads can exceed
    Cloudflare's 100 s request limit, so they never block a request)."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() and not training_running() else "cpu"
    except Exception:
        device = "cpu"
    full = "%s@%s" % (key, device)
    with LOCK:
        if AGENTS["key"] == full:
            if AGENTS["agent"] is not None:
                return AGENTS["agent"]
            if AGENTS["error"]:
                err = AGENTS["error"]
                AGENTS.update(key=None, error=None)
                raise ValueError("model failed to load: " + err)
            return None
        path = _resolve_model(key)
        _unload_agent()
        AGENTS.update(key=full, loading=time.time())

    def load():
        try:
            a = _load_agent(path, device)
            with LOCK:
                if AGENTS["key"] == full:
                    AGENTS.update(agent=a, loading=None)
        except Exception as e:
            traceback.print_exc()
            with LOCK:
                if AGENTS["key"] == full:
                    AGENTS.update(error=str(e), loading=None)

    threading.Thread(target=load, daemon=True).start()
    return None


def predict(body):
    key = str(body.get("model") or BASE_MODELS[0]["id"])
    text = body.get("state")
    if isinstance(text, str):
        s = text.strip()
        state = {"text": text}
        if s[:1] == "{":
            try:
                state = json.loads(s)       # allow pasting a JSON document
            except ValueError:
                pass
    else:
        state = text
    if not state:
        raise ValueError("enter a document first")
    qs = body.get("questions") or {}
    questions = {qid: D.normalize_question(qid, q) for qid, q in qs.items()}
    if not questions:
        raise ValueError("no questions")
    agent = get_agent(key)
    if agent is None:
        return {"loading": True, "device_note": "cpu" if training_running() else None}
    t0 = time.perf_counter()
    res = agent.predict(state, questions)
    return {"loading": False, "ms": round((time.perf_counter() - t0) * 1000, 1), "result": res,
            "questions": questions}


# ================================================================ HF push
def push_run(name, repo_id, private=True):
    if not CFG["hf_token"]:
        raise ValueError("add an HF_TOKEN Kaggle secret (write access) and restart the server cell")
    rd = _run_dir(name)
    if not os.path.exists(os.path.join(rd, "model", "model.safetensors")):
        raise ValueError("run has no saved model")
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo_id or ""):
        raise ValueError("repo id must look like username/model-name")
    PUSH[name] = {"state": "uploading", "message": "Uploading to %s..." % repo_id}

    def run():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=CFG["hf_token"])
            api.create_repo(repo_id, private=private, exist_ok=True)
            api.upload_folder(folder_path=os.path.join(rd, "model"), repo_id=repo_id,
                              commit_message="Laya Studio run %s" % name)
            for extra in ("metrics.json",):
                if os.path.exists(os.path.join(rd, extra)):
                    api.upload_file(path_or_fileobj=os.path.join(rd, extra), path_in_repo=extra, repo_id=repo_id)
            PUSH[name] = {"state": "done", "message": "Pushed to https://huggingface.co/%s" % repo_id}
        except Exception as e:
            PUSH[name] = {"state": "error", "message": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return PUSH[name]


# ================================================================ dataset import
def import_records(recs, mode):
    with LOCK:
        if mode == "replace":
            DATA["records"] = list(recs)
        else:
            DATA["records"].extend(recs)
        _save_records()
        return len(DATA["records"])


def parse_upload(text, fmt, template):
    if fmt == "csv":
        tpl = DATA["templates"].get(template)
        if not tpl:
            raise ValueError("pick a template for CSV import")
        return D.parse_csv(text, tpl["questions"])
    return D.parse_jsonl(text)


def list_sources():
    out = []
    for root in IMPORT_ROOTS:
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith((".jsonl", ".csv")):
                    fp = os.path.join(dp, f)
                    out.append({"path": fp, "size": os.path.getsize(fp)})
                    if len(out) >= 200:
                        return out
    return out


# ================================================================ HTTP
class ChunkWriter:
    """File-like object that sends HTTP chunked-encoding frames (zipfile streams into it)."""

    def __init__(self, wfile):
        self.w = wfile

    def write(self, data):
        if data:
            self.w.write(b"%x\r\n" % len(data) + bytes(data) + b"\r\n")
        return len(data)

    def flush(self):
        self.w.flush()

    def close(self):
        self.w.write(b"0\r\n\r\n")
        self.w.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LayaStudio"

    def log_message(self, fmt, *args):
        pass

    # ---------------------------------------------------------- helpers
    def _authed(self):
        pw = CFG["password"]
        if not pw:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                user, _, given = base64.b64decode(h[6:]).decode("utf-8").partition(":")
                if hmac.compare_digest(given.encode(), pw.encode()):
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Laya Studio", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body, ensure_ascii=False, default=str)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("upload too large")
        return self.rfile.read(n) if n else b""

    def _json(self):
        raw = self._body()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _stream_file(self, path, name, ctype):
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1024 * 1024)

    def _stream_zip(self, folder, name):
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
        self.end_headers()
        cw = ChunkWriter(self.wfile)
        with zipfile.ZipFile(cw, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for dp, dn, fn in os.walk(folder):
                for f in sorted(fn):
                    if f.endswith(".tmp"):
                        continue
                    fp = os.path.join(dp, f)
                    z.write(fp, os.path.relpath(fp, os.path.dirname(folder)))
        cw.close()

    # ---------------------------------------------------------- routes
    def do_GET(self):
        if not self._authed():
            return
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "laya_studio_ui.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html")
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if u.path == "/api/info":
                with LOCK:
                    stats = D.dataset_stats(DATA["records"])
                    templates = DATA["templates"]
                return self._send(200, {
                    "stats": stats, "templates": templates, "base_models": BASE_MODELS, "gpus": gpu_info(),
                    "hf": bool(CFG["hf_token"]), "train": train_status(), "work": CFG["work"],
                    "csv_examples": {t: D.csv_example(v["questions"]) for t, v in templates.items()}})
            if u.path == "/api/records":
                off, lim = int(qs.get("offset", 0)), min(200, int(qs.get("limit", 50)))
                flt = qs.get("q", "").lower()
                with LOCK:
                    rows = [(i, r) for i, r in enumerate(DATA["records"])]
                if flt:
                    rows = [(i, r) for i, r in rows
                            if flt in json.dumps([r["state"], r["gold"]], ensure_ascii=False).lower()]
                out = []
                for i, r in rows[off: off + lim]:
                    labels = {}
                    for qid, g in r["gold"].items():
                        try:
                            labels[qid] = D.gold_display(r["questions"][qid], g)
                        except ValueError:
                            labels[qid] = "?"
                    out.append({"index": i, "preview": D.state_preview(r["state"]), "labels": labels})
                return self._send(200, {"total": len(rows), "rows": out})
            if u.path == "/api/record":
                with LOCK:
                    return self._send(200, DATA["records"][int(qs["index"])])
            if u.path == "/api/dataset/download":
                with LOCK:
                    _save_records()
                return self._stream_file(p("dataset.jsonl"), "laya_dataset.jsonl", "application/x-ndjson")
            if u.path == "/api/dataset/sources":
                return self._send(200, {"files": list_sources()})
            if u.path == "/api/train/status":
                return self._send(200, train_status())
            if u.path == "/api/runs":
                return self._send(200, {"runs": list_runs()})
            if u.path.startswith("/api/runs/") and u.path.endswith("/download"):
                name = unquote(u.path.split("/")[3])
                rd = _run_dir(name)
                return self._stream_zip(rd, "laya-%s.zip" % _safe_name(name))
            if u.path.startswith("/api/runs/") and u.path.endswith("/questions"):
                rd = _run_dir(unquote(u.path.split("/")[3]))
                return self._send(200, _read_json(os.path.join(rd, "model", "questions.json"), {}))
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, KeyError, IndexError) as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        if not self._authed():
            return
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/dataset/upload":
                text = self._body().decode("utf-8-sig", "replace")
                recs, errs = parse_upload(text, qs.get("format", "jsonl"), qs.get("template"))
                total = import_records(recs, qs.get("mode", "append")) if recs else len(DATA["records"])
                return self._send(200, {"added": len(recs), "total": total, "errors": errs[:50],
                                        "error_count": len(errs)})
            if u.path == "/api/dataset/import":
                b = self._json()
                path = os.path.realpath(b.get("path", ""))
                if not any(path.startswith(os.path.realpath(r) + os.sep) for r in IMPORT_ROOTS):
                    raise ValueError("can only import files from /kaggle/input")
                text = open(path, encoding="utf-8-sig", errors="replace").read()
                fmt = "csv" if path.lower().endswith(".csv") else "jsonl"
                recs, errs = parse_upload(text, fmt, b.get("template"))
                total = import_records(recs, b.get("mode", "append")) if recs else len(DATA["records"])
                return self._send(200, {"added": len(recs), "total": total, "errors": errs[:50],
                                        "error_count": len(errs)})
            if u.path == "/api/records/add":
                b = self._json()
                rec = D.normalize_record(b)
                return self._send(200, {"total": import_records([rec], "append")})
            if u.path == "/api/records/delete":
                b = self._json()
                idx = sorted({int(i) for i in b.get("indices", [])}, reverse=True)
                with LOCK:
                    for i in idx:
                        if 0 <= i < len(DATA["records"]):
                            del DATA["records"][i]
                    _save_records()
                    return self._send(200, {"total": len(DATA["records"])})
            if u.path == "/api/records/clear":
                with LOCK:
                    DATA["records"] = []
                    _save_records()
                return self._send(200, {"total": 0})
            if u.path == "/api/templates":
                t = D.validate_templates(self._json())
                with LOCK:
                    DATA["templates"] = t
                    with open(p("templates.json"), "w") as f:
                        json.dump(t, f, indent=2, ensure_ascii=False)
                return self._send(200, {"templates": t})
            if u.path == "/api/templates/reset":
                with LOCK:
                    DATA["templates"] = D.default_templates()
                    if os.path.exists(p("templates.json")):
                        os.remove(p("templates.json"))
                return self._send(200, {"templates": DATA["templates"]})
            if u.path == "/api/train/start":
                return self._send(200, start_training(self._json()))
            if u.path == "/api/train/stop":
                return self._send(200, stop_training())
            if u.path == "/api/predict":
                return self._send(200, predict(self._json()))
            if u.path.startswith("/api/runs/") and u.path.endswith("/push"):
                b = self._json()
                return self._send(200, push_run(unquote(u.path.split("/")[3]), b.get("repo_id"), bool(b.get("private", True))))
            if u.path.startswith("/api/runs/") and u.path.endswith("/delete"):
                name = unquote(u.path.split("/")[3])
                if TRAIN["run"] == name and training_running():
                    raise ValueError("stop the run first")
                rd = _run_dir(name)
                if AGENTS["key"] and AGENTS["key"].startswith("run:%s@" % name):
                    _unload_agent()
                shutil.rmtree(rd)
                return self._send(200, {"deleted": name})
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, KeyError, IndexError, TypeError) as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})


class V4Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class V6Server(V4Server):
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


def start(work_dir, port=7860, password=None, hf_token=None):
    """Start (or restart) the UI on IPv4 and IPv6 so cloudflared's 'localhost' always connects."""
    stop()
    CFG.update(work=os.path.abspath(work_dir), password=password or None, hf_token=hf_token or None)
    os.makedirs(CFG["work"], exist_ok=True)
    _load_state()
    for cls, addr in ((V4Server, "0.0.0.0"), (V6Server, "::")):
        try:
            srv = cls((addr, port), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            SERVERS.append(srv)
        except OSError as e:
            print("Could not listen on [%s]:%d -> %s" % (addr, port, e))
    if not SERVERS:
        raise RuntimeError("nothing is listening on port %d" % port)
    return SERVERS


def stop():
    while SERVERS:
        srv = SERVERS.pop()
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            pass
