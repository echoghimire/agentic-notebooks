"""Unsloth Fine-tuning Lab: the minimalist LLM factory on port 7860.

Runs as its own process (started by the notebook with studio_http.spawn):
    APP_PASSWORD=... HF_TOKEN=... python lab_server.py --port 7860

Dataset in (upload, Hugging Face Hub, or added by an agent) -> 4-bit LoRA fine-tune with Unsloth ->
chat with the result -> export (LoRA adapter, merged 16-bit, GGUF) -> download or push to the Hub.

GPU plan with two GPUs: training on GPU 0, chat and exports on GPU 1. With one GPU, chat and exports
wait until training is finished. Training, chat and exports each run in their own process, so a crash
or an out-of-memory error never takes the server down, and they keep running if the server restarts.
"""
import argparse
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

import lab_data as D
import studio_http as K

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR", "/kaggle/working/unsloth_lab")
EXPORTS = os.environ.get("EXPORT_DIR", "/tmp/unsloth_lab/exports")     # big files: outside /kaggle/working's 20 GB
LOGS = os.environ.get("LOG_DIR", "/kaggle/working/logs")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
INFER_PORT = int(os.environ.get("INFER_PORT", "7861"))
MAX_WAIT = 85
BASE_MODELS = [
    {"id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", "label": "Llama 3.2 1B Instruct (fastest)"},
    {"id": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "label": "Llama 3.2 3B Instruct"},
    {"id": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", "label": "Qwen2.5 1.5B Instruct"},
    {"id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "label": "Qwen2.5 3B Instruct"},
    {"id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "label": "Qwen2.5 7B Instruct (T4: short sequences)"},
    {"id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "label": "Llama 3.1 8B Instruct (T4: short sequences)"},
    {"id": "unsloth/gemma-2-2b-it-bnb-4bit", "label": "Gemma 2 2B it"},
    {"id": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "label": "Mistral 7B Instruct v0.3"},
]
EXPORT_FORMATS = ["merged_16bit", "gguf_q4_k_m", "gguf_q8_0", "gguf_f16"]
HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
log = K.file_logger(os.environ.get("APP_LOG", os.path.join(LOGS, "unsloth_lab.log")), "lab")

LOCK = threading.RLock()
DATA = {"records": []}
IMPORT = {"state": "idle"}
PUSH = {}
INFER = {"key": None, "since": 0}


# ====================================================================== storage
def p(*parts):
    return os.path.join(WORK, *parts)


def load_dataset():
    recs = []
    if os.path.exists(p("dataset.jsonl")):
        recs, errs = D.parse(open(p("dataset.jsonl"), encoding="utf-8").read(), "jsonl")
        if errs:
            log.warning("dataset.jsonl: skipped %d bad lines", len(errs))
    DATA["records"] = recs


def save_dataset():
    tmp = p("dataset.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in DATA["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p("dataset.jsonl"))


def add_records(recs, mode="append"):
    if mode not in ("append", "replace"):
        raise ValueError("mode must be append or replace")
    with LOCK:
        DATA["records"] = list(recs) if mode == "replace" else DATA["records"] + list(recs)
        save_dataset()
        return len(DATA["records"])


def dataset_info():
    with LOCK:
        return dict(D.stats(DATA["records"]), import_job=dict(IMPORT))


def import_hf(name, split="train", config=None, max_rows=5000, mode="append"):
    if not HUB_ID.match(str(name or "")):
        raise ValueError("name must look like owner/dataset")
    max_rows = max(1, min(200000, int(max_rows or 5000)))
    if IMPORT.get("state") == "running":
        raise ValueError("an import is already running")
    IMPORT.clear()
    IMPORT.update(state="running", name=name, split=split, started=time.time())

    def go():
        try:
            from datasets import load_dataset
            ds = load_dataset(name, config or None, split="%s[:%d]" % (split, max_rows), token=HF_TOKEN)
            recs, errs = [], 0
            for row in ds:
                try:
                    recs.append(D.normalize(dict(row)))
                except ValueError:
                    errs += 1
            if not recs:
                raise ValueError("no usable rows (columns: %s)" % ", ".join(ds.column_names))
            total = add_records(recs, mode)
            IMPORT.update(state="done", added=len(recs), skipped=errs, total=total)
            log.info("imported %d rows from %s", len(recs), name)
        except Exception as e:
            log.error("import failed\n%s", traceback.format_exc())
            IMPORT.update(state="error", error=str(e)[:500])
    threading.Thread(target=go, daemon=True).start()
    return dict(IMPORT)


# ====================================================================== GPUs
def gpu_count():
    return len(K.gpu_info())


def training_running():
    return K.process_alive("lab-train")


def side_gpu():
    """GPU for chat and exports: GPU 1 with two GPUs; GPU 0 only while no training runs."""
    n = gpu_count()
    if n >= 2:
        return "1"
    if n == 1:
        if training_running():
            raise ValueError("training is using the only GPU; chat and exports are available when it finishes "
                             "(or pick the 'GPU T4 x2' accelerator)")
        return "0"
    raise ValueError("no GPU: pick a GPU accelerator in the notebook settings")


# ====================================================================== training
def run_dir(name):
    name = K.safe_name(name)
    if not name or not os.path.isfile(p("runs", name, "job.json")):
        raise K.HTTPError(404, "unknown run %r" % name)
    return p("runs", name)


def current_run():
    return (K.read_json(p("current_run.json"), {}) or {}).get("run")


def start_training(o):
    with LOCK:
        if training_running():
            raise ValueError("a training run is already in progress")
        if not gpu_count():
            raise ValueError("no GPU: pick a GPU accelerator in the notebook settings")
        if len(DATA["records"]) < 10:
            raise ValueError("add at least 10 examples first (you have %d)" % len(DATA["records"]))
        name = K.safe_name(o.get("name")) or time.strftime("run-%Y%m%d-%H%M%S")
        if os.path.exists(p("runs", name)):
            raise ValueError("a run named %r already exists" % name)
        base = str(o.get("base_model") or BASE_MODELS[0]["id"]).strip()
        if base.startswith("run:"):
            prev = os.path.join(run_dir(base[4:]), "adapter")
            if not os.path.exists(os.path.join(prev, "adapter_config.json")):
                raise ValueError("that run has no saved adapter")
            base = prev
        elif not HUB_ID.match(base):
            raise ValueError("base_model must be a Hugging Face id like unsloth/Llama-3.2-3B-Instruct-bnb-4bit")

        def num(key, default, cast, lo, hi):
            v = o.get(key)
            v = default if v in (None, "") else cast(v)
            if not lo <= v <= hi:
                raise ValueError("%s must be between %s and %s" % (key, lo, hi))
            return v
        rd = p("runs", name)
        os.makedirs(rd)
        shutil.copy(p("dataset.jsonl"), os.path.join(rd, "dataset.jsonl"))
        job = {"name": name, "run_dir": rd, "dataset": os.path.join(rd, "dataset.jsonl"), "base": base,
               "base_label": str(o.get("base_model") or BASE_MODELS[0]["id"]),
               "epochs": num("epochs", 2, float, 0.1, 50), "max_steps": num("max_steps", 0, int, 0, 100000),
               "learning_rate": num("learning_rate", 2e-4, float, 1e-6, 1e-2),
               "lora_r": num("lora_r", 16, int, 1, 256), "lora_alpha": num("lora_alpha", 16, int, 1, 512),
               "max_seq_length": num("max_seq_length", 2048, int, 128, 32768),
               "batch_size": num("batch_size", 2, int, 1, 64), "grad_accum": num("grad_accum", 4, int, 1, 128),
               "val_frac": num("val_frac", 0.05, float, 0.0, 0.5), "seed": num("seed", 3407, int, 0, 2 ** 31),
               "created": time.time(), "examples": len(DATA["records"])}
        K.write_json(os.path.join(rd, "job.json"), job)
        K.write_json(os.path.join(rd, "status.json"), {"phase": "starting"})
        if gpu_count() == 1:
            stop_infer()                        # free the only GPU
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        K.spawn("lab-train", [sys.executable, os.path.join(HERE, "train_unsloth.py"), os.path.join(rd, "job.json")],
                os.path.join(rd, "train.log"), env=env, cwd=HERE)
        K.write_json(p("current_run.json"), {"run": name})
        log.info("training %s on %s", name, base)
        return {"run": name, "job": job}


def run_status(name):
    rd = p("runs", name)
    st = K.read_json(os.path.join(rd, "status.json"), {}) or {}
    phase = st.get("phase", "?")
    live = name == current_run() and training_running()
    if not live and phase not in ("done", "error", "stopped"):
        phase = "error"
        st.setdefault("error", "the training process exited early; see the log")
    st["phase"] = phase
    st["state"] = "running" if live else phase
    return st


def train_status(with_log=True):
    name = current_run()
    if not name or not os.path.isdir(p("runs", name)):
        return {"state": "idle"}
    st = run_status(name)
    st["run"] = name
    if with_log:
        st["log"] = K.tail(os.path.join(p("runs", name), "train.log"), 40)
    return st


def stop_training(force=False):
    info = K.read_json(os.path.join(K.PID_DIR, "lab-train.json"))
    if not info or not training_running():
        return {"stopped": False, "reason": "no training is running"}
    if force:
        K.stop_process("lab-train", timeout=5)
        return {"stopped": True, "note": "killed; the adapter was not saved"}
    os.killpg(info["pid"], signal.SIGTERM)     # train_unsloth.py finishes the step, then saves the adapter
    return {"stopped": True, "note": "stopping after the current step; the adapter trained so far will be saved"}


def list_runs():
    out = []
    if not os.path.isdir(p("runs")):
        return out
    for name in sorted(os.listdir(p("runs")), reverse=True):
        rd = p("runs", name)
        job = K.read_json(os.path.join(rd, "job.json"))
        if not job:
            continue
        st = run_status(name)
        out.append({"name": name, "base": job.get("base_label"), "state": st["state"], "created": job["created"],
                    "examples": job.get("examples"), "metrics": K.read_json(os.path.join(rd, "metrics.json")),
                    "has_adapter": os.path.exists(os.path.join(rd, "adapter", "adapter_config.json")),
                    "exports": export_list(name), "push": PUSH.get(name),
                    "params": {k: job[k] for k in ("epochs", "max_steps", "learning_rate", "lora_r", "lora_alpha",
                                                   "max_seq_length", "batch_size", "grad_accum")}})
    return out


def run_detail(name):
    rd = run_dir(name)
    return {"name": name, "job": K.read_json(os.path.join(rd, "job.json")), "status": run_status(name),
            "metrics": K.read_json(os.path.join(rd, "metrics.json")),
            "samples": K.read_json(os.path.join(rd, "samples.json"), []), "exports": export_list(name)}


def delete_run(name):
    rd = run_dir(name)
    if name == current_run() and training_running():
        raise ValueError("stop the run first")
    if INFER["key"] == "run:" + name:
        stop_infer()
    shutil.rmtree(rd)
    shutil.rmtree(os.path.join(EXPORTS, K.safe_name(name)), ignore_errors=True)
    return {"deleted": name}


# ====================================================================== chat (inference worker)
def infer_health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % INFER_PORT, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def stop_infer():
    K.stop_process("lab-infer")
    INFER.update(key=None, since=0)


def model_path(key):
    if key.startswith("run:"):
        path = os.path.join(run_dir(key[4:]), "adapter")
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            raise ValueError("that run has no saved adapter yet")
        return path
    if not HUB_ID.match(key):
        raise ValueError("model must be a Hugging Face id or run:<name>")
    return key


def chat(model, messages, max_new_tokens=512, temperature=0.7):
    key = str(model or "").strip() or BASE_MODELS[0]["id"]
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    msgs = D.normalize({"messages": list(messages) + [{"role": "assistant", "content": "x"}]})["messages"][:-1]
    h = infer_health()
    if h and h.get("model") == key:
        if h["state"] == "ready":
            body = json.dumps({"messages": msgs, "max_new_tokens": min(int(max_new_tokens or 512), 1024),
                               "temperature": float(temperature)}).encode()
            req = urllib.request.Request("http://127.0.0.1:%d/generate" % INFER_PORT, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=MAX_WAIT) as r:
                    return dict(json.loads(r.read()), loading=False)
            except urllib.error.HTTPError as e:
                raise ValueError(json.loads(e.read() or b"{}").get("error", "generation failed"))
            except OSError:
                raise ValueError("generation took longer than %d s; ask for fewer max_new_tokens" % MAX_WAIT)
        if h["state"] == "error":
            stop_infer()
            raise ValueError("model failed to load: %s" % h.get("error"))
        return {"loading": True, "model": key, "message": "loading the model; ask again in a minute"}
    if INFER["key"] == key and time.time() - INFER["since"] < 600 and K.process_alive("lab-infer"):
        return {"loading": True, "model": key, "message": "starting the model; ask again in a minute"}
    path = model_path(key)
    gpu = side_gpu()
    with LOCK:
        if K.process_alive("lab-export"):
            raise ValueError("an export is using the GPU; chat is available when it finishes")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MODEL_PATH=path, MODEL_KEY=key, PYTHONUNBUFFERED="1",
                   TOKENIZERS_PARALLELISM="false")
        K.spawn("lab-infer", [sys.executable, os.path.join(HERE, "infer_worker.py"), "--port", str(INFER_PORT)],
                os.path.join(LOGS, "unsloth_infer.log"), env=env, cwd=HERE)
        INFER.update(key=key, since=time.time())
    log.info("loading %s for chat on GPU %s", key, gpu)
    return {"loading": True, "model": key, "message": "loading the model (1-3 minutes); ask again then"}


def chat_status():
    h = infer_health()
    if h:
        return h
    if INFER["key"] and K.process_alive("lab-infer"):
        return {"state": "loading", "model": INFER["key"]}
    return {"state": "off"}


# ====================================================================== exports and Hub
def export_dir(name, fmt):
    return os.path.join(EXPORTS, K.safe_name(name), fmt)


def export_list(name):
    out = {}
    for fmt in EXPORT_FORMATS:
        st = K.read_json(os.path.join(export_dir(name, fmt), "status.json"))
        if st:
            if st.get("phase") not in ("done", "error") and not K.process_alive("lab-export"):
                st = dict(st, phase="error", error=st.get("error") or "the export process stopped; see the log")
            out[fmt] = st
    return out


def start_export(name, fmt):
    if fmt not in EXPORT_FORMATS:
        raise ValueError("format must be one of " + ", ".join(EXPORT_FORMATS))
    rd = run_dir(name)
    adapter = os.path.join(rd, "adapter")
    if not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        raise ValueError("that run has no saved adapter")
    if K.process_alive("lab-export"):
        raise ValueError("another export is running; one at a time")
    gpu = side_gpu()
    stop_infer()                                   # exports need the GPU memory
    out = export_dir(name, fmt)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    job = K.read_json(os.path.join(rd, "job.json"))
    K.write_json(os.path.join(out, "job.json"), {"adapter": adapter, "format": fmt, "export_dir": out,
                                                 "max_seq_length": job["max_seq_length"]})
    K.write_json(os.path.join(out, "status.json"), {"phase": "starting", "started": time.time()})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
    K.spawn("lab-export", [sys.executable, os.path.join(HERE, "export_unsloth.py"), os.path.join(out, "job.json")],
            os.path.join(LOGS, "unsloth_export.log"), env=env, cwd=HERE)
    return {"run": name, "format": fmt, "state": "starting"}


def artifact_dir(name, what):
    if what == "adapter":
        d = os.path.join(run_dir(name), "adapter")
    elif what in EXPORT_FORMATS:
        d = os.path.join(export_dir(name, what), "model")
        if (K.read_json(os.path.join(export_dir(name, what), "status.json"), {}) or {}).get("phase") != "done":
            raise ValueError("export %s is not finished; start it with export_run" % what)
    else:
        raise ValueError("what must be adapter or one of " + ", ".join(EXPORT_FORMATS))
    if not os.path.isdir(d):
        raise ValueError("nothing saved yet for %s" % what)
    return d


def push(name, repo_id, what="adapter", private=True):
    if not HF_TOKEN:
        raise ValueError("add an HF_TOKEN Kaggle secret with write access and re-run the server cell")
    if not HUB_ID.match(str(repo_id or "")):
        raise ValueError("repo_id must look like username/model-name")
    folder = artifact_dir(name, what)
    PUSH[name] = {"state": "uploading", "what": what, "repo_id": repo_id}

    def go():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id, private=bool(private), exist_ok=True)
            api.upload_folder(folder_path=folder, repo_id=repo_id, commit_message="Unsloth Lab run %s (%s)" % (name, what))
            PUSH[name] = {"state": "done", "what": what, "url": "https://huggingface.co/%s" % repo_id}
        except Exception as e:
            log.error("push failed\n%s", traceback.format_exc())
            PUSH[name] = {"state": "error", "what": what, "error": str(e)[:500]}
    threading.Thread(target=go, daemon=True).start()
    return PUSH[name]


def info():
    return {"gpus": K.gpu_info(), "base_models": BASE_MODELS, "export_formats": EXPORT_FORMATS, "hf_token": bool(HF_TOKEN),
            "dataset": dataset_info(), "train": train_status(with_log=False), "chat": chat_status(),
            "export_running": K.process_alive("lab-export")}


# ====================================================================== app
def build_app():
    app = K.App("Unsloth Fine-tuning Lab", password=os.environ.get("APP_PASSWORD"), log=log, max_body=1 << 30,
                instructions=(
                    "Fine-tunes small LLMs with Unsloth (4-bit LoRA). Typical flow: add_examples (or import_hf_dataset) "
                    "-> dataset_info -> start_training -> training_status until state is 'done' -> chat with "
                    "model='run:<name>' -> export_run (gguf_q4_k_m for Ollama/llama.cpp) -> export_status -> push_to_hub. "
                    "Long operations return immediately; poll their status tools. chat returns loading=true while "
                    "a model loads: call it again after ~30-60 s."))
    app.page("/", os.path.join(HERE, "lab_ui.html"))
    app.static("/static/", HERE)

    @app.route("GET", "/api/info")
    def _info(req):
        return info()

    # ---------------- dataset
    @app.route("GET", "/api/dataset")
    def _dataset(req):
        off, lim, q = req.arg("offset", 0, int), min(200, req.arg("limit", 50, int)), (req.arg("q", "") or "").lower()
        with LOCK:
            rows = list(enumerate(DATA["records"]))
        if q:
            rows = [(i, r) for i, r in rows if q in json.dumps(r, ensure_ascii=False).lower()]
        return {"total": len(rows), "rows": [{"index": i, "preview": D.preview(r), "kind": "text" if "text" in r else
                                              "chat", "chars": D.chars(r)} for i, r in rows[off:off + lim]]}

    @app.route("GET", "/api/dataset/record")
    def _record(req):
        i = req.arg("index", cast=int)
        with LOCK:
            if i is None or not 0 <= i < len(DATA["records"]):
                raise ValueError("no record %s" % i)
            return DATA["records"][i]

    @app.route("POST", "/api/dataset/upload")
    def _upload(req):
        recs, errs = D.parse(req.body().decode("utf-8-sig", "replace"), req.arg("format", "auto"))
        total = add_records(recs, req.arg("mode", "append")) if recs else len(DATA["records"])
        return {"added": len(recs), "total": total, "errors": errs[:30], "error_count": len(errs)}

    @app.route("POST", "/api/dataset/add")
    def _add(req):
        return t_add(**req.json())

    @app.route("POST", "/api/dataset/import_hf")
    def _import(req):
        b = req.json()
        return import_hf(b.get("name"), b.get("split") or "train", b.get("config"), b.get("max_rows", 5000),
                         b.get("mode", "append"))

    @app.route("POST", "/api/dataset/delete")
    def _delete_rows(req):
        idx = sorted({int(i) for i in req.json().get("indices", [])}, reverse=True)
        with LOCK:
            for i in idx:
                if 0 <= i < len(DATA["records"]):
                    del DATA["records"][i]
            save_dataset()
            return {"total": len(DATA["records"])}

    @app.route("POST", "/api/dataset/clear")
    def _clear(req):
        return {"total": add_records([], "replace")}

    @app.route("GET", "/api/dataset/download")
    def _download_ds(req):
        with LOCK:
            save_dataset()
        return K.FileResponse(p("dataset.jsonl"), name="dataset.jsonl", ctype="application/x-ndjson")

    # ---------------- training and runs
    @app.route("POST", "/api/train")
    def _train(req):
        return start_training(req.json())

    @app.route("GET", "/api/train/status")
    def _train_status(req):
        return train_status()

    @app.route("POST", "/api/train/stop")
    def _stop(req):
        return stop_training(bool(req.json().get("force")))

    @app.route("GET", "/api/runs")
    def _runs(req):
        return {"runs": list_runs()}

    @app.route("GET", r"/api/runs/(?P<name>[\w.-]+)")
    def _run(req):
        return run_detail(req.params["name"])

    @app.route("POST", r"/api/runs/(?P<name>[\w.-]+)/delete")
    def _delrun(req):
        return delete_run(req.params["name"])

    @app.route("POST", r"/api/runs/(?P<name>[\w.-]+)/export")
    def _export(req):
        return start_export(req.params["name"], req.json().get("format"))

    @app.route("GET", r"/api/runs/(?P<name>[\w.-]+)/download/(?P<what>[\w]+)")
    def _download(req):
        name, what = req.params["name"], req.params["what"]
        d = artifact_dir(name, what)
        files = [(os.path.join(dp, f), os.path.join("%s-%s" % (name, what), os.path.relpath(os.path.join(dp, f), d)))
                 for dp, dn, fn in os.walk(d) for f in sorted(fn)]
        if what == "adapter":
            rd = run_dir(name)
            files += [(os.path.join(rd, f), "%s-adapter/%s" % (name, f)) for f in ("job.json", "metrics.json",
                                                                                  "samples.json")]
        return K.ZipResponse(files, "%s-%s.zip" % (name, what))

    @app.route("POST", r"/api/runs/(?P<name>[\w.-]+)/push")
    def _push(req):
        b = req.json()
        return push(req.params["name"], b.get("repo_id"), b.get("what") or "adapter", b.get("private", True))

    # ---------------- chat
    @app.route("POST", "/api/chat")
    def _chat(req):
        b = req.json()
        return chat(b.get("model"), b.get("messages") or [], b.get("max_new_tokens", 512), b.get("temperature", 0.7))

    @app.route("POST", "/api/chat/unload")
    def _unload(req):
        stop_infer()
        return {"state": "off"}

    # ---------------- MCP tools
    @app.tool("list_base_models", "Suggested base models (any Hugging Face id also works) and the GPUs.")
    def t_bases():
        return {"base_models": BASE_MODELS, "gpus": K.gpu_info()}

    @app.tool("dataset_info", "Size and shape of the training dataset, plus the status of a Hub import.")
    def t_info():
        return dataset_info()

    @app.tool("add_examples", "Add training examples. Each record: {messages:[{role,content}...]}, "
              "{instruction, input?, output}, {prompt, completion} or {text}. mode=replace starts over.", {
                  "records": {"type": "array", "items": {"type": "object"}},
                  "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"}}, ["records"])
    def t_add(records, mode="append"):
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        recs, errs = [], []
        for i, r in enumerate(records):
            try:
                recs.append(D.normalize(r))
            except ValueError as e:
                errs.append({"index": i, "error": str(e)})
        total = add_records(recs, mode) if recs or mode == "replace" else len(DATA["records"])
        return {"added": len(recs), "total": total, "errors": errs[:30]}

    @app.tool("import_hf_dataset", "Import rows from a Hugging Face dataset (runs in the background; see dataset_info).", {
        "name": {"type": "string", "description": "owner/dataset, e.g. mlabonne/FineTome-100k"},
        "split": {"type": "string", "default": "train"}, "config": {"type": "string"},
        "max_rows": {"type": "integer", "default": 5000},
        "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"}}, ["name"])
    def t_import(name, split="train", config=None, max_rows=5000, mode="append"):
        return import_hf(name, split, config, max_rows, mode)

    @app.tool("start_training", "Start a 4-bit LoRA fine-tune on the current dataset. Returns at once; poll "
              "training_status.", {
                  "name": {"type": "string"},
                  "base_model": {"type": "string", "description": "Hub id, or run:<name> to continue a run"},
                  "epochs": {"type": "number", "default": 2}, "max_steps": {"type": "integer", "default": 0,
                                                                           "description": ">0 overrides epochs"},
                  "learning_rate": {"type": "number", "default": 0.0002}, "lora_r": {"type": "integer", "default": 16},
                  "lora_alpha": {"type": "integer", "default": 16},
                  "max_seq_length": {"type": "integer", "default": 2048},
                  "batch_size": {"type": "integer", "default": 2}, "grad_accum": {"type": "integer", "default": 4},
                  "val_frac": {"type": "number", "default": 0.05}})
    def t_train(**o):
        return start_training(o)

    @app.tool("training_status", "Progress of the current or last run: step, loss, eval loss, ETA and the log tail. "
              "wait_seconds (max 85) waits for it to finish.", {"wait_seconds": {"type": "number", "default": 0}})
    def t_status(wait_seconds=0):
        t = time.time()
        while True:
            st = train_status()
            if st["state"] != "running" or time.time() - t >= min(MAX_WAIT, float(wait_seconds or 0)):
                st.pop("losses", None)
                return st
            time.sleep(3)

    @app.tool("stop_training", "Stop training after the current step and save the adapter trained so far.",
              {"force": {"type": "boolean", "default": False, "description": "kill at once without saving"}})
    def t_stop(force=False):
        return stop_training(force)

    @app.tool("list_runs", "All runs with metrics, exports and Hub pushes.")
    def t_runs():
        return list_runs()

    @app.tool("get_run", "One run: settings, status, metrics and sample answers on held-out examples.",
              {"name": {"type": "string"}}, ["name"])
    def t_run(name):
        return run_detail(name)

    @app.tool("chat", "Chat with a base model (Hub id) or a fine-tuned run (run:<name>). Returns loading=true while "
              "the model loads; call again after 30-60 s.", {
                  "model": {"type": "string"},
                  "messages": {"type": "array", "items": {"type": "object"},
                               "description": "[{role, content}], or pass a plain string as the user message"},
                  "max_new_tokens": {"type": "integer", "default": 512}, "temperature": {"type": "number", "default": 0.7}},
              ["model", "messages"])
    def t_chat(model, messages, max_new_tokens=512, temperature=0.7):
        return chat(model, messages, max_new_tokens, temperature)

    @app.tool("export_run", "Export a run: merged_16bit (full HF model), gguf_q4_k_m / gguf_q8_0 / gguf_f16 "
              "(llama.cpp, Ollama, LM Studio). Runs in the background; poll export_status.", {
                  "name": {"type": "string"}, "format": {"type": "string", "enum": EXPORT_FORMATS}}, ["name", "format"])
    def t_export(name, format):
        return start_export(name, format)

    @app.tool("export_status", "Status of a run's exports, with download paths.", {"name": {"type": "string"}}, ["name"])
    def t_export_status(name):
        run_dir(name)
        ex = export_list(name)
        for fmt, st in ex.items():
            if st.get("phase") == "done":
                st["download"] = "/api/runs/%s/download/%s" % (name, fmt)
        return {"exports": ex, "adapter_download": "/api/runs/%s/download/adapter" % name}

    @app.tool("push_to_hub", "Upload a run's adapter or a finished export to the Hugging Face Hub (needs HF_TOKEN "
              "with write access).", {
                  "name": {"type": "string"}, "repo_id": {"type": "string"},
                  "what": {"type": "string", "enum": ["adapter"] + EXPORT_FORMATS, "default": "adapter"},
                  "private": {"type": "boolean", "default": True}}, ["name", "repo_id"])
    def t_push(name, repo_id, what="adapter", private=True):
        return push(name, repo_id, what, private)

    @app.tool("delete_run", "Delete a run and its exports.", {"name": {"type": "string"}}, ["name"])
    def t_delete(name):
        return delete_run(name)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    os.makedirs(p("runs"), exist_ok=True)
    os.makedirs(EXPORTS, exist_ok=True)
    load_dataset()
    if not os.path.exists(p("dataset.jsonl")):
        save_dataset()
    build_app().serve_forever(ap.parse_args().port)
