"""Generates unsloth-finetuning-lab.ipynb. Run: python src/build_notebook.py"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "..", "..", "..", "shared"))
import nbkit  # noqa: E402

APP_DIR = "/kaggle/working/unsloth_app"
nb = nbkit.Notebook(APP_DIR)

nb.md("""
# Unsloth Fine-tuning Lab on Kaggle: the minimalist LLM factory behind your Cloudflare Tunnel

Fine-tune small open LLMs on a free Kaggle GPU with [Unsloth](https://github.com/unslothai/unsloth) (4-bit QLoRA, about 2x faster and much less memory than plain Hugging Face training), from a web page or from an agent:

1. **Dataset:** upload JSONL / JSON / CSV (chat messages, ShareGPT, Alpaca, prompt/completion or raw text), import from the Hugging Face Hub, or let an agent add examples.
2. **Train:** pick a base model (Llama 3.2, Qwen2.5, Gemma 2, Mistral, or any Hub id), watch loss and eval loss live, stop early any time (the adapter is still saved).
3. **Test:** chat with the base model and your fine-tuned runs side by side; see the model's answers on held-out examples.
4. **Ship:** download the LoRA adapter, export a merged 16-bit model or GGUF (`q4_k_m`, `q8_0`, `f16` for Ollama, llama.cpp, LM Studio), or push to the Hub.

| Path | What |
|---|---|
| `/` | The lab page |
| `/api/*` | JSON API (dataset, training, runs, exports, chat) |
| `/mcp` | MCP server for agents (tools: `list_base_models`, `dataset_info`, `add_examples`, `import_hf_dataset`, `start_training`, `training_status`, `stop_training`, `list_runs`, `get_run`, `chat`, `export_run`, `export_status`, `push_to_hub`, `delete_run`) |

### Before you run
1. **Accelerator:** GPU T4 x2: training on GPU 0, chat and exports on GPU 1. (One GPU works; chat and exports then wait for training.) **Internet:** on.
2. **A Cloudflare Tunnel** with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then tick them for this notebook):
   - `UNSLOTH_TUNNEL_TOKEN`: the tunnel token. Without it everything still runs, only without a public URL.
   - `UNSLOTH_UI_PASSWORD` (recommended): browser login is any username + this password; agents send `Authorization: Bearer <password>`. Without it a random password is printed below.
   - `HF_TOKEN` (optional): for gated base models (Llama) and for pushing to the Hub (needs write access).
4. **Run All.** The install takes about 3–5 minutes. A 3B model on 1,000 short examples trains in roughly 10–20 minutes on a T4.

Runs (datasets, adapters, metrics) are kept in `/kaggle/working/unsloth_lab`; exports are large and go to `/tmp/unsloth_lab/exports`. Both disappear when the session ends: download or push what you want to keep. Logs: `/kaggle/working/logs/`.
""")

nb.code("""
# 1. Settings
PORT = 7860
WORK_DIR = "/kaggle/working/unsloth_lab"
EXPORT_DIR = "/tmp/unsloth_lab/exports"    # merged models are 3-16 GB: outside /kaggle/working's 20 GB
LOG_DIR = "/kaggle/working/logs"
APP_DIR = %r
print("Settings saved.")
""" % APP_DIR)

nb.md("### 2. App files\nThese cells only write files into `APP_DIR`.")
nb.kit_files()
for name, title in [("lab_data.py", "Dataset formats and statistics"),
                    ("train_unsloth.py", "Training process (Unsloth QLoRA + TRL SFTTrainer)"),
                    ("infer_worker.py", "Chat process"),
                    ("export_unsloth.py", "Export process (merged 16-bit, GGUF)"),
                    ("lab_server.py", "Lab server on port 7860: API and MCP"),
                    ("lab_ui.html", "Lab page")]:
    nb.writefile(os.path.join(SRC, name), title)

nb.code("# 3. Load the helpers\n" + nbkit.setup_cell(APP_DIR, "Notebook"))

nb.code(r'''
# 4. Install Unsloth (quiet; 3-5 minutes). Unsloth picks the PyTorch / xformers / bitsandbytes versions it
#    needs, so this notebook never imports torch itself: training, chat and exports run in their own processes.
import subprocess
''' + nbkit.PIP_KEEP_CORE + r'''

os.makedirs(LOG_DIR, exist_ok=True)
print("Unsloth...")
pip_install("unsloth", keep_core=False)
pip_install("huggingface_hub", "datasets", keep_core=False)
K.ensure_cloudflared()
check = subprocess.run([sys.executable, "-c", "import unsloth, trl, transformers, torch; "
                        "print('unsloth', unsloth.__version__, '| trl', trl.__version__, '| transformers', "
                        "transformers.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())"],
                       capture_output=True, text=True, timeout=600)
print(check.stdout.strip().splitlines()[-1] if check.returncode == 0 else
      "Unsloth could not load (it needs a GPU accelerator):\n" + check.stderr[-1500:])
gpus = K.gpu_info()
print("GPUs:", ", ".join("%d: %s" % (g["index"], g["name"]) for g in gpus) or "none - pick 'GPU T4 x2' in the settings")
''')

nb.code(r'''
# 5. Start the lab on port 7860 (background process). Log: /kaggle/working/logs/unsloth_lab.log
PASSWORD = K.ui_password("UNSLOTH_UI_PASSWORD")
HF_TOKEN = K.kaggle_secret("HF_TOKEN")
env = dict(os.environ, APP_PASSWORD=PASSWORD, WORK_DIR=WORK_DIR, EXPORT_DIR=EXPORT_DIR, LOG_DIR=LOG_DIR,
           APP_LOG=LOG_DIR + "/unsloth_lab.log", PYTHONUNBUFFERED="1")
env.pop("HF_TOKEN", None)
if HF_TOKEN:
    env["HF_TOKEN"] = HF_TOKEN
else:
    print("No HF_TOKEN secret: gated models (Llama) and 'Push to Hub' are unavailable; everything else works.")
K.spawn("unsloth-lab", [sys.executable, os.path.join(APP_DIR, "lab_server.py"), "--port", str(PORT)],
        LOG_DIR + "/unsloth_lab.log", env=env, cwd=APP_DIR)
if not K.wait_http("http://127.0.0.1:%d/health" % PORT, timeout=60, name="unsloth-lab"):
    print(K.tail(LOG_DIR + "/unsloth_lab.log", 40))
    raise RuntimeError("The lab did not start; log above")
for host in ("127.0.0.1", "[::1]"):
    print(host, "->", "up" if K.wait_http("http://%s:%d/health" % (host, PORT), timeout=3) else "not reachable")
print("Training, chat and export processes keep running if you re-run this cell.")
''')

nb.code(r'''
# 6. Publish through your Cloudflare Tunnel (token from the UNSLOTH_TUNNEL_TOKEN secret, never printed)
tunnel = K.start_tunnel("UNSLOTH_TUNNEL_TOKEN", PORT, LOG_DIR + "/cloudflared.log")
''')

nb.code(r'''
# 7. Keep-alive monitor. Leave it running; interrupting it only stops the status lines.
def train_line():
    t = K.api_get(PORT, "/api/info", PASSWORD)["train"]
    if t.get("state") == "running":
        return "training %s step %s/%s loss %s" % (t.get("run"), t.get("step", 0), t.get("total_steps", "?"), t.get("loss", "-"))
    return "training=%s" % t.get("state", "idle")

K.keep_alive(PORT, ["unsloth-lab", "cloudflared"], extra=train_line)
''')

nb.save(os.path.join(SRC, "..", "unsloth-finetuning-lab.ipynb"))
