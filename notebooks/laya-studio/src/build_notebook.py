import json, os

SRC = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": s, "outputs": [], "execution_count": None})


def writefile(name, title):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        body = f.read()
    assert "'''" not in body and not body.endswith("\\")
    code("# %s\n# Writes %s into APP_DIR. Safe to re-run.\nimport os\nAPP_DIR = \"/kaggle/working/laya_app\"\n"
         "os.makedirs(APP_DIR, exist_ok=True)\n"
         "with open(os.path.join(APP_DIR, %r), \"w\", encoding=\"utf-8\") as f:\n"
         "    f.write(r'''%s''')\nprint(\"wrote\", os.path.join(APP_DIR, %r))" % (title, name, name, body, name))


md("""# Laya Studio: label, fine-tune and test Laya on Kaggle, behind a Cloudflare Tunnel

A web UI on port **7860** with five tabs:

| Tab | What it does |
|---|---|
| **Dataset** | Upload JSONL/CSV, import files attached under `/kaggle/input`, browse/search/delete records, see label balance |
| **Label** | Paste a bill/receipt/transaction, answer the template questions (optionally pre-filled by a model), save |
| **Train** | Fine-tune `laya`, `laya-typed-decisions`, `laya-multilingual` or one of your earlier runs; live progress and before/after accuracy |
| **Test** | Run any base or fine-tuned model on a document and see each decision with its probabilities |
| **Models** | Download a run as .zip, push it to Hugging Face, delete runs |

**What Laya does:** typed decisions over text or JSON: *choice* (which category), *noul* (yes/no) and *score* (ordered level), in one forward pass.
It does **not** read images or extract values such as totals or dates. For scanned bills, run OCR first and feed the text (plus any fields you already have) as the document.

### Before you run
1. **Accelerator:** GPU T4 x2 (one GPU also works). **Internet:** on.
2. **A new Cloudflare Tunnel** for this notebook, e.g. `laya.example.com` → `http://localhost:7860`.
   Do not reuse the Ollama notebook's tunnel: two notebooks on one tunnel would randomly split traffic between them.
3. **Kaggle secrets** (Add-ons → Secrets):
   - `LAYA_TUNNEL_TOKEN`: the new tunnel's token (required)
   - `LAYA_UI_PASSWORD`: password for the UI (recommended; if missing, a random one is printed below)
   - `HF_TOKEN`: Hugging Face token with write access (optional, for "Push to HF")
4. Run All. The last cell keeps the session alive; the UI keeps working while it runs.

`/kaggle/working` is wiped when the session ends. Keep your dataset by downloading it (Dataset tab), or attach it as a Kaggle dataset and use **Import**; keep models via Download or Push to HF.""")

code("""# 1. Install Laya and cloudflared (torch/transformers are already on Kaggle)
!pip install -q laya || pip install -q git+https://github.com/NandhaKishorM/laya
!curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /kaggle/working/cloudflared && chmod +x /kaggle/working/cloudflared
!/kaggle/working/cloudflared --version
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: enable a GPU accelerator to train"
import laya, torch, transformers
print("laya", getattr(laya, "__version__", "?"), "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| GPUs:", torch.cuda.device_count())""")

md("### 2. App files\nThese cells only write files; nothing runs yet.")
writefile("laya_data.py", "Dataset helpers and the bills / receipts / fintech question templates")
writefile("train_laya.py", "Fine-tuning script launched by the UI (based on the official Laya Kaggle notebook)")
writefile("laya_studio_server.py", "Web server for the UI")
writefile("laya_studio_ui.html", "UI")

code("""# 3. Start the UI on port 7860 (background thread)
import os, sys, secrets, threading, importlib, requests

APP_DIR = "/kaggle/working/laya_app"
_need = ["laya_data.py", "train_laya.py", "laya_studio_server.py", "laya_studio_ui.html"]
_missing = [f for f in _need if not os.path.exists(os.path.join(APP_DIR, f))]
if _missing:
    raise RuntimeError("App files missing from %s: %s\\nRun the four 'Writes ... into APP_DIR' cells above "
                       "first (or use Run All)." % (APP_DIR, ", ".join(_missing)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

def kaggle_secret(name):
    \"\"\"Read a Kaggle secret; returns None if it doesn't exist or isn't attached to this notebook.\"\"\"
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None

PORT = 7860
WORK_DIR = "/kaggle/working/laya_studio"
PASSWORD = kaggle_secret("LAYA_UI_PASSWORD")
if not PASSWORD:
    PASSWORD = secrets.token_urlsafe(9)
    print("No LAYA_UI_PASSWORD secret - using this one for this session:", PASSWORD)
HF_TOKEN = kaggle_secret("HF_TOKEN")   # optional: only needed for "Push to HF"

import laya_data, laya_studio_server
if "S" in globals():
    S.stop()                                  # release port 7860 before restarting
    if not S.training_running():              # reload edits, but never orphan a running training job
        importlib.reload(laya_data)
        S = importlib.reload(laya_studio_server)
else:
    S = laya_studio_server
S.start(WORK_DIR, port=PORT, password=PASSWORD, hf_token=HF_TOKEN)

for host in ("127.0.0.1", "[::1]"):
    try:
        print(host, "->", requests.get(f"http://{host}:{PORT}/health", auth=("user", PASSWORD), timeout=3).json())
    except Exception as e:
        print(host, "-> not reachable:", type(e).__name__)
print("UI login: any username, password as above. Hugging Face push:", "enabled" if HF_TOKEN else "off (no HF_TOKEN)")

# Warm the cache with the default base model so the first test/training run starts faster
def _prefetch():
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("convaiinnovations/laya",
                          allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
        print("Base model convaiinnovations/laya cached.")
    except Exception as e:
        print("Prefetch skipped:", e)
threading.Thread(target=_prefetch, daemon=True).start()""")

code("""# 4. Start this notebook's Cloudflare Tunnel in the background
import subprocess, time

TOKEN = kaggle_secret("LAYA_TUNNEL_TOKEN")
assert TOKEN, "Add the new tunnel's token as a Kaggle secret named LAYA_TUNNEL_TOKEN"

if "tunnel_proc" in globals() and tunnel_proc.poll() is None:
    tunnel_proc.terminate()

tunnel_log = open("/kaggle/working/cloudflared.log", "a")
tunnel_proc = subprocess.Popen(["/kaggle/working/cloudflared", "tunnel", "--no-autoupdate", "run"],
                               stdout=tunnel_log, stderr=subprocess.STDOUT,
                               env={**os.environ, "TUNNEL_TOKEN": TOKEN})   # token never printed

for _ in range(30):
    time.sleep(1)
    if "Registered tunnel connection" in open("/kaggle/working/cloudflared.log").read():
        print("Tunnel connected. Open the public hostname you routed to http://localhost:7860")
        break
else:
    print("Tunnel not registered yet - last log lines:")
    print("".join(open("/kaggle/working/cloudflared.log").readlines()[-15:]))""")

code("""# 5. Keep-alive monitor. Leave it running; interrupt to stop watching (the UI and tunnel keep running).
import datetime
while True:
    ui = "up"
    try:
        requests.get(f"http://127.0.0.1:{PORT}/health", auth=("user", PASSWORD), timeout=3)
    except Exception:
        ui = "DOWN"
    tr = S.train_status()
    line = f"{datetime.datetime.now():%H:%M:%S} ui={ui} tunnel={'up' if tunnel_proc.poll() is None else 'DOWN'} training={tr.get('state')}"
    if tr.get("state") == "running":
        line += f" {tr.get('phase')} step {tr.get('step', 0)}/{tr.get('total_steps', '?')} loss {tr.get('loss', '-')}"
    print(line, flush=True)
    time.sleep(60)""")

nb = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": True},
    },
    "nbformat": 4, "nbformat_minor": 4, "cells": cells,
}
out = os.path.join(os.path.dirname(SRC), "laya-studio.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
