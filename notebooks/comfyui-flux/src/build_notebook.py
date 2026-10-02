"""Generates comfyui-flux.ipynb. Run: python src/build_notebook.py"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "..", "..", "..", "shared"))
import nbkit  # noqa: E402

APP_DIR = "/kaggle/working/comfy_app"
nb = nbkit.Notebook(APP_DIR)

nb.md("""
# ComfyUI + FLUX on Kaggle: generative art engine behind your Cloudflare Tunnel

Runs [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with **FLUX.1** on a free Kaggle GPU and publishes it on your own hostname, behind a password:

| Path | What |
|---|---|
| `/` | The full ComfyUI node editor (load workflows, add LoRAs, inpaint, upscale) |
| `/flux` | A one-box prompt → image page |
| `/flux/api/*` | Small JSON API: `generate`, `jobs`, `workflow`, `models`, `info` |
| `/mcp` | MCP server for coding agents (tools: `generate_image`, `get_job`, `run_workflow`, `list_jobs`, `cancel_job`, `list_models`, `server_status`) |

### Before you run
1. **Accelerator:** GPU T4 x2 (ComfyUI uses one GPU) or P100. **Internet:** on.
2. **A Cloudflare Tunnel** with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then tick them for this notebook):
   - `COMFYUI_TUNNEL_TOKEN`: the tunnel token. Without it everything still runs, only without a public URL.
   - `COMFYUI_UI_PASSWORD` (recommended): login password. Browser: any username + this password. Agents: `Authorization: Bearer <password>`. Without it a random password is printed below.
   - `HF_TOKEN` (only for `FLUX_VARIANT = "dev"`): a Hugging Face token whose account accepted the FLUX.1-dev licence.
4. **Run All.** First run: about 10 minutes (install + 17 GB model download). The first image then takes 1–3 minutes while the model loads; after that FLUX.1-schnell takes roughly 30–60 s per 1024² image on a T4.

### Licences
- FLUX.1-**schnell** (default): Apache 2.0, commercial use allowed.
- FLUX.1-**dev**: FLUX.1 [dev] Non-Commercial License. Set `FLUX_VARIANT = "dev"` only if that suits you.
- ComfyUI: GPL-3.0.

Images are saved to `/kaggle/working/comfyui-output` (kept in the notebook's output when you save a version). Logs: `/kaggle/working/logs/`.
""")

nb.code("""
# 1. Settings (edit if you like)
FLUX_VARIANT = "schnell"      # "schnell" (Apache 2.0, 4 steps) or "dev" (non-commercial, ~20 steps, needs HF_TOKEN)
COMFYUI_REF = ""              # "" = newest ComfyUI; or a tag/commit to pin, e.g. "v0.3.60"
EXTRA_COMFY_ARGS = []         # e.g. ["--lowvram"] if you load extra models and run out of GPU memory
PORT = 7860
COMFY_PORT = 8188
COMFY_DIR = "/tmp/ComfyUI"    # code + models live outside /kaggle/working (its 20 GB are for your images)
OUTPUT_DIR = "/kaggle/working/comfyui-output"
LOG_DIR = "/kaggle/working/logs"
APP_DIR = %r
CHECKPOINTS = {
    "schnell": ("Comfy-Org/flux1-schnell", "flux1-schnell-fp8.safetensors"),
    "dev": ("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
}
assert FLUX_VARIANT in CHECKPOINTS, "FLUX_VARIANT must be 'schnell' or 'dev'"
print("Settings saved.")
""" % APP_DIR)

nb.md("### 2. App files\nThese cells only write files into `APP_DIR`.")
nb.kit_files()
nb.writefile(os.path.join(SRC, "comfy_gateway.py"), "Gateway on port 7860: password, ComfyUI proxy, /flux page, API and MCP")
nb.writefile(os.path.join(SRC, "flux_ui.html"), "The /flux page")

nb.code("# 3. Load the helpers\n" + nbkit.setup_cell(APP_DIR, "Notebook"))

nb.code(r'''
# 4. Install ComfyUI and cloudflared (quiet; about 3 minutes; re-running skips finished steps)
import re, subprocess, shutil
''' + nbkit.PIP_KEEP_CORE + r'''

def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
        raise RuntimeError("failed: " + cmd[:120])
    return p.stdout

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir(COMFY_DIR + "/.git"):
    print("ComfyUI source...")
    sh("git clone -q https://github.com/comfyanonymous/ComfyUI %s" % COMFY_DIR)
if COMFYUI_REF:
    sh("cd %s && git fetch -q origin %s && git checkout -q %s" % (COMFY_DIR, COMFYUI_REF, COMFYUI_REF))
print("ComfyUI at", sh("cd %s && git describe --tags --always" % COMFY_DIR).strip())

print("Python packages (keeping Kaggle's own PyTorch)...")
reqs = [l.strip() for l in open(COMFY_DIR + "/requirements.txt") if l.strip() and not l.lstrip().startswith("#")]
keep = [r for r in reqs if re.split(r"[<>=~!\[; ]", r, 1)[0].lower() not in ("torch", "torchvision", "torchaudio")]
open("/tmp/comfy_requirements.txt", "w").write("\n".join(keep) + "\n")
pip_install("-r", "/tmp/comfy_requirements.txt")
pip_install("huggingface_hub")
K.ensure_cloudflared()

import torch
print("Install complete. torch", torch.__version__, "| GPUs:",
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "none - pick a GPU accelerator")
''')

nb.code(r'''
# 5. Download the FLUX checkpoint (~17 GB, a few minutes; skipped if already there)
from huggingface_hub import hf_hub_download

repo, CKPT = CHECKPOINTS[FLUX_VARIANT]
ckpt_dir = os.path.join(COMFY_DIR, "models", "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
if os.path.exists(os.path.join(ckpt_dir, CKPT)):
    print("Already downloaded:", CKPT)
else:
    free_gb = shutil.disk_usage(ckpt_dir).free / 2**30
    if free_gb < 20:
        print("Warning: only %.0f GB free under %s; the download needs about 18 GB." % (free_gb, ckpt_dir))
    print("Downloading %s from %s ..." % (CKPT, repo))
    try:
        hf_hub_download(repo, CKPT, local_dir=ckpt_dir, token=K.kaggle_secret("HF_TOKEN"))
    except Exception as e:
        hint = (" FLUX.1-dev is gated: accept its licence on huggingface.co with the account of your HF_TOKEN "
                "secret, and attach that secret." if FLUX_VARIANT == "dev" else "")
        raise RuntimeError("Download failed: %s.%s" % (e, hint)) from None
print("Checkpoint ready:", CKPT)
''')

nb.code(r'''
# 6. Start ComfyUI in the background on 127.0.0.1:8188 (GPU 0). Log: /kaggle/working/logs/comfyui.log
comfy_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
comfy_cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
             "--output-directory", OUTPUT_DIR, "--disable-auto-launch", "--preview-method", "auto"] + EXTRA_COMFY_ARGS
K.spawn("comfyui", comfy_cmd, LOG_DIR + "/comfyui.log", env=comfy_env, cwd=COMFY_DIR)
print("Starting ComfyUI...")
if not K.wait_http("http://127.0.0.1:%d/system_stats" % COMFY_PORT, timeout=300, name="comfyui"):
    print(K.tail(LOG_DIR + "/comfyui.log", 40))
    raise RuntimeError("ComfyUI did not start; log above")
print("ComfyUI is up.")
''')

nb.code(r'''
# 7. Start the gateway on port 7860 (password, /flux page, API, MCP). Log: /kaggle/working/logs/gateway.log
CKPT = CHECKPOINTS[FLUX_VARIANT][1]
PASSWORD = K.ui_password("COMFYUI_UI_PASSWORD")
gw_env = dict(os.environ, APP_PASSWORD=PASSWORD, COMFY_URL="http://127.0.0.1:%d" % COMFY_PORT,
              FLUX_VARIANT=FLUX_VARIANT, FLUX_CKPT=CKPT, APP_LOG=LOG_DIR + "/gateway.log")
K.spawn("gateway", [sys.executable, os.path.join(APP_DIR, "comfy_gateway.py"), "--port", str(PORT)],
        LOG_DIR + "/gateway.log", env=gw_env, cwd=APP_DIR)
if not K.wait_http("http://127.0.0.1:%d/health" % PORT, timeout=30, name="gateway"):
    print(K.tail(LOG_DIR + "/gateway.log", 40))
    raise RuntimeError("The gateway did not start; log above")
for host in ("127.0.0.1", "[::1]"):
    print(host, "->", "up" if K.wait_http("http://%s:%d/health" % (host, PORT), timeout=3) else "not reachable")
print("Paths: / = ComfyUI editor, /flux = quick generator, /mcp = MCP server for agents.")
''')

nb.code(r'''
# 8. Publish through your Cloudflare Tunnel (token from the COMFYUI_TUNNEL_TOKEN secret, never printed)
tunnel = K.start_tunnel("COMFYUI_TUNNEL_TOKEN", PORT, LOG_DIR + "/cloudflared.log")
''')

nb.code(r'''
# 9. Keep-alive monitor. Leave it running; interrupting it only stops the status lines.
def queue_line():
    q = K.api_get(PORT, "/flux/api/info", PASSWORD).get("queue") or {}
    return "queue=%s running/%s waiting" % (q.get("running", "?"), q.get("pending", "?"))

K.keep_alive(PORT, ["comfyui", "gateway", "cloudflared"], extra=queue_line)
''')

nb.save(os.path.join(SRC, "..", "comfyui-flux.ipynb"))
