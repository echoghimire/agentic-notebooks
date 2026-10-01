"""Generates openshorts-studio.ipynb. Run: python src/build_notebook.py"""
import json
import os

SRC = os.path.dirname(os.path.abspath(__file__))
OPENSHORTS_COMMIT = "faff66de0b91277510587a1c778be433ff3d26a4"   # tested layout; change to update OpenShorts
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": s, "outputs": [], "execution_count": None})


def writefile(name, title):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        body = f.read()
    assert "'''" not in body and not body.endswith("\\")
    code("# %s\n# Writes %s into APP_DIR. Safe to re-run.\nimport os\nAPP_DIR = \"/kaggle/working/studio_app\"\n"
         "os.makedirs(APP_DIR, exist_ok=True)\n"
         "with open(os.path.join(APP_DIR, %r), \"w\", encoding=\"utf-8\") as f:\n"
         "    f.write(r'''%s''')\nprint(\"wrote\", os.path.join(APP_DIR, %r))" % (title, name, name, body, name))


md("""# OpenShorts Studio on Kaggle: long video → vertical shorts with Stable Diffusion b-roll

Runs [OpenShorts](https://github.com/mutonby/openshorts) (MIT) on a free Kaggle GPU session, fully local and free:

| Step | Runs on |
|---|---|
| Transcription (faster-whisper `large-v3-turbo`), scene detection, face-tracked 9:16 reframing, subtitles | GPU 0 |
| Picking the best moments (Ollama, `qwen2.5:7b`) | GPU 1 |
| **B-roll** (new): moments + prompts from Ollama, images from Stable Diffusion (SDXL 1.0), slow zoom, cut into the clip | GPU 1 |

Everything is served on port **7860** behind a password and published through your Cloudflare Tunnel:
the OpenShorts dashboard at `/`, the B-roll page at `/broll`, and OpenShorts' MCP server at `/mcp` for agents.

### Before you run
1. **Accelerator:** GPU T4 x2. **Internet:** on. Run All takes roughly 10–15 minutes the first time (installs + model downloads).
2. **A Cloudflare Tunnel** for this notebook with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then attach them to this notebook):
   - `OPENSHORTS_TUNNEL_TOKEN` (required)
   - `OPENSHORTS_UI_PASSWORD` (recommended; otherwise a random one is printed below). Browser login: any username + this password. Agents: `Authorization: Bearer <password>` or `X-Access-Token: <password>`.
   - `OPENSHORTS_YT_COOKIES` (optional): your YouTube cookies in Netscape format, if YouTube blocks downloads from Kaggle.

### Good to know
- Cloudflare's free plan caps one upload at **100 MB**. For longer videos, attach the file as a Kaggle dataset and give OpenShorts its path, or use a direct download link.
- Not included (they need paid services): AI actors, lip-sync, ElevenLabs voices, auto-posting. The Remotion renderer is not started.
- `/kaggle/working` is wiped when the session ends; download the clips you want to keep.
- B-roll uses SDXL 1.0 base (CreativeML OpenRAIL++, commercial use allowed). SDXL-Turbo is much faster but its licence is non-commercial; switch in the settings cell if that suits you.""")

code("""# 1. Settings (edit if you like)
LLM_MODEL = "qwen2.5:7b"              # Ollama model that picks moments and b-roll prompts
WHISPER_MODEL = "large-v3-turbo"      # OpenShorts transcription
SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"   # OpenRAIL++ (commercial use OK), ~20 s per image on a T4
SD_STEPS = 20
# Faster but NON-COMMERCIAL licence: SD_MODEL = "stabilityai/sdxl-turbo"; SD_STEPS = 4
PORT = 7860
OPENSHORTS_COMMIT = %r
REPO_DIR = "/kaggle/working/openshorts"
APP_DIR = "/kaggle/working/studio_app"
print("Settings saved.")""" % OPENSHORTS_COMMIT)

code(r'''# 2. Install everything (about 8-12 minutes). Re-running skips what is already done.
import os, subprocess, sys, glob, shutil, time

def sh(cmd, check=True, quiet=True):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=quiet, text=True)
    if check and p.returncode != 0:
        print((p.stdout or "")[-3000:], (p.stderr or "")[-3000:])
        raise RuntimeError("failed: " + cmd[:120])
    return p

print("System packages..."); sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg fontconfig fonts-liberation fonts-noto-color-emoji zstd > /dev/null")

if not os.path.exists("/kaggle/working/cloudflared"):
    print("cloudflared..."); sh("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /kaggle/working/cloudflared && chmod +x /kaggle/working/cloudflared")

if not shutil.which("ollama"):
    print("Ollama..."); sh("curl -fsSL https://ollama.com/install.sh | sh")

NODE_DIR = "/opt/node20"
if not os.path.exists(NODE_DIR + "/bin/node"):
    print("Node.js 20..."); sh("mkdir -p %s && curl -sL https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz | tar -xJ -C %s --strip-components=1" % (NODE_DIR, NODE_DIR))
os.environ["PATH"] = NODE_DIR + "/bin:" + os.environ["PATH"]

if not os.path.isdir(REPO_DIR + "/.git"):
    print("OpenShorts source..."); sh("git clone -q https://github.com/mutonby/openshorts %s" % REPO_DIR)
sh("cd %s && git fetch -q origin %s 2>/dev/null; git checkout -q %s" % (REPO_DIR, OPENSHORTS_COMMIT, OPENSHORTS_COMMIT), check=False)
print("OpenShorts at", sh("cd %s && git rev-parse --short HEAD" % REPO_DIR).stdout.strip())

print("Python packages (keeping Kaggle's own PyTorch)...")
reqs = [l.strip() for l in open(REPO_DIR + "/requirements.txt") if l.strip() and not l.startswith("#")]
keep = [r for r in reqs if not r.split("==")[0].strip().lower() in ("torch", "torchvision")]
open("/tmp/req_kaggle.txt", "w").write("\n".join(keep) + "\n")
if sh("pip install -q -r /tmp/req_kaggle.txt", check=False).returncode != 0:
    print("  Pinned versions did not all install on this Python; retrying with relaxed pins.")
    open("/tmp/req_relaxed.txt", "w").write("\n".join(r.split("==")[0] for r in keep) + "\n")
    sh("pip install -q -r /tmp/req_relaxed.txt")
sh('pip install -q --upgrade --pre "yt-dlp[default]"')
sh('pip install -q diffusers accelerate "nvidia-cublas-cu12<13" "nvidia-cudnn-cu12>=9,<10"')

# CUDA libraries for faster-whisper (CTranslate2)
libdirs = sorted({os.path.dirname(p) for pat in ("nvidia/cublas/lib/*.so*", "nvidia/cudnn/lib/*.so*", "nvidia/cuda_runtime/lib/*.so*")
                  for sp in sys.path if sp.endswith("site-packages") for p in glob.glob(os.path.join(sp, pat))})
os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")

# Subtitle fonts used by OpenShorts
os.makedirs(os.path.expanduser("~/.fonts"), exist_ok=True)
for f in glob.glob(REPO_DIR + "/fonts/*.ttf"):
    shutil.copy(f, os.path.expanduser("~/.fonts/"))
os.makedirs(os.path.expanduser("~/.config/fontconfig/conf.d"), exist_ok=True)
if os.path.exists(REPO_DIR + "/fonts/openshorts-fontmap.conf"):
    shutil.copy(REPO_DIR + "/fonts/openshorts-fontmap.conf", os.path.expanduser("~/.config/fontconfig/conf.d/60-openshorts.conf"))
sh("fc-cache -f", check=False)

# Dashboard: the self-hosted UI blocks every job until an Upload-Post (auto-posting) key is set,
# even with a local LLM. Auto-posting is optional here, so only the LLM is required.
app_jsx = REPO_DIR + "/dashboard/src/App.jsx"
src = open(app_jsx).read()
old = "const keysMissing = !billingEnabled && (!geminiOk || !uploadPostKey);"
if old in src:
    open(app_jsx, "w").write(src.replace(old, "const keysMissing = !billingEnabled && !geminiOk;"))
    print("Patched dashboard: Upload-Post key is optional.")
elif "!billingEnabled && !geminiOk;" not in src:
    print("WARNING: dashboard key check changed upstream; the UI may ask for an Upload-Post key. Pin OPENSHORTS_COMMIT back if so.")

dist = REPO_DIR + "/dashboard/dist"
if not os.path.exists(dist + "/index.html"):
    print("Building the dashboard (npm)...")
    sh("cd %s/dashboard && npm ci --no-audit --no-fund --loglevel=error && npm run build" % REPO_DIR)
idx = open(dist + "/index.html").read()
if "id=\"broll-link\"" not in idx:
    link = ('<a id="broll-link" href="/broll/" style="position:fixed;right:16px;bottom:16px;z-index:9999;'
            'background:#e4572e;color:#fff;font:600 14px system-ui;padding:10px 14px;border-radius:999px;'
            'text-decoration:none;box-shadow:0 6px 18px rgba(0,0,0,.25)">+ B-roll</a>')
    open(dist + "/index.html", "w").write(idx.replace("</body>", link + "</body>"))

import torch
print("Install complete. GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])''')

md("### 3. App files\nThese cells only write files.")
writefile("gateway.py", "Gateway on port 7860: password, OpenShorts dashboard + backend passthrough, /broll")
writefile("broll.py", "B-roll engine: Ollama picks moments, Stable Diffusion draws them, FFmpeg cuts them in")
writefile("broll_ui.html", "B-roll page")

code(r'''# 4. Start Ollama (GPU 1, or GPU 0 if there is only one) and pull the model
import os, subprocess, time, requests, torch

N_GPU = torch.cuda.device_count()
LLM_GPU = "1" if N_GPU > 1 else "0"
if N_GPU < 2:
    print("Only %d GPU: everything shares it. Pick 'GPU T4 x2' for a smoother run." % N_GPU)

if "ollama_proc" in globals() and ollama_proc.poll() is None:
    ollama_proc.terminate(); ollama_proc.wait(10)
ollama_env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", CUDA_VISIBLE_DEVICES=LLM_GPU,
                  OLLAMA_CONTEXT_LENGTH="16384", OLLAMA_KEEP_ALIVE="30m")
ollama_proc = subprocess.Popen(["ollama", "serve"], env=ollama_env, stdout=open("/kaggle/working/ollama.log", "a"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2); break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Ollama did not start - see /kaggle/working/ollama.log")
subprocess.run(["ollama", "pull", LLM_MODEL], env=ollama_env, check=True)
print("Ollama ready with", LLM_MODEL, "on GPU", LLM_GPU)''')

code(r'''# 5. Start the OpenShorts backend on 127.0.0.1:8000 (GPU 0)
import os, subprocess, time, requests

def kaggle_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None

if "backend_proc" in globals() and backend_proc.poll() is None:
    backend_proc.terminate(); backend_proc.wait(15)
backend_env = dict(os.environ,
    CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
    LLM_BASE_URL="http://127.0.0.1:11434/v1", LLM_MODEL=LLM_MODEL, LLM_TIMEOUT="900",
    WHISPER_DEVICE="cuda", WHISPER_COMPUTE="float16", WHISPER_MODEL=WHISPER_MODEL,
    FFMPEG_ENCODER="auto", MAX_CONCURRENT_JOBS="1")
cookies = kaggle_secret("OPENSHORTS_YT_COOKIES")
if cookies:
    backend_env["YOUTUBE_COOKIES"] = cookies
backend_log = open("/kaggle/working/openshorts-backend.log", "a")
backend_proc = subprocess.Popen(["python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000",
                                 "--proxy-headers", "--forwarded-allow-ips", "*"],
                                cwd=REPO_DIR, env=backend_env, stdout=backend_log, stderr=subprocess.STDOUT)
for _ in range(180):
    if backend_proc.poll() is not None:
        break
    try:
        if requests.get("http://127.0.0.1:8000/health/ready", timeout=2).ok:
            print("OpenShorts backend ready:", requests.get("http://127.0.0.1:8000/api/config", timeout=5).json().get("localLlm"))
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("Backend is slow to start; check below.")
if backend_proc.poll() is not None:
    print("".join(open("/kaggle/working/openshorts-backend.log").readlines()[-40:]))
    raise RuntimeError("OpenShorts backend exited - log above")''')

code(r'''# 6. Start the gateway + B-roll page on port 7860
import sys, secrets, importlib, requests, torch
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
need = ["gateway.py", "broll.py", "broll_ui.html"]
missing = [f for f in need if not os.path.exists(os.path.join(APP_DIR, f))]
if missing:
    raise RuntimeError("Run the app-file cells above first (missing: %s)" % ", ".join(missing))

PASSWORD = kaggle_secret("OPENSHORTS_UI_PASSWORD")
if not PASSWORD:
    PASSWORD = secrets.token_urlsafe(9)
    print("No OPENSHORTS_UI_PASSWORD secret - this session's password:", PASSWORD)

import gateway, broll
if "gw" in globals():
    gateway.stop()
gateway = importlib.reload(gateway); broll = importlib.reload(broll)
sd_gpu = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
broll_app = broll.build_app(output_dir=os.path.join(REPO_DIR, "output"), llm_model=LLM_MODEL, sd_model=SD_MODEL, sd_steps=SD_STEPS,
                            sd_device=sd_gpu, whisper_index=int(sd_gpu[-1]), unload_after=torch.cuda.device_count() < 2)
gw = gateway.start(port=PORT, password=PASSWORD, dist=os.path.join(REPO_DIR, "dashboard/dist"), broll_app=broll_app)
time.sleep(2)
for host in ("127.0.0.1", "[::1]"):
    try:
        print(host, "->", requests.get("http://%s:%d/healthz" % (host, PORT), timeout=3).json())
    except Exception as e:
        print(host, "-> not reachable:", type(e).__name__)
print("Login: any username + the password. Agents/MCP: header 'Authorization: Bearer <password>'.")''')

code(r'''# 7. Start this notebook's Cloudflare Tunnel
TOKEN = kaggle_secret("OPENSHORTS_TUNNEL_TOKEN")
assert TOKEN, "Add the tunnel token as a Kaggle secret named OPENSHORTS_TUNNEL_TOKEN and attach it"
if "tunnel_proc" in globals() and tunnel_proc.poll() is None:
    tunnel_proc.terminate()
tunnel_proc = subprocess.Popen(["/kaggle/working/cloudflared", "tunnel", "--no-autoupdate", "run"],
                               stdout=open("/kaggle/working/cloudflared.log", "a"), stderr=subprocess.STDOUT,
                               env={**os.environ, "TUNNEL_TOKEN": TOKEN})
for _ in range(30):
    time.sleep(1)
    if "Registered tunnel connection" in open("/kaggle/working/cloudflared.log").read():
        print("Tunnel connected. Open your hostname: dashboard at /, b-roll at /broll, MCP at /mcp")
        break
else:
    print("".join(open("/kaggle/working/cloudflared.log").readlines()[-15:]))''')

code(r'''# 8. Keep-alive monitor. Leave it running; interrupt to stop watching (everything keeps running).
import datetime
while True:
    def up(url):
        try:
            return "up" if requests.get(url, timeout=3).ok else "DOWN"
        except Exception:
            return "DOWN"
    gpu = ", ".join("%d: %.1f/%.0f GB" % (i, (torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) / 1e9,
                                         torch.cuda.mem_get_info(i)[1] / 1e9) for i in range(torch.cuda.device_count()))
    running = [j["step"] for j in broll.JOBS.values() if j["state"] in ("queued", "running")]
    print("%s gateway=%s backend=%s ollama=%s tunnel=%s | GPU %s%s" % (
        datetime.datetime.now().strftime("%H:%M:%S"), up("http://127.0.0.1:%d/healthz" % PORT),
        up("http://127.0.0.1:8000/health"), up("http://127.0.0.1:11434/api/tags"),
        "up" if tunnel_proc.poll() is None else "DOWN", gpu, (" | b-roll: " + running[0]) if running else ""), flush=True)
    time.sleep(60)''')

nb = {"metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                              "language": "python", "sourceType": "notebook", "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4, "cells": cells}
out = os.path.join(os.path.dirname(SRC), "openshorts-studio.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
