"""Generates whisper-diarization-studio.ipynb. Run: python src/build_notebook.py"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "..", "..", "..", "shared"))
import nbkit  # noqa: E402

APP_DIR = "/kaggle/working/whisper_app"
nb = nbkit.Notebook(APP_DIR)

nb.md("""
# Whisper Diarization Studio on Kaggle: the meeting & audio OS behind your Cloudflare Tunnel

Drop in a meeting recording, podcast, lecture or video and get back:
- a transcript with **who spoke when** (faster-whisper `large-v3-turbo` + pyannote speaker diarization);
- a **meeting summary** with decisions, action items and open questions (local Ollama model, no paid API);
- downloads as Markdown, plain text, SRT and VTT subtitles, or JSON with word timings.

Inputs: upload (≤100 MB through Cloudflare), any URL yt-dlp understands (direct files, YouTube, podcast pages), or files attached to the notebook as a Kaggle dataset (no size limit). Speakers can be renamed; the summary then uses the real names.

| Path | What |
|---|---|
| `/` | The studio page |
| `/api/*` | JSON API (jobs, uploads, downloads, speakers, summaries) |
| `/mcp` | MCP server for agents (tools: `transcribe`, `get_job`, `get_transcript`, `list_jobs`, `rename_speakers`, `summarize`, `delete_job`, `list_input_files`, `server_status`) |

### Before you run
1. **Accelerator:** GPU T4 x2 (Whisper on GPU 0; speaker diarization and the summary model on GPU 1). One GPU works too. **Internet:** on.
2. **A Cloudflare Tunnel** with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then tick them for this notebook):
   - `WHISPER_TUNNEL_TOKEN`: the tunnel token. Without it everything still runs, only without a public URL.
   - `WHISPER_UI_PASSWORD` (recommended): browser login is any username + this password; agents send `Authorization: Bearer <password>`. Without it a random password is printed below.
   - `HF_TOKEN` (for speaker labels): a Hugging Face read token whose account accepted the terms of [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). Without it you still get transcripts, just without speakers.
4. **Run All.** First run: about 8 minutes (installs, Whisper model, Ollama model). A one-hour meeting then takes very roughly 5–10 minutes on a T4.

Jobs are stored in `/kaggle/working/whisper_studio/jobs/` and are wiped when the session ends: download what you want to keep. Logs: `/kaggle/working/logs/`.
""")

nb.code("""
# 1. Settings (edit if you like)
WHISPER_MODEL = "large-v3-turbo"                       # or "large-v3" (slower, a bit more accurate), "medium", "small"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SUMMARY_MODEL = "qwen2.5:7b"                           # any Ollama model; "" turns summaries off (saves ~5 GB download)
PORT = 7860
WORK_DIR = "/kaggle/working/whisper_studio"
LOG_DIR = "/kaggle/working/logs"
APP_DIR = %r
print("Settings saved.")
""" % APP_DIR)

nb.md("### 2. App files\nThese cells only write files into `APP_DIR`.")
nb.kit_files()
nb.writefile(os.path.join(SRC, "whisper_core.py"), "Speaker assignment, transcript formats and summaries (pure Python)")
nb.writefile(os.path.join(SRC, "whisper_server.py"), "Studio server on port 7860: jobs, API and MCP")
nb.writefile(os.path.join(SRC, "whisper_ui.html"), "Studio page")

nb.code("# 3. Load the helpers\n" + nbkit.setup_cell(APP_DIR, "Notebook"))

nb.code(r'''
# 4. Install (quiet; about 4 minutes; re-running skips finished steps)
import shutil, subprocess
''' + nbkit.PIP_KEEP_CORE + r'''

os.makedirs(LOG_DIR, exist_ok=True)
if not shutil.which("ffmpeg"):
    print("ffmpeg...")
    subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True, capture_output=True)
print("faster-whisper, yt-dlp...")
pip_install("faster-whisper", "yt-dlp", "nvidia-cublas-cu12", "nvidia-cudnn-cu12")

DIARIZATION_OK = True
print("pyannote.audio (speaker diarization)...")
try:
    pip_install("pyannote.audio")
except RuntimeError as e:
    DIARIZATION_OK = False
    print("  pyannote.audio did not install next to this Kaggle image's PyTorch; transcripts will have no speakers.")

if SUMMARY_MODEL and not shutil.which("ollama"):
    print("Ollama...")
    subprocess.run("apt-get -qq install -y zstd && curl -fsSL https://ollama.com/install.sh | sh",
                   shell=True, capture_output=True)
K.ensure_cloudflared()

print("Downloading the Whisper model %s ..." % WHISPER_MODEL)
from faster_whisper import download_model
download_model(WHISPER_MODEL)
import torch
print("Install complete. GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "none (CPU is very slow)")
''')

nb.code(r'''
# 5. Start Ollama for summaries (GPU 1 when there are two) and pull the model. Log: /kaggle/working/logs/ollama.log
import torch
N_GPU = torch.cuda.device_count()
if SUMMARY_MODEL:
    ollama_env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", CUDA_VISIBLE_DEVICES="1" if N_GPU > 1 else "0",
                      OLLAMA_CONTEXT_LENGTH="32768", OLLAMA_KEEP_ALIVE="30m")
    K.spawn("ollama", ["ollama", "serve"], LOG_DIR + "/ollama.log", env=ollama_env)
    if not K.wait_http("http://127.0.0.1:11434/api/tags", timeout=60, name="ollama"):
        print(K.tail(LOG_DIR + "/ollama.log"))
        raise RuntimeError("Ollama did not start; log above. Or set SUMMARY_MODEL = \"\" to skip summaries.")
    print("Pulling %s (first time ~5 GB)..." % SUMMARY_MODEL)
    p = subprocess.run(["ollama", "pull", SUMMARY_MODEL], env=ollama_env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-1500:])
        raise RuntimeError("ollama pull failed")
    print("Summary model ready.")
else:
    K.stop_process("ollama")
    print("Summaries are off (SUMMARY_MODEL is empty).")
''')

nb.code(r'''
# 6. Start the studio on port 7860 (background process). Log: /kaggle/working/logs/whisper.log
import glob, site
# CTranslate2 (faster-whisper) loads cuBLAS / cuDNN from the pip-installed NVIDIA wheels
LIB_DIRS = sorted({os.path.dirname(p) for sp in site.getsitepackages() + [site.getusersitepackages()]
                   for pat in ("nvidia/cublas/lib/*.so*", "nvidia/cudnn/lib/*.so*") for p in glob.glob(os.path.join(sp, pat))})
PASSWORD = K.ui_password("WHISPER_UI_PASSWORD")
HF_TOKEN = K.kaggle_secret("HF_TOKEN")
if not HF_TOKEN:
    print("No HF_TOKEN secret: transcripts will not have speaker labels (see the first cell).")
env = dict(os.environ, APP_PASSWORD=PASSWORD, WORK_DIR=WORK_DIR, APP_LOG=LOG_DIR + "/whisper.log",
           WHISPER_MODEL=WHISPER_MODEL, DIARIZATION_MODEL=DIARIZATION_MODEL,
           SUMMARY_MODEL=SUMMARY_MODEL, OLLAMA_URL="http://127.0.0.1:11434", PYTHONUNBUFFERED="1",
           LD_LIBRARY_PATH=":".join(LIB_DIRS + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":"))
env.pop("HF_TOKEN", None)
if HF_TOKEN and globals().get("DIARIZATION_OK", True):
    env["HF_TOKEN"] = HF_TOKEN
K.spawn("whisper-studio", [sys.executable, os.path.join(APP_DIR, "whisper_server.py"), "--port", str(PORT)],
        LOG_DIR + "/whisper.log", env=env, cwd=APP_DIR)
if not K.wait_http("http://127.0.0.1:%d/health" % PORT, timeout=60, name="whisper-studio"):
    print(K.tail(LOG_DIR + "/whisper.log", 40))
    raise RuntimeError("The studio did not start; log above")
for host in ("127.0.0.1", "[::1]"):
    print(host, "->", "up" if K.wait_http("http://%s:%d/health" % (host, PORT), timeout=3) else "not reachable")
print(K.api_get(PORT, "/api/info", PASSWORD)["diarization"])
''')

nb.code(r'''
# 7. Publish through your Cloudflare Tunnel (token from the WHISPER_TUNNEL_TOKEN secret, never printed)
tunnel = K.start_tunnel("WHISPER_TUNNEL_TOKEN", PORT, LOG_DIR + "/cloudflared.log")
''')

nb.code(r'''
# 8. Keep-alive monitor. Leave it running; interrupting it only stops the status lines.
def queue_line():
    q = K.api_get(PORT, "/api/info", PASSWORD)["queue"]
    return "jobs: %d running, %d queued, %d done" % (q["running"], q["queued"], q["done"])

K.keep_alive(PORT, ["whisper-studio", "ollama", "cloudflared"] if SUMMARY_MODEL else ["whisper-studio", "cloudflared"],
             extra=queue_line)
''')

nb.save(os.path.join(SRC, "..", "whisper-diarization-studio.ipynb"))
