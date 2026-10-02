"""Generates video-studio.ipynb. Run: python src/build_notebook.py"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "..", "..", "..", "shared"))
import nbkit  # noqa: E402

APP_DIR = "/kaggle/working/video_app"
nb = nbkit.Notebook(APP_DIR)

nb.md("""
# Video Studio on Kaggle: any link → narrated video, as landscape and reel

Paste a link (a **GitHub repo**, an **article** or any web page, a **YouTube** video, a **PDF**) or just describe an idea, and get back a narrated explainer video with music, rendered both as **16:9 landscape** and **9:16 reel**. Everything runs on the free Kaggle GPU; no paid APIs.

| Step | Runs on |
|---|---|
| Read the source (GitHub API, web page, YouTube subtitles via yt-dlp, PDF text) | CPU |
| Write the script and storyboard (local LLM, Ollama `qwen2.5:7b`) | GPU 0 |
| Draw one image per scene (Stable Diffusion XL) and compose background music (MusicGen) | GPU 1 |
| Narration (Kokoro-82M text-to-speech) | CPU |
| Animate and render both formats in parallel (headless Chromium + ffmpeg), music ducked under the voice | CPU |

Inspired by [nexu-io/html-video](https://github.com/nexu-io/html-video) (Apache 2.0). Here a free local model only fills in a script; ready-made animated templates do the design, so results stay reliable.

| Path | What |
|---|---|
| `/` | The studio page: paste, pick options, watch progress, play and download both videos, edit the script and re-render |
| `/api/*` | JSON API |
| `/mcp` | MCP server for agents (tools: `make_video`, `get_job`, `get_storyboard`, `update_storyboard`, `render`, `list_jobs`, `delete_job`, `list_options`) |

### Before you run
1. **Accelerator:** GPU T4 x2 (one GPU works, more slowly). **Internet:** on.
2. **A Cloudflare Tunnel** with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then tick them for this notebook):
   - `VIDEO_TUNNEL_TOKEN`: the tunnel token. Without it everything still runs, only without a public URL.
   - `VIDEO_UI_PASSWORD` (recommended): browser login is any username + this password; agents send `Authorization: Bearer <password>`. Without it a random password is printed below.
   - `GITHUB_TOKEN` (optional): any GitHub token, to lift the API limit of 60 repository lookups an hour.
4. **Run All.** First run: about 10 minutes (installs + ~12 GB of models). Then a 60-second video takes roughly 6–10 minutes on T4 x2 (images and rendering take most of it); 720p renders about twice as fast.

### Licences
SDXL 1.0: CreativeML OpenRAIL++ (commercial use allowed). Kokoro-82M: Apache 2.0. Qwen2.5: Apache 2.0. **MusicGen weights are CC-BY-NC 4.0 (non-commercial)**: set `MUSIC_MODEL = ""` for videos you sell or monetise, or add your own music later. Respect the licences of the pages and repos you turn into videos.

Videos are saved in `/kaggle/working/video_studio/jobs/` and disappear when the session ends: download what you want to keep. Logs: `/kaggle/working/logs/`.
""")

nb.code("""
# 1. Settings
LLM_MODEL = "qwen2.5:7b"                                    # "qwen2.5:14b" writes better scripts, slower
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"    # "" = no images (gradient backgrounds only, much faster)
IMAGE_STEPS = 25
MUSIC_MODEL = "facebook/musicgen-small"                     # "" = no music (MusicGen weights are non-commercial)
NARRATION = True                                           # Kokoro text-to-speech
RENDER_QUALITY = "1080p"                                   # or "720p" (about twice as fast); the page can override it
PORT = 7860
WORK_DIR = "/kaggle/working/video_studio"
LOG_DIR = "/kaggle/working/logs"
APP_DIR = %r
print("Settings saved.")
""" % APP_DIR)

nb.md("### 2. App files\nThese cells only write files into `APP_DIR`.")
nb.kit_files()
for name, title in [("vs_source.py", "Reads any link: GitHub, web pages, YouTube, PDF"),
                    ("vs_story.py", "Script and storyboard (local LLM), validation and fallback"),
                    ("vs_scenes.py", "Animated scene templates, landscape and reel, four styles"),
                    ("vs_media.py", "Images (SDXL), narration (Kokoro), music (MusicGen), audio mix"),
                    ("vs_render.py", "Renderer: headless Chromium frames -> ffmpeg"),
                    ("vs_server.py", "Studio server on port 7860: jobs, API and MCP"),
                    ("vs_ui.html", "Studio page")]:
    nb.writefile(os.path.join(SRC, name), title)

nb.code("# 3. Load the helpers\n" + nbkit.setup_cell(APP_DIR, "Notebook"))

nb.code(r'''
# 4. Install everything (quiet; 5-8 minutes; re-running skips finished steps)
import shutil, subprocess
''' + nbkit.PIP_KEEP_CORE + r'''

def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-2500:])
        raise RuntimeError("failed: " + cmd[:100])

os.makedirs(LOG_DIR, exist_ok=True)
print("System packages (ffmpeg, espeak-ng, fonts)...")
sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg espeak-ng zstd "
   "fonts-noto-core fonts-noto-mono fonts-dejavu-core > /dev/null")
print("Python packages...")
pip_install("playwright", "yt-dlp", "pymupdf", "soundfile", "diffusers", "accelerate", "kokoro>=0.9")
print("Headless Chromium...")
sh(sys.executable + " -m playwright install --with-deps chromium > /dev/null")
if not shutil.which("ollama"):
    print("Ollama...")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
K.ensure_cloudflared()

print("Downloading models (first time only)...")
from huggingface_hub import snapshot_download
if IMAGE_MODEL:
    snapshot_download(IMAGE_MODEL, allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*"])
    if "xl" in IMAGE_MODEL.lower():
        snapshot_download("madebyollin/sdxl-vae-fp16-fix", allow_patterns=["*.json", "*.safetensors"])
if MUSIC_MODEL:
    snapshot_download(MUSIC_MODEL, ignore_patterns=["*.bin", "*.msgpack", "*.h5"])
if NARRATION:
    snapshot_download("hexgrad/Kokoro-82M", allow_patterns=["*.json", "*.pth", "voices/af_heart.pt", "voices/am_michael.pt"])
gpus = K.gpu_info()
print("Install complete. GPUs:", ", ".join("%d: %s" % (g["index"], g["name"]) for g in gpus) or "none - pick 'GPU T4 x2'")
''')

nb.code(r'''
# 5. Start Ollama (the scriptwriter) on GPU 0 and pull the model. Log: /kaggle/working/logs/ollama.log
import subprocess
ollama_env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", CUDA_VISIBLE_DEVICES="0", OLLAMA_CONTEXT_LENGTH="16384")
K.spawn("ollama", ["ollama", "serve"], LOG_DIR + "/ollama.log", env=ollama_env)
if not K.wait_http("http://127.0.0.1:11434/api/tags", timeout=60, name="ollama"):
    print(K.tail(LOG_DIR + "/ollama.log"))
    raise RuntimeError("Ollama did not start; log above")
print("Pulling %s (first time ~5 GB)..." % LLM_MODEL)
p = subprocess.run(["ollama", "pull", LLM_MODEL], env=ollama_env, capture_output=True, text=True)
if p.returncode != 0:
    print(p.stderr[-1500:])
    raise RuntimeError("ollama pull failed")
print("Scriptwriter ready.")
''')

nb.code(r'''
# 6. Start the studio on port 7860 (background process). Log: /kaggle/working/logs/video_studio.log
PASSWORD = K.ui_password("VIDEO_UI_PASSWORD")
n_gpu = len(K.gpu_info())
env = dict(os.environ, APP_PASSWORD=PASSWORD, WORK_DIR=WORK_DIR, APP_LOG=LOG_DIR + "/video_studio.log",
           PYTHONUNBUFFERED="1", OLLAMA_URL="http://127.0.0.1:11434", LLM_MODEL=LLM_MODEL,
           IMAGE_MODEL=IMAGE_MODEL, IMAGE_STEPS=str(IMAGE_STEPS), MUSIC_MODEL=MUSIC_MODEL,
           TTS="1" if NARRATION else "0", TTS_DEVICE="cpu", RENDER_QUALITY=RENDER_QUALITY,
           MEDIA_DEVICE="cuda:1" if n_gpu > 1 else "cuda:0",
           LLM_KEEP_ALIVE="10m" if n_gpu > 1 else "0",      # one GPU: free the LLM's memory before drawing
           UNLOAD_AFTER="0" if n_gpu > 1 else "1", TOKENIZERS_PARALLELISM="false")
env.pop("GITHUB_TOKEN", None)
gh = K.kaggle_secret("GITHUB_TOKEN")
if gh:
    env["GITHUB_TOKEN"] = gh
K.spawn("video-studio", [sys.executable, os.path.join(APP_DIR, "vs_server.py"), "--port", str(PORT)],
        LOG_DIR + "/video_studio.log", env=env, cwd=APP_DIR)
if not K.wait_http("http://127.0.0.1:%d/health" % PORT, timeout=60, name="video-studio"):
    print(K.tail(LOG_DIR + "/video_studio.log", 40))
    raise RuntimeError("The studio did not start; log above")
for host in ("127.0.0.1", "[::1]"):
    print(host, "->", "up" if K.wait_http("http://%s:%d/health" % (host, PORT), timeout=3) else "not reachable")
print("Models load on the first video (1-2 minutes extra).")
''')

nb.code(r'''
# 7. Publish through your Cloudflare Tunnel (token from the VIDEO_TUNNEL_TOKEN secret, never printed)
tunnel = K.start_tunnel("VIDEO_TUNNEL_TOKEN", PORT, LOG_DIR + "/cloudflared.log")
''')

nb.code(r'''
# 8. Keep-alive monitor. Leave it running; interrupting it only stops the status lines.
def queue_line():
    q = K.api_get(PORT, "/api/info", PASSWORD)["queue"]
    return "videos: %d rendering, %d waiting, %d done" % (q["running"], q["queued"], q["done"])

K.keep_alive(PORT, ["video-studio", "ollama", "cloudflared"], extra=queue_line)
''')

nb.save(os.path.join(SRC, "..", "video-studio.ipynb"))
