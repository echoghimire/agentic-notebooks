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

Paste a link (a **news article in Nepali or English**, any web page, a **GitHub repo**, a **YouTube** video, a **PDF**) or describe an idea, and get back a narrated video with music, rendered both as **16:9 landscape** and **9:16 reel**. It uses the link's **own photos**, speaks the link's **language** (Nepali and Hindi included), and picks a look from the content: broadcast-style graphics for news, documentary, tech and more. Everything runs on the free Kaggle GPU; no paid APIs.

| Step | Tool | Runs on |
|---|---|---|
| Read the link: article text, date, photos and captions | [trafilatura](https://github.com/adbar/trafilatura) + headless Chromium fallback, GitHub API, yt-dlp, pymupdf | CPU |
| Pick and edit the photos; describe them so each scene gets the right one | Pillow, Gemma 3 vision | CPU / GPU 0 |
| Write the script in the video's language (checked and retried if it drifts) | Ollama `gemma3:12b` (140+ languages) | GPU 0 |
| Narration | Indic Parler-TTS (Nepali, Hindi...), Piper (light Nepali), Kokoro (English...) | GPU 1 / CPU |
| Illustrations only for non-news topics without photos, and background music matched to the tone | SDXL, MusicGen | GPU 1 |
| Animate (broadcast lower-thirds, kinetic type, film grain, wipes) and render both formats in parallel | headless Chromium + ffmpeg | CPU |

Inspired by [nexu-io/html-video](https://github.com/nexu-io/html-video) and [HyperFrames](https://github.com/heygen-com/hyperframes) (Apache 2.0). A local model only fills in a script; ready-made templates do the design, and every frame is rendered exactly.

| Path | What |
|---|---|
| `/` | The studio: paste a link, pick 15/30/60/90 s and formats, watch progress, play and download both videos, edit the script |
| `/api/*` | JSON API |
| `/mcp` | MCP server for agents (tools: `make_video`, `get_job`, `get_storyboard`, `update_storyboard`, `render`, `list_jobs`, `delete_job`, `list_options`) |

### Before you run
1. **Accelerator:** GPU T4 x2 (one GPU works, more slowly). **Internet:** on.
2. **A Cloudflare Tunnel** with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then tick them for this notebook):
   - `VIDEO_TUNNEL_TOKEN`: the tunnel token. Without it everything still runs, only without a public URL.
   - `VIDEO_UI_PASSWORD` (recommended): the password for the studio's sign-in page; agents send `Authorization: Bearer <password>`. Without it a random password is printed below.
   - `GITHUB_TOKEN` (optional): any GitHub token, to lift the API limit of 60 repository lookups an hour.
4. **Run All.** First run: about 12–15 minutes (installs + ~20 GB of models). Then a 60-second video takes roughly 5–10 minutes on T4 x2; 720p renders about twice as fast.

### Licences and responsibility
Gemma 3: Gemma Terms of Use. Indic Parler-TTS, Kokoro, Qwen: Apache 2.0. Piper: MIT (Nepali voice from the OpenSLR corpus). SDXL: CreativeML OpenRAIL++. **MusicGen weights are CC-BY-NC 4.0 (non-commercial)**: set `MUSIC_MODEL = ""` for monetised videos. **Photos and text from a link belong to their publisher**: the video credits the site, but make sure you may reuse them (for example your own site, or with permission).

Videos are saved in `/kaggle/working/video_studio/jobs/` and disappear when the session ends: download what you want to keep. Logs: `/kaggle/working/logs/`.
""")

nb.code("""
# 1. Settings
LLM_MODEL = "gemma3:12b"                                    # writes Nepali well and can look at photos; "gemma3:4b" is faster
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"    # only for non-news topics without photos; "" = never draw
IMAGE_STEPS = 25
MUSIC_MODEL = "facebook/musicgen-small"                     # "" = no music (MusicGen weights are non-commercial)
NARRATION = True
INDIC_VOICES = True                                         # Indic Parler-TTS for Nepali / Hindi (~4 GB, own virtualenv)
RENDER_QUALITY = "1080p"                                    # or "720p" (about twice as fast); the page can override it
PORT = 7860
WORK_DIR = "/kaggle/working/video_studio"
LOG_DIR = "/kaggle/working/logs"
PARLER_ENV = "/tmp/parler_env"                             # outside /kaggle/working (kept out of the notebook's output)
PIPER_DIR = "/kaggle/working/piper"
APP_DIR = %r
print("Settings saved.")
""" % APP_DIR)

nb.md("### 2. App files\nThese cells only write files into `APP_DIR`.")
nb.kit_files()
for name, title in [("vs_source.py", "Reads any link: articles (trafilatura), GitHub, YouTube, PDF; photos and language"),
                    ("vs_story.py", "Script and storyboard in the video's language, validation and fallback"),
                    ("vs_scenes.py", "Animated scene templates: broadcast, documentary and more; landscape and reel"),
                    ("vs_media.py", "Photos, images (SDXL), music (MusicGen), Kokoro voices, audio mix"),
                    ("vs_tts.py", "Voices by language: Indic Parler-TTS, Piper, Kokoro"),
                    ("vs_parler_worker.py", "Indic Parler-TTS worker (runs in its own virtualenv)"),
                    ("vs_render.py", "Renderer: headless Chromium frames -> ffmpeg"),
                    ("vs_server.py", "Studio server on port 7860: jobs, API and MCP"),
                    ("vs_ui.html", "Studio page")]:
    nb.writefile(os.path.join(SRC, name), title)

nb.code("# 3. Load the helpers\n" + nbkit.setup_cell(APP_DIR, "Notebook"))

nb.code(r'''
# 4. Install everything (quiet; 8-12 minutes; re-running skips finished steps)
import shutil, subprocess
''' + nbkit.PIP_KEEP_CORE + r'''

def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-2500:])
        raise RuntimeError("failed: " + cmd[:100])

os.makedirs(LOG_DIR, exist_ok=True)
print("System packages (ffmpeg, espeak-ng, Noto fonts incl. Devanagari)...")
sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg espeak-ng zstd "
   "fonts-noto-core fonts-noto-mono fonts-dejavu-core > /dev/null")
print("Python packages...")
pip_install("playwright", "yt-dlp", "pymupdf", "soundfile", "diffusers", "accelerate", "kokoro>=0.9", "trafilatura", "piper-tts")
print("Headless Chromium...")
sh(sys.executable + " -m playwright install --with-deps chromium > /dev/null")
if not shutil.which("ollama"):
    print("Ollama...")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
K.ensure_cloudflared()

from huggingface_hub import hf_hub_download, snapshot_download
print("Nepali voices...")
os.makedirs(PIPER_DIR, exist_ok=True)
try:
    for ext in (".onnx", ".onnx.json"):
        p = hf_hub_download("rhasspy/piper-voices", "ne/ne_NP/google/medium/ne_NP-google-medium" + ext)
        shutil.copy(p, os.path.join(PIPER_DIR, "ne_NP-google-medium" + ext))
except Exception as e:
    print("  Piper Nepali voice unavailable:", e)
if INDIC_VOICES and not os.path.exists(PARLER_ENV + "/bin/python"):
    try:
        # parler-tts pins its own transformers: a virtualenv that reuses Kaggle's PyTorch keeps it apart
        sh("%s -m venv --system-site-packages %s" % (sys.executable, PARLER_ENV))
        sh("%s/bin/pip install -q --disable-pip-version-check git+https://github.com/huggingface/parler-tts.git" % PARLER_ENV)
    except RuntimeError:
        print("  Indic Parler-TTS did not install; Nepali uses the Piper voice.")
        shutil.rmtree(PARLER_ENV, ignore_errors=True)
if INDIC_VOICES and os.path.exists(PARLER_ENV + "/bin/python"):
    snapshot_download("ai4bharat/indic-parler-tts")
    snapshot_download("google/flan-t5-large", allow_patterns=["*.json", "*.model", "tokenizer*"])

print("Image, music and English voice models (first time only)...")
if IMAGE_MODEL:
    snapshot_download(IMAGE_MODEL, allow_patterns=["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*"])
    if "xl" in IMAGE_MODEL.lower():
        snapshot_download("madebyollin/sdxl-vae-fp16-fix", allow_patterns=["*.json", "*.safetensors"])
if MUSIC_MODEL:
    snapshot_download(MUSIC_MODEL, ignore_patterns=["*.bin", "*.msgpack", "*.h5"])
if NARRATION:
    snapshot_download("hexgrad/Kokoro-82M", allow_patterns=["*.json", "*.pth", "voices/*.pt"])
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
print("Pulling %s (first time ~8 GB)..." % LLM_MODEL)
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
parler_py = PARLER_ENV + "/bin/python" if INDIC_VOICES and os.path.exists(PARLER_ENV + "/bin/python") else ""
env = dict(os.environ, APP_PASSWORD=PASSWORD, WORK_DIR=WORK_DIR, APP_LOG=LOG_DIR + "/video_studio.log",
           PARLER_LOG=LOG_DIR + "/parler.log", PYTHONUNBUFFERED="1", OLLAMA_URL="http://127.0.0.1:11434",
           LLM_MODEL=LLM_MODEL, IMAGE_MODEL=IMAGE_MODEL, IMAGE_STEPS=str(IMAGE_STEPS), MUSIC_MODEL=MUSIC_MODEL,
           TTS="1" if NARRATION else "0", TTS_DEVICE="cpu", RENDER_QUALITY=RENDER_QUALITY,
           PARLER_PYTHON=parler_py, PIPER_DIR=PIPER_DIR,
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
print("Voices:", ", ".join(v for v in K.api_get(PORT, "/api/info", PASSWORD)["voices"]))
print("Models load on the first video (1-3 minutes extra).")
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
