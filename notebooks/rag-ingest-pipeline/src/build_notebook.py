"""Generates rag-ingest-pipeline.ipynb. Run: python src/build_notebook.py"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "..", "..", "..", "shared"))
import nbkit  # noqa: E402

APP_DIR = "/kaggle/working/rag_app"
nb = nbkit.Notebook(APP_DIR)

nb.md("""
# RAG Ingest Pipeline on Kaggle: a heavy document vectorizer behind your Cloudflare Tunnel

Use a free Kaggle GPU to turn piles of documents into a searchable vector index, and take the vectors with you:

- **In:** PDF, DOCX, HTML, Markdown, text, CSV/JSON, source code and notebooks, from uploads, URLs, raw text, or whole folders attached as a Kaggle dataset (thousands of files).
- **Processing:** text extraction with page numbers and headings → overlapping chunks → embeddings on the GPU with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (multilingual, 100+ languages, 1024 dimensions). Duplicate files are skipped.
- **Search:** exact semantic search, optionally reranked with [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), from the web page, the API or an agent (MCP).
- **Out:** a zip per collection with `chunks.jsonl` + `embeddings.npy` (+ `documents.jsonl`, `manifest.json`), ready to load into Qdrant, pgvector, Chroma, LanceDB, FAISS...

| Path | What |
|---|---|
| `/` | Collections, ingest, search and export page |
| `/api/*` | JSON API |
| `/mcp` | MCP server for agents (tools: `list_collections`, `create_collection`, `ingest`, `job_status`, `search`, `get_chunks`, `list_documents`, `delete_document`, `export_collection`, `list_input_files`, `server_status`) |

### Before you run
1. **Accelerator:** GPU T4 x2 (ingestion embeds on GPU 0 while searches and reranking use GPU 1). One GPU or even CPU works, more slowly. **Internet:** on.
2. **A Cloudflare Tunnel** with a public hostname pointing to `http://localhost:7860`.
3. **Kaggle secrets** (Add-ons → Secrets, then tick them for this notebook):
   - `RAG_TUNNEL_TOKEN`: the tunnel token. Without it everything still runs, only without a public URL.
   - `RAG_UI_PASSWORD` (recommended): browser login is any username + this password; agents send `Authorization: Bearer <password>`. Without it a random password is printed below.
4. **Big corpora:** attach them as a Kaggle dataset (Add Input) and ingest the folder from the page or with the `ingest` tool, instead of uploading through Cloudflare (100 MB per request).
5. **Run All.** First run: about 4 minutes (install + 4.5 GB of models).

Scanned PDFs without a text layer need OCR first; the page reports which pages had no text. Collections live in `/kaggle/working/rag/collections` and disappear when the session ends: export what you want to keep. Logs: `/kaggle/working/logs/`.
""")

nb.code("""
# 1. Settings
EMBED_MODEL = "BAAI/bge-m3"                    # any sentence-transformers model; e.g. "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"       # any cross-encoder
QUERY_PREFIX, DOC_PREFIX = "", ""              # e5 models want "query: " and "passage: "
EMBED_BATCH = 32                               # lower it if you see CUDA out-of-memory errors in the log
PORT = 7860
WORK_DIR = "/kaggle/working/rag"
LOG_DIR = "/kaggle/working/logs"
APP_DIR = %r
print("Settings saved.")
""" % APP_DIR)

nb.md("### 2. App files\nThese cells only write files into `APP_DIR`.")
nb.kit_files()
for name, title in [("rag_parse.py", "Parsing (PDF, DOCX, HTML, Markdown, text, code) and chunking"),
                    ("rag_store.py", "Collections: SQLite + float16 vectors, exact search, export"),
                    ("rag_server.py", "Server on port 7860: ingest queue, search, API and MCP"),
                    ("rag_ui.html", "Page")]:
    nb.writefile(os.path.join(SRC, name), title)

nb.code("# 3. Load the helpers\n" + nbkit.setup_cell(APP_DIR, "Notebook"))

nb.code(r'''
# 4. Install and download the models (quiet; about 4 minutes; re-running skips finished steps)
import subprocess
''' + nbkit.PIP_KEEP_CORE + r'''

os.makedirs(LOG_DIR, exist_ok=True)
print("sentence-transformers, pymupdf, python-docx...")
pip_install("sentence-transformers", "pymupdf", "python-docx")
K.ensure_cloudflared()
from huggingface_hub import snapshot_download
for m in (EMBED_MODEL, RERANK_MODEL):
    print("Downloading", m, "...")
    snapshot_download(m, token=K.kaggle_secret("HF_TOKEN"),
                      ignore_patterns=["onnx/*", "*.onnx", "*.msgpack", "*.h5", "*.ot", "openvino/*"])
gpus = K.gpu_info()
print("Install complete. GPUs:", ", ".join("%d: %s" % (g["index"], g["name"]) for g in gpus) or "none (CPU is slow)")
''')

nb.code(r'''
# 5. Start the pipeline on port 7860 (background process). Log: /kaggle/working/logs/rag.log
PASSWORD = K.ui_password("RAG_UI_PASSWORD")
env = dict(os.environ, APP_PASSWORD=PASSWORD, WORK_DIR=WORK_DIR, APP_LOG=LOG_DIR + "/rag.log", PYTHONUNBUFFERED="1",
           EMBED_MODEL=EMBED_MODEL, RERANK_MODEL=RERANK_MODEL, QUERY_PREFIX=QUERY_PREFIX, DOC_PREFIX=DOC_PREFIX,
           EMBED_BATCH=str(EMBED_BATCH), TOKENIZERS_PARALLELISM="false")
K.spawn("rag-pipeline", [sys.executable, os.path.join(APP_DIR, "rag_server.py"), "--port", str(PORT)],
        LOG_DIR + "/rag.log", env=env, cwd=APP_DIR)
if not K.wait_http("http://127.0.0.1:%d/health" % PORT, timeout=60, name="rag-pipeline"):
    print(K.tail(LOG_DIR + "/rag.log", 40))
    raise RuntimeError("The pipeline did not start; log above")
for host in ("127.0.0.1", "[::1]"):
    print(host, "->", "up" if K.wait_http("http://%s:%d/health" % (host, PORT), timeout=3) else "not reachable")
print("Models load on first use (about 30 s).")
''')

nb.code(r'''
# 6. Publish through your Cloudflare Tunnel (token from the RAG_TUNNEL_TOKEN secret, never printed)
tunnel = K.start_tunnel("RAG_TUNNEL_TOKEN", PORT, LOG_DIR + "/cloudflared.log")
''')

nb.code(r'''
# 7. Keep-alive monitor. Leave it running; interrupting it only stops the status lines.
def jobs_line():
    j = K.api_get(PORT, "/api/info", PASSWORD)["jobs"]
    return "ingest jobs: %d running, %d queued" % (j["running"], j["queued"])

K.keep_alive(PORT, ["rag-pipeline", "cloudflared"], extra=jobs_line)
''')

nb.save(os.path.join(SRC, "..", "rag-ingest-pipeline.ipynb"))
