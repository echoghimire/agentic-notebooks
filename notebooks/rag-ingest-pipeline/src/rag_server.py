"""RAG Ingest Pipeline: a heavy document vectorizer on port 7860.

Runs as its own process (started by the notebook with studio_http.spawn):
    APP_PASSWORD=... python rag_server.py --port 7860

Files, folders, URLs and raw text -> parsed (PDF, DOCX, HTML, Markdown, text, code, notebooks) ->
chunked with page and heading metadata -> embedded on the GPU (BAAI/bge-m3 by default) -> stored per
collection -> searchable (with optional reranking) and exportable as chunks.jsonl + embeddings.npy
for any vector database. With two GPUs, ingestion embeds on GPU 0 while searches use GPU 1.
"""
import argparse
import hashlib
import os
import re
import shutil
import threading
import time
import traceback
import urllib.request
import uuid

import rag_parse as P
import rag_store as R
import studio_http as K

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR", "/kaggle/working/rag")
COLS_DIR = os.path.join(WORK, "collections")
UPLOADS = os.path.join(WORK, "uploads")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
QUERY_PREFIX = os.environ.get("QUERY_PREFIX", "")
DOC_PREFIX = os.environ.get("DOC_PREFIX", "")
MAX_SEQ = int(os.environ.get("EMBED_MAX_TOKENS", "1024"))
BATCH = int(os.environ.get("EMBED_BATCH", "32"))
INPUT_ROOTS = [r for r in ("/kaggle/input", "/kaggle/working") if os.path.isdir(r)] or [WORK]
MAX_FETCH = 200 << 20
MAX_WAIT = 85
log = K.file_logger(os.environ.get("APP_LOG", "/kaggle/working/logs/rag.log"), "rag")

LOCK = threading.RLock()
WAKE = threading.Condition(LOCK)
COLS = {}
JOBS = {}               # id -> job (newest last)
QUEUE = []
M = {"devices": None, "ingest": None, "query": None, "rerank": None, "searcher": None}


# ====================================================================== models
def devices():
    if M["devices"] is None:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        M["devices"] = {"n": n, "ingest": "cuda:0" if n else "cpu",
                        "query": "cuda:1" if n > 1 else ("cuda:0" if n else "cpu")}
    return M["devices"]


class Embedder:
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        self.device = device
        self.model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=False)
        if device.startswith("cuda"):
            self.model.half()
        self.model.max_seq_length = min(MAX_SEQ, self.model.max_seq_length or MAX_SEQ)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.lock = threading.Lock()
        log.info("loaded %s on %s (dim %d)", EMBED_MODEL, device, self.dim)

    def encode(self, texts, prefix=""):
        with self.lock:
            return self.model.encode([prefix + t for t in texts], batch_size=BATCH, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype("float16")


def embedder(role):
    with LOCK:
        if M[role] is None:
            dv = devices()
            if role == "query" and dv["query"] == dv["ingest"] and M["ingest"] is not None:
                M["query"] = M["ingest"]
            elif role == "ingest" and dv["query"] == dv["ingest"] and M["query"] is not None:
                M["ingest"] = M["query"]
            else:
                M[role] = Embedder(dv[role])
        return M[role]


def reranker():
    with LOCK:
        if M["rerank"] is None:
            from sentence_transformers import CrossEncoder
            dev = devices()["query"]
            ce = CrossEncoder(RERANK_MODEL, device=dev, max_length=1024)
            if dev.startswith("cuda"):
                ce.model.half()
            M["rerank"] = ce
            log.info("loaded reranker %s on %s", RERANK_MODEL, dev)
        return M["rerank"]


def searcher():
    if M["searcher"] is None:
        M["searcher"] = R.Searcher(devices()["query"])
    return M["searcher"]


# ====================================================================== collections
def get_collection(name, create=False, **kw):
    with LOCK:
        if name in COLS:
            return COLS[name]
        try:
            col = R.Collection(COLS_DIR, name)
        except KeyError:
            if not create:
                raise K.HTTPError(404, "no collection named %r (create it, or ingest into it)" % name)
            col = R.Collection(COLS_DIR, name, model=EMBED_MODEL, create=True, **kw)
            log.info("created collection %s", name)
        COLS[name] = col
        return col


def create_collection(name, chunk_size=1200, chunk_overlap=150):
    if not R.NAME.match(str(name or "")):
        raise ValueError("collection names use letters, digits, - and _ (max 40)")
    if os.path.exists(os.path.join(COLS_DIR, name, "meta.json")):
        raise ValueError("collection %r already exists" % name)
    size, overlap = int(chunk_size or 1200), int(chunk_overlap or 0)
    if not 200 <= size <= 8000 or not 0 <= overlap <= size // 2:
        raise ValueError("chunk_size must be 200-8000 and chunk_overlap 0-chunk_size/2")
    return get_collection(name, create=True, chunk_size=size, chunk_overlap=overlap).stats()


def list_collections():
    out = []
    if os.path.isdir(COLS_DIR):
        for name in sorted(os.listdir(COLS_DIR)):
            if os.path.exists(os.path.join(COLS_DIR, name, "meta.json")):
                out.append(get_collection(name).stats())
    return out


def delete_collection(name):
    col = get_collection(name)
    with LOCK:
        if any(j["collection"] == name and j["state"] in ("queued", "running") for j in JOBS.values()):
            raise ValueError("an ingest job is using this collection")
        COLS.pop(name, None)
        if M["searcher"]:
            M["searcher"].forget(name)
        col.destroy()
    return {"deleted": name}


# ====================================================================== ingest jobs
def safe_input(path):
    full = os.path.realpath(str(path or ""))
    if not any(full == r or full.startswith(os.path.realpath(r) + os.sep) for r in INPUT_ROOTS):
        raise ValueError("paths must be under " + " or ".join(INPUT_ROOTS))
    if not os.path.exists(full):
        raise ValueError("no such file or folder: %s" % path)
    return full


def expand(items):
    """Folders become their supported files."""
    out = []
    for it in items:
        if it["type"] == "path" and os.path.isdir(it["path"]):
            for dp, dn, fn in os.walk(it["path"]):
                dn.sort()
                for f in sorted(fn):
                    if f.lower().endswith(P.SUPPORTED):
                        out.append({"type": "path", "path": os.path.join(dp, f), "name": f})
                        if len(out) > 20000:
                            raise ValueError("more than 20,000 files; ingest sub-folders separately")
        else:
            out.append(it)
    return out


def new_job(colname, items):
    if not items:
        raise ValueError("nothing to ingest: give urls, paths or texts")
    get_collection(colname, create=True)
    items = expand(items)
    job = {"id": uuid.uuid4().hex[:12], "collection": colname, "state": "queued", "created": time.time(),
           "total": len(items), "done": 0, "added": 0, "duplicates": 0, "failed": 0, "chunks": 0,
           "current": None, "results": [], "items": items}
    with WAKE:
        JOBS[job["id"]] = job
        QUEUE.append(job["id"])
        while len(JOBS) > 200:
            old = next(iter(JOBS))
            if JOBS[old]["state"] in ("queued", "running"):
                break
            JOBS.pop(old)
        WAKE.notify_all()
    log.info("job %s: %d items into %s", job["id"], len(items), colname)
    return job


def public(job, results=20):
    out = {k: v for k, v in job.items() if k not in ("items", "results")}
    out["results"] = job["results"][-results:] if results else []
    out["progress"] = round(job["done"] / job["total"], 3) if job["total"] else 1.0
    return out


def fetch_url(url, dest_dir):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (agentic-notebooks RAG ingest)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        ctype = r.headers.get("Content-Type", "")
        name = os.path.basename(r.geturl().split("?")[0].rstrip("/")) or "page"
        kind = P.kind_of(name, ctype) or ("html" if "html" in ctype else None)
        if kind is None:
            raise ValueError("unsupported content type %r" % ctype)
        if P.kind_of(name) is None:
            name += {"html": ".html", "pdf": ".pdf", "docx": ".docx", "markdown": ".md"}.get(kind, ".txt")
        path = os.path.join(dest_dir, uuid.uuid4().hex[:8] + "_" + K.safe_name(name, 80))
        size = 0
        with open(path, "wb") as f:
            while True:
                data = r.read(1 << 20)
                if not data:
                    break
                size += len(data)
                if size > MAX_FETCH:
                    raise ValueError("larger than %d MB" % (MAX_FETCH >> 20))
                f.write(data)
    return path, name, ctype


def ingest_one(col, it):
    tmp = None
    try:
        if it["type"] == "text":
            text = str(it.get("text") or "")
            if not text.strip():
                raise ValueError("empty text")
            title = str(it.get("title") or text.strip().splitlines()[0][:80])
            blocks, _ = P._paras(text, markdown=True)
            parsed = {"title": title, "kind": "text", "blocks": blocks, "pages": None, "warnings": []}
            source, raw = it.get("source") or "text:" + title, text.encode("utf-8")
        else:
            if it["type"] == "url":
                os.makedirs(UPLOADS, exist_ok=True)
                tmp, name, ctype = fetch_url(it["url"], UPLOADS)
                path, source = tmp, it["url"]
            else:
                path, name, ctype = it["path"], it.get("name") or os.path.basename(it["path"]), ""
                source = it.get("source") or it["path"]
                if it["type"] == "upload":
                    tmp = path
            raw = open(path, "rb").read()
            parsed = P.parse_file(path, name, ctype)
        sha = hashlib.sha256(raw).hexdigest()
        dup = col.find_sha(sha)
        if dup:
            return {"source": source, "status": "duplicate", "document_id": dup}
        chunks = P.chunk_blocks(parsed["blocks"], col.meta["chunk_size"], col.meta["chunk_overlap"])
        if not chunks:
            return {"source": source, "status": "empty", "warnings": parsed["warnings"] or ["no text found"]}
        vecs = embedder("ingest").encode([P.embed_text(c, parsed["title"]) for c in chunks], DOC_PREFIX)
        doc_id = col.add_document({"source": source, "title": parsed["title"], "kind": parsed["kind"], "sha256": sha,
                                   "pages": parsed["pages"], "meta": {"warnings": parsed["warnings"]}}, chunks, vecs)
        return {"source": source, "status": "added", "document_id": doc_id, "title": parsed["title"],
                "chunks": len(chunks), "warnings": parsed["warnings"]}
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def worker():
    while True:
        with WAKE:
            while not QUEUE:
                WAKE.wait(30)
            job = JOBS.get(QUEUE.pop(0))
        if not job:
            continue
        job.update(state="running", started=time.time())
        try:
            col = get_collection(job["collection"], create=True)
            for it in job["items"]:
                job["current"] = it.get("url") or it.get("name") or it.get("path") or it.get("title") or "text"
                try:
                    res = ingest_one(col, it)
                except Exception as e:
                    log.warning("ingest failed for %s: %s", job["current"], e)
                    res = {"source": job["current"], "status": "error", "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
                job["results"].append(res)
                job["done"] += 1
                job["added"] += res["status"] == "added"
                job["duplicates"] += res["status"] == "duplicate"
                job["failed"] += res["status"] in ("error", "empty")
                job["chunks"] += res.get("chunks", 0) if res["status"] == "added" else 0
            job.update(state="done", current=None, finished=time.time())
        except Exception as e:
            log.error("job %s failed\n%s", job["id"], traceback.format_exc())
            job.update(state="error", error=str(e)[:500], current=None)
        for it in job["items"]:
            if it["type"] == "upload" and os.path.exists(it["path"]):
                os.remove(it["path"])
        log.info("job %s: %s, %d added, %d chunks", job["id"], job["state"], job["added"], job["chunks"])


def wait_job(jid, seconds):
    job = JOBS.get(jid)
    if not job:
        raise K.HTTPError(404, "unknown job %r (jobs are kept in memory until the server restarts)" % jid)
    t = time.time()
    while job["state"] in ("queued", "running") and time.time() - t < max(0, min(MAX_WAIT, float(seconds or 0))):
        time.sleep(1)
    return job


# ====================================================================== search
def search(colname, query, top_k=5, rerank=True, document_ids=None):
    col = get_collection(colname)
    query = str(query or "").strip()
    if not query:
        raise ValueError("query is empty")
    top_k = max(1, min(50, int(top_k or 5)))
    if col.meta.get("model") and col.meta["model"] != EMBED_MODEL:
        raise ValueError("this collection was embedded with %s but the server uses %s" % (col.meta["model"], EMBED_MODEL))
    t = time.time()
    q = embedder("query").encode([query], QUERY_PREFIX)[0]
    allowed = None
    if document_ids:
        ids = [int(d) for d in document_ids]
        allowed = [r[0] for r in col.db.execute("SELECT id FROM chunks WHERE deleted=0 AND doc_id IN (%s)"
                                                % ",".join("?" * len(ids)), ids)]
    hits = searcher().search(col, q, top_k * 4 if rerank else top_k, allowed)
    found = {c["chunk_id"]: c for c in (x["chunks"][0] for x in col.get_chunks([h[0] for h in hits]))}
    results = [dict(found[cid], score=round(s, 4)) for cid, s in hits if cid in found]
    if rerank and results:
        scores = reranker().predict([(query, r["text"]) for r in results], batch_size=16, show_progress_bar=False)
        for r, s in zip(results, scores):
            r["rerank_score"] = round(float(s), 4)
        results.sort(key=lambda r: -r["rerank_score"])
    return {"query": query, "collection": colname, "results": results[:top_k], "ms": round((time.time() - t) * 1000)}


def list_inputs():
    out = []
    for root in INPUT_ROOTS:
        for dp, dn, fn in os.walk(root):
            if os.path.realpath(dp).startswith(os.path.realpath(WORK)):
                dn[:] = []
                continue
            dn.sort()
            files = [f for f in sorted(fn) if f.lower().endswith(P.SUPPORTED)]
            if files:
                out.append({"folder": dp, "files": len(files), "examples": files[:5]})
            if len(out) >= 300:
                return out
    return out


def info():
    dv = devices()
    return {"embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL, "devices": dv, "gpus": K.gpu_info(),
            "loaded": {k: M[k] is not None for k in ("ingest", "query", "rerank")},
            "jobs": {s: sum(1 for j in JOBS.values() if j["state"] == s) for s in ("queued", "running", "done", "error")},
            "supported": sorted(P.KINDS), "input_roots": INPUT_ROOTS}


# ====================================================================== app
def build_app():
    app = K.App("RAG Ingest Pipeline", password=os.environ.get("APP_PASSWORD"), log=log, max_body=1 << 30,
                instructions=(
                    "Document vectorizer and semantic search. ingest() queues files/folders under /kaggle/input "
                    "(see list_input_files), URLs or raw texts into a collection (created on first use) and returns "
                    "a job; poll job_status. search() returns the best chunks with title, source, pages and scores; "
                    "use get_chunks with context>0 to read around a hit. export_collection gives a zip "
                    "(chunks.jsonl + embeddings.npy) for any vector database. Uploads go through REST: "
                    "POST /api/collections/<name>/upload?name=<file> with the raw file body."))
    app.page("/", os.path.join(HERE, "rag_ui.html"))
    app.static("/static/", HERE)

    @app.route("GET", "/api/info")
    def _info(req):
        return info()

    @app.route("GET", "/api/collections")
    def _cols(req):
        return {"collections": list_collections()}

    @app.route("POST", "/api/collections")
    def _create(req):
        b = req.json()
        return create_collection(b.get("name"), b.get("chunk_size", 1200), b.get("chunk_overlap", 150))

    @app.route("POST", r"/api/collections/(?P<c>[\w-]+)/delete")
    def _delcol(req):
        return delete_collection(req.params["c"])

    @app.route("GET", r"/api/collections/(?P<c>[\w-]+)/documents")
    def _docs(req):
        return get_collection(req.params["c"]).documents(req.arg("offset", 0, int), min(200, req.arg("limit", 50, int)),
                                                     req.arg("q", "") or "")

    @app.route("GET", r"/api/collections/(?P<c>[\w-]+)/documents/(?P<d>\d+)")
    def _doc(req):
        return get_collection(req.params["c"]).document(req.params["d"])

    @app.route("POST", r"/api/collections/(?P<c>[\w-]+)/documents/(?P<d>\d+)/delete")
    def _deldoc(req):
        return get_collection(req.params["c"]).delete_document(req.params["d"])

    @app.route("POST", r"/api/collections/(?P<c>[\w-]+)/upload")
    def _upload(req):
        name = K.safe_name(req.arg("name", "upload.txt"), 100) or "upload.txt"
        if P.kind_of(name) is None:
            raise ValueError("unsupported file type %s; supported: %s" % (name, " ".join(sorted(P.KINDS))))
        os.makedirs(UPLOADS, exist_ok=True)
        path = os.path.join(UPLOADS, uuid.uuid4().hex[:8] + "_" + name)
        req.save_body(path, 1 << 30)
        try:
            job = new_job(req.params["c"], [{"type": "upload", "path": path, "name": name, "source": "upload:" + name}])
        except BaseException:
            os.remove(path)
            raise
        return public(job)

    @app.route("POST", r"/api/collections/(?P<c>[\w-]+)/ingest")
    def _ingest(req):
        b = req.json()
        return public(t_ingest(req.params["c"], b.get("urls"), b.get("paths"), b.get("texts")))

    @app.route("POST", r"/api/collections/(?P<c>[\w-]+)/search")
    def _search(req):
        b = req.json()
        return search(req.params["c"], b.get("query"), b.get("top_k", 5), b.get("rerank", True), b.get("document_ids"))

    @app.route("GET", r"/api/collections/(?P<c>[\w-]+)/export")
    def _export(req):
        col = get_collection(req.params["c"])
        return K.ZipResponse(col.export_files(), "%s-export.zip" % col.name)

    @app.route("GET", "/api/jobs")
    def _jobs(req):
        return {"jobs": [public(j, 0) for j in reversed(list(JOBS.values()))]}

    @app.route("GET", r"/api/jobs/(?P<j>\w+)")
    def _job(req):
        return public(wait_job(req.params["j"], req.arg("wait", 0, float)), 500)

    @app.route("GET", "/api/sources")
    def _sources(req):
        return {"folders": list_inputs()}

    # ---------------------------------------------------------------- MCP tools
    @app.tool("list_collections", "Collections with document and chunk counts, embedding model and chunk settings.")
    def t_cols():
        return list_collections()

    @app.tool("create_collection", "Create an empty collection. ingest() also creates one with default settings.", {
        "name": {"type": "string"}, "chunk_size": {"type": "integer", "default": 1200, "description": "characters"},
        "chunk_overlap": {"type": "integer", "default": 150}}, ["name"])
    def t_create(name, chunk_size=1200, chunk_overlap=150):
        return create_collection(name, chunk_size, chunk_overlap)

    @app.tool("ingest", "Add documents to a collection: URLs (web pages, PDFs, DOCX...), file or folder paths under "
              "/kaggle/input, and/or raw texts. Duplicates are skipped. Returns a job; poll job_status.", {
                  "collection": {"type": "string"},
                  "urls": {"type": "array", "items": {"type": "string"}},
                  "paths": {"type": "array", "items": {"type": "string"}},
                  "texts": {"type": "array", "items": {"type": "object", "properties": {
                      "title": {"type": "string"}, "text": {"type": "string"}, "source": {"type": "string"}}}},
                  "wait_seconds": {"type": "number", "default": 0}}, ["collection"])
    def t_ingest(collection, urls=None, paths=None, texts=None, wait_seconds=0):
        colname = collection
        if not R.NAME.match(str(colname or "")):
            raise ValueError("collection names use letters, digits, - and _ (max 40)")
        items = []
        for u in urls or []:
            if not re.match(r"^https?://", str(u)):
                raise ValueError("urls must start with http:// or https://: %s" % u)
            items.append({"type": "url", "url": str(u)})
        for pth in paths or []:
            items.append({"type": "path", "path": safe_input(pth), "name": os.path.basename(str(pth))})
        for t in texts or []:
            if not isinstance(t, dict):
                t = {"text": str(t)}
            items.append({"type": "text", "text": t.get("text", ""), "title": t.get("title"), "source": t.get("source")})
        job = new_job(colname, items)
        return public(wait_job(job["id"], wait_seconds)) if wait_seconds else public(job)

    @app.tool("job_status", "Progress and per-item results of an ingest job.", {
        "job_id": {"type": "string"}, "wait_seconds": {"type": "number", "default": 0}}, ["job_id"])
    def t_job(job_id, wait_seconds=0):
        return public(wait_job(job_id, wait_seconds), 100)

    @app.tool("search", "Semantic search in a collection. Returns chunks with text, title, source, pages, heading and "
              "scores (rerank_score when rerank is on).", {
                  "collection": {"type": "string"}, "query": {"type": "string"},
                  "top_k": {"type": "integer", "default": 5, "maximum": 50},
                  "rerank": {"type": "boolean", "default": True},
                  "document_ids": {"type": "array", "items": {"type": "integer"}, "description": "only these documents"}},
              ["collection", "query"])
    def t_search(collection, query, top_k=5, rerank=True, document_ids=None):
        return search(collection, query, top_k, rerank, document_ids)

    @app.tool("get_chunks", "Read chunks by id, with `context` neighbouring chunks on each side.", {
        "collection": {"type": "string"}, "chunk_ids": {"type": "array", "items": {"type": "integer"}},
        "context": {"type": "integer", "default": 1, "maximum": 5}}, ["collection", "chunk_ids"])
    def t_chunks(collection, chunk_ids, context=1):
        return get_collection(collection).get_chunks(chunk_ids, context)

    @app.tool("list_documents", "Documents in a collection, newest first.", {
        "collection": {"type": "string"}, "q": {"type": "string", "description": "filter by title or source"},
        "offset": {"type": "integer", "default": 0}, "limit": {"type": "integer", "default": 50}}, ["collection"])
    def t_docs(collection, q="", offset=0, limit=50):
        return get_collection(collection).documents(offset, min(200, int(limit)), q or "")

    @app.tool("delete_document", "Remove a document (and its chunks) from search.", {
        "collection": {"type": "string"}, "document_id": {"type": "integer"}}, ["collection", "document_id"])
    def t_deldoc(collection, document_id):
        return get_collection(collection).delete_document(document_id)

    @app.tool("export_collection", "Where to download a collection as a zip: chunks.jsonl, embeddings.npy (float16, "
              "L2-normalised), documents.jsonl, manifest.json.", {"collection": {"type": "string"}}, ["collection"])
    def t_export(collection):
        st = get_collection(collection).stats()
        return {"download": "/api/collections/%s/export" % collection, "documents": st["documents"],
                "chunks": st["chunks"], "note": "GET it with the same password (Authorization: Bearer <password>)"}

    @app.tool("list_input_files", "Folders under /kaggle/input with supported documents, to pass to ingest as paths.")
    def t_inputs():
        return list_inputs()

    @app.tool("server_status", "Models, devices, GPU memory and job counts.")
    def t_status():
        return info()

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    os.makedirs(COLS_DIR, exist_ok=True)
    shutil.rmtree(UPLOADS, ignore_errors=True)       # leftovers from a previous run
    threading.Thread(target=worker, daemon=True).start()
    build_app().serve_forever(ap.parse_args().port)
