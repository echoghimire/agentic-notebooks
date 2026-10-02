"""Collections for the RAG ingest pipeline: SQLite for documents and chunk text, an append-only
float16 matrix for the vectors (row i = chunk id i). No vector-database dependency.

Search is exact (a matrix product over every live vector) on the GPU when torch + CUDA are there,
else on the CPU with numpy. That stays fast into the low millions of chunks on a T4.
"""
import json
import os
import re
import shutil
import sqlite3
import threading
import time

import numpy as np

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, title TEXT, kind TEXT,
    sha256 TEXT, pages INTEGER, chunks INTEGER, chars INTEGER, added REAL, meta TEXT, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS documents_sha ON documents(sha256);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, ord INTEGER, text TEXT,
    page_start INTEGER, page_end INTEGER, heading TEXT, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
"""


class Collection:
    def __init__(self, root, name, model=None, dim=None, chunk_size=1200, chunk_overlap=150, create=False):
        if not NAME.match(name or ""):
            raise ValueError("collection names use letters, digits, - and _ (max 40)")
        self.name, self.dir = name, os.path.join(root, name)
        self.lock = threading.RLock()
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            if not create:
                raise KeyError("no collection named %r" % name)
            os.makedirs(self.dir, exist_ok=True)
            self._write_meta({"name": name, "model": model, "dim": dim, "chunk_size": int(chunk_size),
                              "chunk_overlap": int(chunk_overlap), "created": time.time()})
        self.meta = json.load(open(meta_path))
        self.db = sqlite3.connect(os.path.join(self.dir, "chunks.sqlite"), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.vec_path = os.path.join(self.dir, "vectors.f16")
        self.version = 0
        self._repair()

    def _write_meta(self, meta):
        tmp = os.path.join(self.dir, "meta.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "meta.json"))

    # ------------------------------------------------------------------ vectors
    @property
    def dim(self):
        return self.meta.get("dim")

    def rows(self):
        if not self.dim or not os.path.exists(self.vec_path):
            return 0
        return os.path.getsize(self.vec_path) // (self.dim * 2)

    def _repair(self):
        """After a crash between writing vectors and committing rows, drop the extra vectors."""
        with self.lock:
            n_db = self.db.execute("SELECT COALESCE(MAX(id) + 1, 0) FROM chunks").fetchone()[0]
            if self.dim and self.rows() > n_db:
                with open(self.vec_path, "r+b") as f:
                    f.truncate(n_db * self.dim * 2)

    def vectors(self):
        n = self.rows()
        if not n:
            return np.zeros((0, self.dim or 1), dtype=np.float16)
        return np.memmap(self.vec_path, dtype=np.float16, mode="r", shape=(n, self.dim))

    # ------------------------------------------------------------------ documents
    def find_sha(self, sha):
        r = self.db.execute("SELECT id FROM documents WHERE sha256=? AND deleted=0", (sha,)).fetchone()
        return r[0] if r else None

    def add_document(self, doc, chunks, vecs):
        """doc: {source, title, kind, sha256, pages, meta}; chunks: [{text, heading, page_start, page_end}];
        vecs: float array (len(chunks), dim), already L2-normalised."""
        vecs = np.ascontiguousarray(vecs, dtype=np.float16)
        if len(chunks) != len(vecs):
            raise ValueError("chunks and vectors differ in length")
        with self.lock:
            if not self.dim:
                self.meta["dim"] = int(vecs.shape[1])
                self._write_meta(self.meta)
            if vecs.shape[1] != self.dim:
                raise ValueError("vector size %d does not match this collection (%d); it was built with %s"
                                 % (vecs.shape[1], self.dim, self.meta.get("model")))
            start = self.db.execute("SELECT COALESCE(MAX(id) + 1, 0) FROM chunks").fetchone()[0]
            if self.rows() != start:
                self._repair()
            with open(self.vec_path, "ab") as f:
                f.write(vecs.tobytes())
            cur = self.db.execute(
                "INSERT INTO documents(source, title, kind, sha256, pages, chunks, chars, added, meta) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (doc.get("source"), doc.get("title"), doc.get("kind"), doc.get("sha256"), doc.get("pages"),
                 len(chunks), sum(len(c["text"]) for c in chunks), time.time(), json.dumps(doc.get("meta") or {})))
            doc_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO chunks(id, doc_id, ord, text, page_start, page_end, heading) VALUES(?,?,?,?,?,?,?)",
                [(start + i, doc_id, i, c["text"], c.get("page_start"), c.get("page_end"), c.get("heading"))
                 for i, c in enumerate(chunks)])
            self.db.commit()
            self.version += 1
            return doc_id

    def delete_document(self, doc_id):
        with self.lock:
            n = self.db.execute("UPDATE documents SET deleted=1 WHERE id=? AND deleted=0", (int(doc_id),)).rowcount
            if not n:
                raise KeyError("no document %s in %s" % (doc_id, self.name))
            self.db.execute("UPDATE chunks SET deleted=1 WHERE doc_id=?", (int(doc_id),))
            self.db.commit()
            self.version += 1
            return {"deleted": int(doc_id)}

    def documents(self, offset=0, limit=50, q=""):
        where, args = "deleted=0", []
        if q:
            where += " AND (title LIKE ? OR source LIKE ?)"
            args += ["%" + q + "%"] * 2
        total = self.db.execute("SELECT COUNT(*) FROM documents WHERE " + where, args).fetchone()[0]
        rows = self.db.execute("SELECT id, source, title, kind, pages, chunks, chars, added, meta FROM documents WHERE "
                               + where + " ORDER BY id DESC LIMIT ? OFFSET ?", args + [int(limit), int(offset)])
        keys = ("id", "source", "title", "kind", "pages", "chunks", "chars", "added", "meta")
        docs = []
        for r in rows:
            d = dict(zip(keys, r))
            d["meta"] = json.loads(d["meta"] or "{}")
            docs.append(d)
        return {"total": total, "documents": docs}

    def document(self, doc_id):
        d = self.db.execute("SELECT id, source, title, kind, pages, chunks, chars, added FROM documents "
                            "WHERE id=? AND deleted=0", (int(doc_id),)).fetchone()
        if not d:
            raise KeyError("no document %s in %s" % (doc_id, self.name))
        out = dict(zip(("id", "source", "title", "kind", "pages", "chunks", "chars", "added"), d))
        out["chunk_list"] = self.chunks_of(doc_id)
        return out

    def chunks_of(self, doc_id):
        return [self._chunk(r) for r in self.db.execute(
            "SELECT c.id, c.doc_id, c.ord, c.text, c.page_start, c.page_end, c.heading, d.title, d.source "
            "FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.doc_id=? AND c.deleted=0 ORDER BY c.ord",
            (int(doc_id),))]

    @staticmethod
    def _chunk(r):
        return {"chunk_id": r[0], "document_id": r[1], "ord": r[2], "text": r[3], "page_start": r[4],
                "page_end": r[5], "heading": r[6], "title": r[7], "source": r[8]}

    def get_chunks(self, ids, context=0):
        """Chunks by id; context=N also returns N neighbours on each side within the same document."""
        ids = [int(i) for i in ids][:100]
        context = max(0, min(5, int(context or 0)))
        out = []
        for cid in ids:
            r = self.db.execute("SELECT doc_id, ord FROM chunks WHERE id=? AND deleted=0", (cid,)).fetchone()
            if not r:
                continue
            rows = self.db.execute(
                "SELECT c.id, c.doc_id, c.ord, c.text, c.page_start, c.page_end, c.heading, d.title, d.source "
                "FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.doc_id=? AND c.ord BETWEEN ? AND ? "
                "AND c.deleted=0 ORDER BY c.ord", (r[0], r[1] - context, r[1] + context))
            out.append({"chunk_id": cid, "chunks": [self._chunk(x) for x in rows]})
        return out

    def stats(self):
        docs, chunks, chars = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(chunks),0), COALESCE(SUM(chars),0) FROM documents WHERE deleted=0").fetchone()
        return dict(self.meta, documents=docs, chunks=chunks, chars=chars,
                    disk_bytes=sum(os.path.getsize(os.path.join(self.dir, f)) for f in os.listdir(self.dir)))

    def live_mask(self):
        n = self.rows()
        mask = np.zeros(n, dtype=bool)
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE deleted=0")]
        ids = [i for i in ids if i < n]
        mask[ids] = True
        return mask

    # ------------------------------------------------------------------ export
    def export_files(self):
        """(path_or_bytes, arcname) pairs for a zip: live chunks, their vectors and documents."""
        with self.lock:
            mask = self.live_mask()
            idx = np.nonzero(mask)[0]
            emb = os.path.join(self.dir, "export_embeddings.npy")
            np.save(emb, np.asarray(self.vectors()[idx]) if len(idx) else np.zeros((0, self.dim or 0), np.float16))
            lines = []
            for i, cid in enumerate(idx.tolist()):
                r = self.db.execute(
                    "SELECT c.id, c.doc_id, c.ord, c.text, c.page_start, c.page_end, c.heading, d.title, d.source "
                    "FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id=?", (cid,)).fetchone()
                lines.append(json.dumps(dict(self._chunk(r), row=i), ensure_ascii=False))
            docs = self.documents(0, 10 ** 9)["documents"]
        a = self.name
        manifest = dict(self.stats(), exported=time.time(), files={
            "chunks.jsonl": "one chunk per line; 'row' is its row in embeddings.npy",
            "embeddings.npy": "float16 matrix (rows, dim), L2-normalised: cosine similarity = dot product",
            "documents.jsonl": "one source document per line"})
        return [(json.dumps(manifest, indent=1).encode(), a + "/manifest.json"),
                (("\n".join(lines) + "\n").encode(), a + "/chunks.jsonl"),
                (emb, a + "/embeddings.npy"),
                (("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n").encode(), a + "/documents.jsonl")]

    def close(self):
        self.db.close()

    def destroy(self):
        self.close()
        shutil.rmtree(self.dir)


class Searcher:
    """Keeps each collection's live vectors in memory (GPU when possible) and runs exact search."""

    def __init__(self, device=None):
        self.device = device
        self.cache = {}

    def _matrix(self, col):
        key = col.name
        hit = self.cache.get(key)
        if hit and hit[0] == (col.version, col.rows()):
            return hit[1], hit[2]
        with col.lock:
            mask = col.live_mask()
            idx = np.nonzero(mask)[0]
            vecs = np.asarray(col.vectors()[idx], dtype=np.float16)
            stamp = (col.version, col.rows())
        mat = vecs
        if self.device and self.device.startswith("cuda"):
            import torch
            mat = torch.from_numpy(vecs).to(self.device)
        else:
            mat = vecs.astype(np.float32)
        self.cache[key] = (stamp, mat, idx)
        return mat, idx

    def search(self, col, qvec, k, allowed=None):
        mat, idx = self._matrix(col)
        if not len(idx):
            return []
        if self.device and self.device.startswith("cuda"):
            import torch
            q = torch.from_numpy(np.asarray(qvec, dtype=np.float16)).to(self.device)
            scores = (mat @ q).float().cpu().numpy()
        else:
            scores = mat @ np.asarray(qvec, dtype=np.float32)
        if allowed is not None:
            scores = np.where(np.isin(idx, allowed), scores, -np.inf)
        k = min(int(k), len(idx))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(idx[t]), float(scores[t])) for t in top if np.isfinite(scores[t])]

    def forget(self, name):
        self.cache.pop(name, None)
