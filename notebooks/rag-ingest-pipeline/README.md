# RAG Ingest Pipeline

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/rag-ingest-pipeline/rag-ingest-pipeline.ipynb)

A heavy document vectorizer: PDFs, Word files, web pages, Markdown, code and whole Kaggle datasets → chunks with page and heading metadata → GPU embeddings → semantic search (with reranking) → an export you can load into any vector database.

- Secrets: `RAG_TUNNEL_TOKEN` (required for a public URL), `RAG_UI_PASSWORD` (recommended)
- Accelerator: GPU T4 x2 (ingest embeds on GPU 0, search and reranking on GPU 1). One GPU or CPU works, slower.
- Models: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) embeddings (MIT, 100+ languages, 1024-d), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) reranker (Apache 2.0); both configurable
- Storage: per collection, SQLite (text + metadata) and a float16 vector file; exact search; no vector-database dependency
- Export: `chunks.jsonl`, `embeddings.npy` (float16, L2-normalised), `documents.jsonl`, `manifest.json`
- Scanned PDFs need OCR first; the page reports which pages had no text layer
- Regenerate the notebook: `python src/build_notebook.py`

## API

```
GET  /api/collections                             POST /api/collections {"name", "chunk_size", "chunk_overlap"}
POST /api/collections/<c>/ingest {"urls": [], "paths": [], "texts": [{"title", "text", "source"}]}  -> job
POST /api/collections/<c>/upload?name=report.pdf  raw file body -> job
GET  /api/jobs/<id>?wait=60                        progress and per-file results
POST /api/collections/<c>/search {"query", "top_k", "rerank", "document_ids"}
GET  /api/collections/<c>/documents?q=&offset=&limit=
GET  /api/collections/<c>/documents/<id>           with its chunks
POST /api/collections/<c>/documents/<id>/delete    POST /api/collections/<c>/delete
GET  /api/collections/<c>/export                   zip
GET  /api/sources                                  folders under /kaggle/input with supported files
```

## MCP tools

`list_collections`, `create_collection(name, chunk_size?, chunk_overlap?)`, `ingest(collection, urls?, paths?, texts?, wait_seconds?)`, `job_status(job_id, wait_seconds?)`, `search(collection, query, top_k?, rerank?, document_ids?)`, `get_chunks(collection, chunk_ids, context?)`, `list_documents(collection, q?, offset?, limit?)`, `delete_document(collection, document_id)`, `export_collection(collection)`, `list_input_files`, `server_status`.

## Loading an export elsewhere

```python
import json, numpy as np
vecs = np.load("kb/embeddings.npy").astype("float32")        # (rows, 1024)
chunks = [json.loads(l) for l in open("kb/chunks.jsonl")]    # chunks[i]["row"] == i
# e.g. Qdrant: client.upsert("kb", points=[PointStruct(id=c["chunk_id"], vector=vecs[c["row"]].tolist(), payload=c) for c in chunks])
```
