# Notes for AI agents

This repo holds Kaggle notebooks that run a web app on a free Kaggle session and expose it through the user's own Cloudflare Tunnel. You will usually reach it through the GitHub MCP server (read files, open issues and pull requests). Once a user has a notebook running, several of them are also MCP servers you can drive directly (see below).

## Layout

```
notebooks/<name>/
  <name>.ipynb          # the file users upload to Kaggle (generated)
  build_notebook.py     # generates the .ipynb (ollama-chat keeps everything in here)
  src/                  # app sources + build_notebook.py, when the app is more than a cell or two
  README.md             # secrets, accelerator, API and MCP tools
shared/
  studio_http.py        # web kit for the newer apps: password, /health, /mcp, REST tool mirror, proxy, process helpers
  kit.css, kit.js       # shared page styles and helpers
  nbkit.py              # build-time helpers for build_notebook.py
```

Edit sources and `build_notebook.py`, then regenerate the `.ipynb` (`python src/build_notebook.py` or `python build_notebook.py` from the notebook folder). Never hand-edit a generated `.ipynb`. CI (`.github/workflows/check-notebooks.yml`) rebuilds every notebook and fails if the committed `.ipynb` differs.

`shared/` is pasted into each notebook at build time (`nbkit.Notebook.kit_files()`), so every `.ipynb` stays self-contained. After changing anything in `shared/`, rebuild every notebook that uses it.

## Helping a user run a notebook

1. Read the notebook's first markdown cell: it lists the Kaggle secrets and accelerator it needs.
2. Give them the raw `.ipynb` link, or fetch it for them, and walk them through the README quick start.
3. You cannot run Kaggle from GitHub. If the user has the Kaggle CLI configured, `kaggle kernels push` can upload and run a notebook, but secrets and the accelerator still have to be set in the Kaggle UI.
4. Never ask the user to paste tokens into the notebook, an issue or a pull request. Tokens belong in Kaggle secrets only.
5. A notebook without its tunnel secret still runs; its tunnel cell prints how to add the secret. Logs are in `/kaggle/working/logs/`.

## MCP servers and APIs

`comfyui-flux`, `whisper-diarization-studio`, `unsloth-finetuning-lab`, `rag-ingest-pipeline` and `video-studio` serve an MCP endpoint at `https://<their hostname>/mcp` (Streamable HTTP, JSON responses, no sessions). Authenticate with the notebook's UI password:

```
claude mcp add --transport http <name> https://<hostname>/mcp --header "Authorization: Bearer <password>"
```

- `X-Access-Token: <password>` and HTTP Basic work too. People sign in on the app's own page (session cookie); agents keep using the header. Ask the user for the password; do not look for it in notebooks or logs.
- Every tool is also plain REST: `GET /mcp/tools` lists them with JSON schemas, and `POST /mcp/tools/<name>` takes the arguments as a JSON object (HTTP 400 with `{"error": ...}` when the tool fails).
- Cloudflare ends a request after 100 s, so long work runs as a job: tools return a job id or state at once, and status tools accept `wait_seconds` (at most 85) to wait for it. Poll until the state is `done` or `error`.
- Binary uploads (audio, documents) go through each app's REST upload endpoint with the raw file as the body; MCP tools take URLs or paths under `/kaggle/input` instead.
- Download URLs in tool results are relative to the hostname and need the same password.

| Notebook | Tools (required inputs first) |
|---|---|
| comfyui-flux | `generate_image(prompt, width?, height?, steps?, seed?, guidance?, batch_size?, wait_seconds?, return_image?)` returns JPEG previews inline when done; `get_job(job_id, wait_seconds?, return_image?)`; `run_workflow(workflow, wait_seconds?)` takes ComfyUI API-format JSON; `list_jobs(limit?)`; `cancel_job(job_id)`; `list_models()`; `server_status()` |
| whisper-diarization-studio | `transcribe(url \| path, language?, task?, diarize?, num_speakers?, min_speakers?, max_speakers?, summarize?, initial_prompt?, title?)`; `get_job(job_id, wait_seconds?)`; `get_transcript(job_id, format?=md\|txt\|srt\|vtt\|json, offset?, max_chars?)`; `list_jobs(limit?)`; `rename_speakers(job_id, names)`; `summarize(job_id, instructions?, wait_seconds?)`; `delete_job(job_id)`; `list_input_files()`; `server_status()` |
| unsloth-finetuning-lab | `add_examples(records, mode?)`; `import_hf_dataset(name, split?, config?, max_rows?, mode?)`; `dataset_info()`; `list_base_models()`; `start_training(name?, base_model?, epochs?, max_steps?, learning_rate?, lora_r?, lora_alpha?, max_seq_length?, batch_size?, grad_accum?, val_frac?)`; `training_status(wait_seconds?)`; `stop_training(force?)`; `list_runs()`; `get_run(name)`; `chat(model, messages, max_new_tokens?, temperature?)` (model is a Hub id or `run:<name>`; returns `loading: true` until the model is loaded); `export_run(name, format)`; `export_status(name)`; `push_to_hub(name, repo_id, what?, private?)`; `delete_run(name)` |
| rag-ingest-pipeline | `ingest(collection, urls?, paths?, texts?, wait_seconds?)` (creates the collection if needed); `job_status(job_id, wait_seconds?)`; `search(collection, query, top_k?, rerank?, document_ids?)`; `get_chunks(collection, chunk_ids, context?)`; `list_documents(collection, q?, offset?, limit?)`; `delete_document(collection, document_id)`; `create_collection(name, chunk_size?, chunk_overlap?)`; `list_collections()`; `export_collection(collection)`; `list_input_files()`; `server_status()` |
| video-studio | `make_video(source, style?, length?, language?, voice?, music?, captions?, formats?, quality?, review?, wait_seconds?)` (source is any link or an idea; length 15/30/60/90; style `auto` picks broadcast for news; language `auto` follows the link, e.g. Nepali; returns a job; rendering takes minutes, so poll); `get_job(job_id, wait_seconds?)` (files.landscape.mp4 / files.reel.mp4 when done); `get_storyboard(job_id)`; `update_storyboard(job_id, storyboard, render?)`; `render(job_id, rewrite?, ...options)`; `list_jobs(limit?)`; `delete_job(job_id)`; `list_options()` |

Each notebook's README lists its REST endpoints. openshorts-studio has its own MCP server (OpenShorts') at `/mcp`, with the same Bearer authentication.

## Rules for adding a notebook

- **No hardcoded domains, hostnames, tokens or passwords.** Read secrets with `kaggle_secrets.UserSecretsClient` (or `studio_http.kaggle_secret`) and give the secret a notebook-specific name (`<NAME>_TUNNEL_TOKEN`).
- **A missing tunnel secret must not crash the notebook.** Use `studio_http.start_tunnel()`, which prints setup steps and keeps the app running locally.
- **The UI listens on port 7860 on both IPv4 and IPv6.** cloudflared may resolve `localhost` to `::1`. (`studio_http.App.start()` does both.)
- **Nothing blocks except the last cell.** Servers run in background threads or subprocesses, logs go to files (not unread pipes), and the final cell is a keep-alive status loop (`studio_http.keep_alive()`).
- **Pass the tunnel token to cloudflared through the `TUNNEL_TOKEN` environment variable**, never on the command line or in printed output. Pass other secrets to app processes through environment variables too.
- **Re-running a cell must be safe.** Stop the old server or process before starting a new one (`studio_http.spawn()` does this with pid files, even after a kernel restart). A cell should not depend on variables that only an install or download cell sets.
- **Prefer the Python standard library for UIs.** Heavy UI frameworks slow the install on Kaggle; this repo exists partly because of that. New apps should build on `shared/studio_http.py`.
- **Expose `GET /health` returning `{"ok": true}`** so the keep-alive cell and future Docker health checks can use it. (openshorts-studio uses `/healthz`, because `/health` belongs to the OpenShorts backend it proxies.)
- **Put a password on anything that holds user data.** `studio_http.App(password=...)` gives a sign-in page with a session cookie for people and Bearer auth for agents; never rely on the browser's Basic-auth pop-up.
- **Keep long operations under 100 s per request** (Cloudflare's limit): run them as jobs and let clients poll.
- **Do not touch Kaggle's own PyTorch / numpy** unless the library requires it: install with the `pip_install` helper from `shared/nbkit.py`, which pins them.
- **Add a row to the README table**, listing required and optional secrets, and document MCP tools in the table above.

## Later: Docker

The same structure is meant to move into Docker for the notebooks that prove useful: the app sources plus a `Dockerfile` that runs the server on 7860 and a cloudflared sidecar reading `TUNNEL_TOKEN` from the environment. `studio_http.kaggle_secret()` already falls back to environment variables outside Kaggle.
