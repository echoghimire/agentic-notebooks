# Notes for AI agents

This repo holds Kaggle notebooks that run a web app on a free Kaggle session and expose it through the user's own Cloudflare Tunnel. You will usually reach it through the GitHub MCP server (read files, open issues and pull requests).

## Layout

```
notebooks/<name>/
  <name>.ipynb          # the file users upload to Kaggle (generated)
  build_notebook.py     # generates the .ipynb (ollama-chat keeps everything in here)
  src/                  # app sources + build_notebook.py, when the app is more than a cell or two
```

Edit sources and `build_notebook.py`, then regenerate the `.ipynb` (`python src/build_notebook.py` or `python build_notebook.py` from the notebook folder). Never hand-edit a generated `.ipynb`.

## Helping a user run a notebook

1. Read the notebook's first markdown cell: it lists the Kaggle secrets and accelerator it needs.
2. Give them the raw `.ipynb` link, or fetch it for them, and walk them through the README quick start.
3. You cannot run Kaggle from GitHub. If the user has the Kaggle CLI configured, `kaggle kernels push` can upload and run a notebook, but secrets and the accelerator still have to be set in the Kaggle UI.
4. Never ask the user to paste tokens into the notebook, an issue or a pull request. Tokens belong in Kaggle secrets only.

## Rules for adding a notebook

- **No hardcoded domains, hostnames, tokens or passwords.** Read secrets with `kaggle_secrets.UserSecretsClient` and give the secret a notebook-specific name (`<NAME>_TUNNEL_TOKEN`).
- **The UI listens on port 7860 on both IPv4 and IPv6.** cloudflared may resolve `localhost` to `::1`.
- **Nothing blocks except the last cell.** Servers run in background threads or subprocesses, logs go to files (not unread pipes), and the final cell is a keep-alive status loop.
- **Pass the tunnel token to cloudflared through the `TUNNEL_TOKEN` environment variable**, never on the command line or in printed output.
- **Re-running a cell must be safe.** Stop the old server or process before starting a new one.
- **Prefer the Python standard library for UIs.** Heavy UI frameworks slow the install on Kaggle; this repo exists partly because of that.
- **Expose `GET /health` returning `{"ok": true}`** so the keep-alive cell and future Docker health checks can use it.
- **Put a password on anything that holds user data.**
- **Add a row to the README table**, listing required and optional secrets.

## Later: Docker

The same structure is meant to move into Docker for the notebooks that prove useful: the app sources plus a `Dockerfile` that runs the server on 7860 and a cloudflared sidecar reading `TUNNEL_TOKEN` from the environment.
