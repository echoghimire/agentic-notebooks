# Ollama chat on Kaggle

Runs Ollama with `llama3.2` and a small streaming chat page on port 7860, published through your Cloudflare Tunnel.

- Secret: `CF_TUNNEL_TOKEN`
- Accelerator: none works (slow); a GPU makes replies much faster
- Regenerate: `python build_notebook.py`
