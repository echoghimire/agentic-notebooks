# Ollama chat on Kaggle

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/ollama-chat/ollama-chat.ipynb)

Runs Ollama with `llama3.2` and a small streaming chat page on port 7860, published through your Cloudflare Tunnel.

- Secrets: `OLLAMA_TUNNEL_TOKEN` (required; the old name `CF_TUNNEL_TOKEN` still works), `OLLAMA_UI_PASSWORD` (optional login, any username)
- Accelerator: none works (slow); a GPU makes replies much faster
- Regenerate: `python build_notebook.py`
