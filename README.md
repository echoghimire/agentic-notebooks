# agentic-notebooks

Self-hosted AI apps that run free on Kaggle notebooks and open in your browser through your own Cloudflare Tunnel. Agent-ready: [`AGENTS.md`](AGENTS.md) tells coding agents how to use and extend it.

Each notebook is self-contained: it installs what it needs, starts its app in the background on port 7860, connects the tunnel, then keeps the session alive.

By Er. Gunjan Ghimire.

| Notebook | What it runs | Accelerator |
|---|---|---|
| [`notebooks/ollama-chat`](notebooks/ollama-chat) | Ollama (`llama3.2`) with a lightweight streaming chat UI | None (CPU) or GPU |
| [`notebooks/laya-studio`](notebooks/laya-studio) | **Laya Studio for Data**: label bills, receipts and fintech text; fine-tune [Laya](https://github.com/NandhaKishorM/laya) on it; test the result | GPU T4 x2 |

## Quick start

1. **Create a Cloudflare Tunnel** (Cloudflare dashboard → Zero Trust → Networks → Tunnels). Add a public hostname on your own domain pointing to `http://localhost:7860`. Copy the tunnel token.
   Use a separate tunnel for each notebook you run at the same time; two notebooks on one tunnel will split traffic between them.
2. **Upload the notebook** to Kaggle (Create → New Notebook → File → Import Notebook).
3. **Add the tunnel token as a Kaggle secret** (Add-ons → Secrets) under the name the notebook's first cell asks for, and attach it to the notebook.
4. **Settings:** Internet on; pick the accelerator from the table above.
5. **Run All**, then open your hostname.

No tokens or domains are stored in the notebooks. Everything comes from Kaggle secrets.

## Secrets per notebook

| Notebook | Required | Optional |
|---|---|---|
| ollama-chat | `CF_TUNNEL_TOKEN` | – |
| laya-studio | `LAYA_TUNNEL_TOKEN` | `LAYA_UI_PASSWORD` (UI login; any username), `HF_TOKEN` (push trained models to Hugging Face) |

## Limits to know

- Kaggle sessions stop after about 12 hours, and GPU time is capped per week. These are for building, testing and demos, not always-on hosting.
- `/kaggle/working` is wiped when a session ends. Download results or push them to Hugging Face.
- Anything behind the tunnel is on the public internet. Put a password on it (Laya Studio has one built in) or add Cloudflare Access.

## For agents

See [`AGENTS.md`](AGENTS.md). It explains the layout, how to hand a notebook to a user, and the rules for adding a new one.

## Credits

- Laya model and training recipe: [Convai Innovations](https://github.com/NandhaKishorM/laya), Apache 2.0.
- Ollama: [ollama.com](https://ollama.com).
