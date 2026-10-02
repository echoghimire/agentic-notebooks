# agentic-notebooks

Self-hosted AI apps that run free on Kaggle notebooks and open in your browser through your own Cloudflare Tunnel. Agent-ready: [`AGENTS.md`](AGENTS.md) tells coding agents how to use and extend it.

Each notebook is self-contained: it installs what it needs, starts its app in the background on port 7860, connects the tunnel, then keeps the session alive. If the tunnel secret is missing, the app still runs inside the session and the notebook tells you how to publish it.

The newer notebooks (ComfyUI, Whisper, Unsloth, RAG) also expose an **MCP server at `/mcp`**, so coding agents such as Claude Code can drive them:

```
claude mcp add --transport http flux https://flux.example.com/mcp --header "Authorization: Bearer <your UI password>"
```

By Er. Gunjan Ghimire.

| Notebook | What it runs | Accelerator | Run |
|---|---|---|---|
| [`notebooks/ollama-chat`](notebooks/ollama-chat) | Ollama (`llama3.2`) with a lightweight streaming chat UI | None (CPU) or GPU | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/ollama-chat/ollama-chat.ipynb) |
| [`notebooks/laya-studio`](notebooks/laya-studio) | **Laya Studio for Data**: label bills, receipts and fintech text; fine-tune [Laya](https://github.com/NandhaKishorM/laya) on it; test the result | GPU T4 x2 | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/laya-studio/laya-studio.ipynb) |
| [`notebooks/openshorts-studio`](notebooks/openshorts-studio) | **OpenShorts Studio**: long video → vertical shorts ([OpenShorts](https://github.com/mutonby/openshorts) + faster-whisper + Ollama), plus Stable Diffusion b-roll; MCP server for agents | GPU T4 x2 | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/openshorts-studio/openshorts-studio.ipynb) |
| [`notebooks/comfyui-flux`](notebooks/comfyui-flux) | **ComfyUI + FLUX**: the full ComfyUI editor with FLUX.1-schnell, a one-box generator page, and an MCP server that returns images to agents | GPU T4 x2 or P100 | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/comfyui-flux/comfyui-flux.ipynb) |
| [`notebooks/whisper-diarization-studio`](notebooks/whisper-diarization-studio) | **Whisper Diarization Studio**: transcripts with who-spoke-when (faster-whisper + pyannote), meeting summaries and action items (Ollama), SRT/VTT; MCP server | GPU T4 x2 | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/whisper-diarization-studio/whisper-diarization-studio.ipynb) |
| [`notebooks/unsloth-finetuning-lab`](notebooks/unsloth-finetuning-lab) | **Unsloth Fine-tuning Lab**: dataset → 4-bit LoRA fine-tune → chat → GGUF / merged export → Hugging Face; MCP server | GPU T4 x2 | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/unsloth-finetuning-lab/unsloth-finetuning-lab.ipynb) |
| [`notebooks/rag-ingest-pipeline`](notebooks/rag-ingest-pipeline) | **RAG Ingest Pipeline**: PDFs, DOCX, web pages and whole datasets → chunks → bge-m3 embeddings → search with reranking → export for any vector DB; MCP server | GPU T4 x2 | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/rag-ingest-pipeline/rag-ingest-pipeline.ipynb) |

## Quick start

1. **Create a Cloudflare Tunnel** (Cloudflare dashboard → Zero Trust → Networks → Tunnels). Add a public hostname on your own domain pointing to `http://localhost:7860`. Copy the tunnel token.
   Use a separate tunnel for each notebook you run at the same time; two notebooks on one tunnel will split traffic between them.
2. **Open the notebook in Kaggle** with its "Open in Kaggle" button above (or Create → New Notebook → File → Import Notebook).
3. **Add the tunnel token as a Kaggle secret** (Add-ons → Secrets) under the name the notebook's first cell asks for, and attach it to the notebook.
4. **Settings:** Internet on; pick the accelerator from the table above.
5. **Run All**, then open your hostname.

No tokens or domains are stored in the notebooks. Everything comes from Kaggle secrets.

## Secrets per notebook

| Notebook | Required | Optional |
|---|---|---|
| ollama-chat | `OLLAMA_TUNNEL_TOKEN` (old name `CF_TUNNEL_TOKEN` still works) | `OLLAMA_UI_PASSWORD` (UI login; any username) |
| laya-studio | `LAYA_TUNNEL_TOKEN` | `LAYA_UI_PASSWORD` (UI login; any username), `HF_TOKEN` (push trained models to Hugging Face) |
| openshorts-studio | `OPENSHORTS_TUNNEL_TOKEN` | `OPENSHORTS_UI_PASSWORD` (UI and MCP login), `OPENSHORTS_YT_COOKIES` (YouTube cookies if downloads are blocked) |
| comfyui-flux | `COMFYUI_TUNNEL_TOKEN` | `COMFYUI_UI_PASSWORD` (UI and MCP login), `HF_TOKEN` (only for FLUX.1-dev) |
| whisper-diarization-studio | `WHISPER_TUNNEL_TOKEN` | `WHISPER_UI_PASSWORD` (UI and MCP login), `HF_TOKEN` (speaker labels; accept the pyannote terms) |
| unsloth-finetuning-lab | `UNSLOTH_TUNNEL_TOKEN` | `UNSLOTH_UI_PASSWORD` (UI and MCP login), `HF_TOKEN` (gated base models, Hub pushes) |
| rag-ingest-pipeline | `RAG_TUNNEL_TOKEN` | `RAG_UI_PASSWORD` (UI and MCP login) |

## Limits to know

- Kaggle sessions stop after about 12 hours, and GPU time is capped per week. These are for building, testing and demos, not always-on hosting.
- `/kaggle/working` is wiped when a session ends. Download results or push them to Hugging Face.
- Anything behind the tunnel is on the public internet. Put a password on it (every notebook here supports one) or add Cloudflare Access.

## For agents

See [`AGENTS.md`](AGENTS.md). It explains the layout, how to hand a notebook to a user, and the rules for adding a new one.

## Credits

- Laya model and training recipe: [Convai Innovations](https://github.com/NandhaKishorM/laya), Apache 2.0.
- Ollama: [ollama.com](https://ollama.com).
- OpenShorts: [mutonby/openshorts](https://github.com/mutonby/openshorts), MIT.
- SDXL 1.0 base: [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), CreativeML OpenRAIL++.
- ComfyUI: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), GPL-3.0. FLUX.1-schnell: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-schnell), Apache 2.0 (FLUX.1-dev is non-commercial).
- faster-whisper: [SYSTRAN](https://github.com/SYSTRAN/faster-whisper), MIT. pyannote.audio: [pyannote](https://github.com/pyannote/pyannote-audio), MIT (models gated).
- Unsloth: [unslothai/unsloth](https://github.com/unslothai/unsloth), Apache 2.0.
- bge-m3 and bge-reranker-v2-m3: [BAAI](https://huggingface.co/BAAI), MIT / Apache 2.0.
