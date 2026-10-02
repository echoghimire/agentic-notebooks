# ComfyUI + FLUX

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/comfyui-flux/comfyui-flux.ipynb)

The generative art & media engine: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with FLUX.1 on a free Kaggle GPU, behind a password on your own hostname.

- Secrets: `COMFYUI_TUNNEL_TOKEN` (required for a public URL), `COMFYUI_UI_PASSWORD` (recommended), `HF_TOKEN` (only for FLUX.1-dev)
- Accelerator: GPU T4 x2 or P100 (ComfyUI uses one GPU)
- First run: ~10 minutes (install + 17 GB checkpoint); then ~30–60 s per 1024² FLUX.1-schnell image on a T4
- Regenerate the notebook: `python src/build_notebook.py`

| Path | What |
|---|---|
| `/` | Full ComfyUI node editor (workflows, LoRAs, inpainting, upscaling) |
| `/flux` | One-box prompt → image page |
| `/flux/api/*` | JSON API |
| `/mcp` | MCP server; same tools as REST at `POST /mcp/tools/<name>` |

## API

```
GET  /flux/api/info                      variant, queue, GPU memory
GET  /flux/api/models                    checkpoints and LoRAs
POST /flux/api/generate                  {"prompt", "width", "height", "steps", "seed", "guidance", "batch_size"} -> {"job_id"}
POST /flux/api/workflow                  {"workflow": <ComfyUI API-format JSON>} -> {"job_id"}
GET  /flux/api/jobs                      recent jobs
GET  /flux/api/jobs/<id>?wait=60         state: queued | running | done | error; images[].url once done
POST /flux/api/jobs/<id>/cancel
GET  /view?filename=...&subfolder=...&type=output   the image (ComfyUI's own endpoint, behind the password)
```

## MCP tools

`generate_image(prompt, width?, height?, steps?, seed?, guidance?, batch_size?, wait_seconds?, return_image?)` waits up to `wait_seconds` (max 85) and returns JPEG previews inline; `get_job(job_id)`, `run_workflow(workflow)`, `list_jobs`, `cancel_job`, `list_models`, `server_status`.

## Licences

FLUX.1-schnell: Apache 2.0. FLUX.1-dev: FLUX.1 [dev] Non-Commercial License (set `FLUX_VARIANT = "dev"` and an `HF_TOKEN` that accepted it). ComfyUI: GPL-3.0. Weights come from the [Comfy-Org](https://huggingface.co/Comfy-Org) fp8 single-file checkpoints.
