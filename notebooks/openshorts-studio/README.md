# OpenShorts Studio

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/openshorts-studio/openshorts-studio.ipynb)

By Er. Gunjan Ghimire. Turns long videos into vertical shorts on a free Kaggle session with [OpenShorts](https://github.com/mutonby/openshorts), and adds Stable Diffusion b-roll to the clips. Everything runs locally; no paid APIs.

| Step | Tool | GPU |
|---|---|---|
| Transcription | faster-whisper `large-v3-turbo` | 0 |
| Scene detection, face-tracked 9:16 reframing, subtitles | OpenShorts (PySceneDetect, YOLOv8, MediaPipe, FFmpeg) | 0 |
| Picking the best moments | Ollama `qwen2.5:7b` | 1 |
| B-roll moments and prompts | Ollama | 1 |
| B-roll images | Stable Diffusion (SDXL 1.0 base), slow zoom with FFmpeg | 1 |

- Secrets: `OPENSHORTS_TUNNEL_TOKEN` (required), `OPENSHORTS_UI_PASSWORD`, `OPENSHORTS_YT_COOKIES` (optional)
- Accelerator: GPU T4 x2 (one GPU works, slower)
- Port 7860: dashboard at `/`, B-roll page at `/broll`, OpenShorts MCP server at `/mcp`
- Agents and MCP clients authenticate with `Authorization: Bearer <password>` or `X-Access-Token: <password>`
- Regenerate the notebook: `python src/build_notebook.py`

## B-roll API

```
GET  /broll/api/clips            clips OpenShorts has made
POST /broll/api/jobs             {"clip": "<job>/<file>.mp4", "count": 3, "mode": "upper" | "full", "style": "..."}
GET  /broll/api/jobs/<id>        progress, chosen moments, prompts, result URL
POST /broll/api/upload?name=x.mp4   raw mp4 body, up to 100 MB
```

`upper` puts b-roll in the top 60% of the frame so OpenShorts' burned-in captions stay visible; `full` covers the whole frame. The clip's original audio is kept.

## Limits

- Cloudflare's free plan caps one upload at 100 MB. Attach long videos as a Kaggle dataset or use a direct link.
- YouTube may block downloads from Kaggle; the `OPENSHORTS_YT_COOKIES` secret helps.
- Left out because they need paid services: AI actors, lip-sync, ElevenLabs voices, auto-posting. The Remotion renderer is not started.
- The notebook pins the OpenShorts version it was written against (`OPENSHORTS_COMMIT`). It loosens OpenShorts' PyTorch pin to keep Kaggle's own build, and makes the dashboard's Upload-Post key optional.

## Credits and licences

- [OpenShorts](https://github.com/mutonby/openshorts) by mutonby: MIT (its `cloud/` folder is under a separate licence and is not used).
- [SDXL 1.0 base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): CreativeML OpenRAIL++ (commercial use allowed). [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) is faster but non-commercial (`sai-nc-community`); switch with `SD_MODEL` / `SD_STEPS` in the settings cell.
- [Ollama](https://ollama.com), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Qwen2.5](https://huggingface.co/Qwen).
