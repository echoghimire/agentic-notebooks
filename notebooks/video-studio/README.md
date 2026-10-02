# Video Studio

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/video-studio/video-studio.ipynb)

Any link (GitHub repo, article or web page, YouTube video, PDF) or an idea → a narrated explainer video with music, rendered as **16:9 landscape** and **9:16 reel**. A local LLM writes the script, Stable Diffusion draws the scenes, Kokoro reads the narration, MusicGen composes the music, and animated HTML templates are rendered frame by frame with headless Chromium and ffmpeg. Inspired by [nexu-io/html-video](https://github.com/nexu-io/html-video).

- Secrets: `VIDEO_TUNNEL_TOKEN` (required for a public URL), `VIDEO_UI_PASSWORD` (recommended), `GITHUB_TOKEN` (optional, lifts the 60 lookups/hour GitHub API limit)
- Accelerator: GPU T4 x2 (LLM on GPU 0; images and music on GPU 1). One GPU works.
- Options: 4 styles (`midnight`, `paper`, `neon`, `swiss`), 30 / 60 / 90 s, 11 voices (English US/UK, Spanish, French, Hindi, Italian, Portuguese), music on/off, burned-in captions on/off, landscape and/or reel, 1080p or 720p
- Text after a link becomes instructions for the scriptwriter: `https://github.com/owner/repo focus on the install steps`
- Edit the script (headings, bullets, narration, image prompts) and re-render; unchanged scenes reuse their images and narration
- Regenerate the notebook: `python src/build_notebook.py`

## API

```
POST /api/jobs {"source", "style", "length": 30|60|90, "voice", "music", "captions",
                "formats": ["landscape", "reel"], "quality": "1080p"|"720p", "review": false}
GET  /api/jobs                              list
GET  /api/jobs/<id>?wait=60                 state (queued | running | review | done | error), progress, files, storyboard
POST /api/jobs/<id>/storyboard {"storyboard", "render": true}
POST /api/jobs/<id>/render {options..., "rewrite": false}
POST /api/jobs/<id>/delete
GET  /api/jobs/<id>/files/landscape.mp4 | reel.mp4 | landscape.jpg | reel.jpg   (?download=1 to save)
```

## MCP tools

`make_video(source, style?, length?, voice?, music?, captions?, formats?, quality?, review?, wait_seconds?)`, `get_job(job_id, wait_seconds?)`, `get_storyboard(job_id)`, `update_storyboard(job_id, storyboard, render?)`, `render(job_id, rewrite?, ...options)`, `list_jobs`, `delete_job`, `list_options`.

## How it stays reliable

The local model never writes HTML. It fills a fixed JSON storyboard (layout, heading, bullets, narration, image prompt), which is validated and repaired; if the model fails, a plain storyboard is built from the source. Real numbers (GitHub stars, forks, YouTube views) are inserted from the source, never from the model. Every animation is a CSS animation that the renderer seeks frame by frame, so videos never drop frames however slow the machine is.

## Licences

SDXL 1.0: CreativeML OpenRAIL++. Kokoro-82M: Apache 2.0. Qwen2.5: Apache 2.0. **MusicGen weights: CC-BY-NC 4.0 (non-commercial)**; set `MUSIC_MODEL = ""` for commercial use. The content you turn into videos keeps its own licence.
