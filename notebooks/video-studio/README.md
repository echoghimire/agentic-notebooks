# Video Studio

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/video-studio/video-studio.ipynb)

Paste a link and get a narrated video with music as **16:9 landscape** and **9:16 reel**. Links can be news articles (Nepali or English), any web page, GitHub repos, YouTube videos or PDFs; you can also describe an idea. It uses the link's **own photos**, speaks the link's **language**, and picks a look from the content. Inspired by [nexu-io/html-video](https://github.com/nexu-io/html-video) and [HyperFrames](https://github.com/heygen-com/hyperframes).

- Secrets: `VIDEO_TUNNEL_TOKEN` (required for a public URL), `VIDEO_UI_PASSWORD` (sign-in password), `GITHUB_TOKEN` (optional)
- Accelerator: GPU T4 x2 (scriptwriter on GPU 0; voices, images and music on GPU 1). One GPU works.
- Regenerate the notebook: `python src/build_notebook.py`

## What it does with a link

| Step | How |
|---|---|
| Read | [trafilatura](https://github.com/adbar/trafilatura) extracts the article, date and author; a headless browser retries sites that block downloads or need JavaScript. GitHub API, yt-dlp (subtitles, no download) and pymupdf for the other kinds. |
| Photos | og:image, JSON-LD, article images including lazy-loaded ones, WordPress full-size originals, captions. Small, odd-shaped and duplicate pictures are dropped; the rest get auto-contrast and sharpening, and Gemma 3 describes each one so every scene gets a fitting photo. |
| Language | Detected from the text (Nepali told apart from Hindi). The script is written in that language, checked for the right script (Devanagari for Nepali), and rejected and retried if the model drifts. You can also pick another language. |
| Script | `gemma3:12b` fills a JSON storyboard: headline, photo, bullets, stats, quote, code and outro scenes, with category and tone. News rules: factual, attributed, respectful. If the model fails, the article's own sentences are used. |
| Voice | Nepali: Indic Parler-TTS **Amrita**, or Piper (light, CPU). Hindi: Divya / Rohit. English and European languages: Kokoro. Delivery follows the tone (calm and slower for tragic news). |
| Look | `auto` picks **broadcast** for news (channel bug, red "ताजा समाचार" label, lower-thirds, ticker, wipes), **documentary** for stories (film grain, light leaks, letterbox), **midnight** for tech, **paper** for education; also neon and swiss. Kinetic word-by-word headlines; portrait photos sit over a blurred fill in 16:9 and landscape ones in 9:16. Captions and lower-thirds keep clear of the Reels/Shorts buttons. |
| Music | MusicGen, matched to the tone: soft and low for sad news, ducked under the voice. Illustrations (SDXL) are only drawn for non-news topics without photos. Real events never get invented pictures. |
| Render | Every frame is seeked exactly in headless Chromium and piped to ffmpeg. Landscape and reel render in parallel; lengths are 15, 30, 60 or 90 seconds. |

## The page

The page has a sign-in screen (no browser pop-up), then a single box: paste a link, pick a length and formats, then **Make video**. The ☰ menu holds your video history, agent setup and sign-out. Results show a phone-framed reel next to the landscape video, the photos found on the link, and a script editor; unchanged scenes keep their photos and narration when you re-render.

## API

```
POST /api/jobs {"source", "style": "auto"|"broadcast"|"documentary"|..., "length": 15|30|60|90, "language": "auto"|"ne"|"en"|...,
                "voice": "auto"|"ne_amrita"|"ne_piper"|"af_heart"|..., "music", "captions", "formats": ["landscape", "reel"],
                "quality": "1080p"|"720p", "review": false}
GET  /api/jobs                              list
GET  /api/jobs/<id>?wait=60                 state, progress, files, storyboard, photos
POST /api/jobs/<id>/storyboard {"storyboard", "render": true}
POST /api/jobs/<id>/render {options..., "rewrite": false}
POST /api/jobs/<id>/delete
GET  /api/jobs/<id>/files/landscape.mp4 | reel.mp4 | landscape.jpg | reel.jpg   (?download=1 to save)
```

## MCP tools

`make_video(source, style?, length?, language?, voice?, music?, captions?, formats?, quality?, review?, wait_seconds?)`, `get_job(job_id, wait_seconds?)`, `get_storyboard(job_id)`, `update_storyboard(job_id, storyboard, render?)`, `render(job_id, rewrite?, ...options)`, `list_jobs`, `delete_job`, `list_options`.

## Licences and responsibility

Gemma 3: Gemma Terms of Use. Indic Parler-TTS, Kokoro: Apache 2.0. Piper: MIT. SDXL: CreativeML OpenRAIL++. **MusicGen weights: CC-BY-NC 4.0 (non-commercial)**; set `MUSIC_MODEL = ""` for monetised videos. **Text and photos from a link belong to their publisher.** The video credits the site, but make sure you are allowed to reuse them.
