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
| Photos | og:image, JSON-LD, article images including lazy-loaded ones, WordPress full-size originals, captions. **Ads are filtered twice**: pictures inside ad, sponsored, related-news, share or sidebar blocks, or linking to another site, are skipped; then Gemma 3 looks at each remaining photo, drops advertisements, products, logos and off-topic pictures, and puts real news photos ahead of stock graphics. Small, odd-shaped and duplicate pictures are dropped; the rest get auto-contrast and sharpening, and each gets a description so every scene gets a fitting photo. |
| Language | Detected from the text (Nepali told apart from Hindi). The script is written in that language, checked for the right script (Devanagari for Nepali), and rejected and retried if the model drifts. You can also pick another language. |
| Script | `gemma3:12b` fills a JSON storyboard: headline, photo, bullets, stats, quote, code and outro scenes, with category and tone. News rules: factual, attributed, respectful. If the model fails, the article's own sentences are used. |
| Voice | Nepali: Indic Parler-TTS **Amrita**, or Piper (light, CPU). Hindi: Divya / Rohit. English and European languages: Kokoro. Delivery follows the tone (calm and slower for tragic news). |
| Look | `auto` picks **broadcast** for news (channel bug, red "ताजा समाचार" label, lower-thirds, ticker, wipes), **documentary** for stories (film grain, light leaks, letterbox), **midnight** for tech, **paper** for education; also neon and swiss. Kinetic word-by-word headlines. **Text never sits bare on a photo**: a dark scrim, white type with a shadow and a frosted plate behind centred headlines (light styles switch to this over photos too). A landscape photo in a reel becomes a band at the top with the text below it; a portrait photo in 16:9 goes on the right with the text on the left. Captions and lower-thirds keep clear of the Reels/Shorts buttons. |
| Music | MusicGen, matched to the tone: soft and low for sad news, ducked under the voice. Illustrations (SDXL) are only drawn for non-news topics without photos. Real events never get invented pictures. |
| Render | Every frame is seeked exactly in headless Chromium and piped to ffmpeg. Landscape and reel render in parallel; lengths are 15, 30, 60 or 90 seconds. The script gets a word budget for its language (Nepali TTS speaks about 1.5 words a second), and narration that still runs long is sped up by up to 1.25x without changing pitch. |

## Motion and sound

| Feature | How | Setting |
|---|---|---|
| **2.5D photo motion** | [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) estimates depth for each photo; a WebGL shader in the scene page moves near and far parts at different speeds while the camera pushes in or drifts. The photo is never changed, only moved, so it is safe for news. Falls back to a plain camera move if WebGL or the model is missing. | `DEPTH_PARALLAX` |
| **Subject lift-off** | [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) cuts out the main subject of headline photos; it rises out of the dimming background. Skipped when a photo has no clear subject. | `SUBJECT_CUTOUT` |
| **Map fly-in** | For news with a real place, the script names it, OpenStreetMap Nominatim finds it, and three [MapLibre](https://maplibre.org) snapshots (country → region → district, OpenFreeMap tiles) are cross-zoomed into one continuous fly-in with a pin and label. 30 s and longer. | `LOCATION_MAPS`, page: *Map of the place* |
| **Motion graphics** | Light sweeps over headline panels, drawn-on underlines, drifting dust in documentary, kinetic type, wipes. | always |
| **Transition sounds** | Whooshes on scene changes and a soft hit on the headline, synthesised in code (no sample libraries), quieter for sad news. | page: *Transition sounds* |
| **AI video clips** (opt-in) | [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) or [Wan 2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2) turn illustrations and photos into 3 s clips, played frame-exactly in the scene. **Never used on news** (it would invent movement). Slow on a T4: minutes per clip, at most `I2V_MAX_CLIPS` (3) per video. | `I2V_MODEL = "ltx"` or `"wan"`, page: *Motion → AI video clips* |

**Voices:** Nepali: Indic Parler-TTS *Amrita*, Svara-TTS *female* and *male*, Piper; Hindi: Parler *Divya* / *Rohit*, Svara *female* / *male*; English and European languages: Kokoro. With `ONLINE_VOICES = True`, Microsoft's neural voices are added: Nepali *Hemkala* and *Sagar*, Hindi *Swara* and *Madhur*, English *Aria* and *Guy*. They are very natural but online, and the narration text is sent to Microsoft. If a voice fails, the next installed voice for the language takes over.

**Music:** MusicGen (non-commercial weights) or `MUSIC_MODEL = "ace-step"` ([ACE-Step](https://github.com/ace-step/ACE-Step), Apache 2.0, fine for monetised videos; slower, in its own virtualenv).

## The page

The page has a sign-in screen (no browser pop-up), then a single box: paste a link, pick a length and formats, then **Make video**. *More options* has style, language, voice, quality, motion, map, music, transition sounds and captions. The ☰ menu holds your video history, agent setup and sign-out. While it works, a large progress bar shows the percentage, the current step and the time left. Results show a phone-framed reel next to the landscape video, the photos found on the link, and a script editor; unchanged scenes keep their photos and narration when you re-render.

**Pick each scene's picture in the script editor:** every scene has a strip with *Auto*, *No picture*, all the link's photos, and **＋ Upload**, for your own picture or a video clip (mp4 / mov / webm; the first 10 s are used and play in the scene). Uploads are added to every scene's strip. Tick *Let me edit the script first* to choose before the first render, or edit and re-render afterwards.

## API

```
POST /api/jobs {"source", "style": "auto"|"broadcast"|"documentary"|..., "length": 15|30|60|90, "language": "auto"|"ne"|"en"|...,
                "voice": "auto"|"ne_amrita"|"ne_piper"|"af_heart"|..., "music", "captions", "formats": ["landscape", "reel"],
                "quality": "1080p"|"720p", "motion": "auto"|"parallax"|"ai"|"none", "map": true, "sfx": true, "review": false}
GET  /api/jobs                              list
GET  /api/jobs/<id>?wait=60                 state, progress, files, storyboard, photos
POST /api/jobs/<id>/storyboard {"storyboard", "render": true}
POST /api/jobs/<id>/render {options..., "rewrite": false}
POST /api/jobs/<id>/assets?name=clip.mp4      raw picture or video as the body (up to 200 MB) -> {"index", "kind", "photos"}
POST /api/jobs/<id>/delete
GET  /api/jobs/<id>/files/landscape.mp4 | reel.mp4 | landscape.jpg | reel.jpg   (?download=1 to save)
```

## MCP tools

`make_video(source, style?, length?, language?, voice?, music?, captions?, formats?, quality?, motion?, map?, sfx?, review?, wait_seconds?)`, `get_job(job_id, wait_seconds?)`, `get_storyboard(job_id)`, `update_storyboard(job_id, storyboard, render?)` (scene `photo`: index, -1 auto, -2 none), `add_asset(job_id, url)`, `render(job_id, rewrite?, ...options)`, `list_jobs`, `delete_job`, `list_options`.

## Licences and responsibility

Gemma 3: Gemma Terms of Use. Indic Parler-TTS, Svara-TTS, Kokoro, Depth Anything V2 Small, ACE-Step, Wan 2.2: Apache 2.0. Piper, BiRefNet: MIT. MapLibre: BSD-3; map data © OpenStreetMap contributors (ODbL), credited on screen. LTX-Video: Lightricks open-weights licence (check before commercial use). SDXL: CreativeML OpenRAIL++. **MusicGen weights: CC-BY-NC 4.0 (non-commercial)**; use `MUSIC_MODEL = "ace-step"` or `""` for monetised videos. Online voices: Microsoft service, not open source. **Text and photos from a link belong to their publisher.** The video credits the site, but make sure you are allowed to reuse them.
