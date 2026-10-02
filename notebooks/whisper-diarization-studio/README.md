# Whisper Diarization Studio

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/whisper-diarization-studio/whisper-diarization-studio.ipynb)

The meeting & audio OS: transcripts with who-spoke-when, meeting summaries with decisions and action items, and subtitles, all local on a free Kaggle GPU.

- Secrets: `WHISPER_TUNNEL_TOKEN` (required for a public URL), `WHISPER_UI_PASSWORD` (recommended), `HF_TOKEN` (speaker labels; accept the terms of `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` first)
- Accelerator: GPU T4 x2 (Whisper on GPU 0; pyannote and the Ollama summary model on GPU 1). One GPU works.
- Models: faster-whisper `large-v3-turbo`, pyannote `speaker-diarization-3.1`, Ollama `qwen2.5:7b` (set `SUMMARY_MODEL = ""` to skip)
- Inputs: upload (≤100 MB via Cloudflare), any yt-dlp URL, or files attached as a Kaggle dataset
- Outputs: `md` (summary, talk time, transcript), `txt`, `srt`, `vtt`, `json` (word timings)
- Regenerate the notebook: `python src/build_notebook.py`

## API

```
POST /api/jobs                 {"url" | "path", "language", "task": "transcribe"|"translate", "diarize", "num_speakers",
                                "min_speakers", "max_speakers", "summarize", "initial_prompt", "title"}
POST /api/jobs/upload?name=meeting.mp3&diarize=true&...   raw file body
GET  /api/jobs                 list
GET  /api/jobs/<id>?wait=60    status, utterances, speakers, summary
GET  /api/jobs/<id>/download/<md|txt|srt|vtt|json>
GET  /api/jobs/<id>/audio      mp3 for playback (byte ranges)
POST /api/jobs/<id>/speakers   {"names": {"SPEAKER_00": "Asha"}}
POST /api/jobs/<id>/summarize  {"instructions": "..."}
POST /api/jobs/<id>/retry | /delete
GET  /api/sources              audio/video under /kaggle/input
```

## MCP tools

`transcribe(url | path, language?, task?, diarize?, num_speakers?, min_speakers?, max_speakers?, summarize?, initial_prompt?, title?)`, `get_job(job_id, wait_seconds?)`, `get_transcript(job_id, format?, offset?, max_chars?)`, `list_jobs`, `rename_speakers(job_id, names)`, `summarize(job_id, instructions?, wait_seconds?)`, `delete_job`, `list_input_files`, `server_status`.

## Licences

faster-whisper: MIT; Whisper weights: MIT. pyannote.audio: MIT (the diarization models are gated: accept their terms on Hugging Face). Qwen2.5-7B: Apache 2.0.
