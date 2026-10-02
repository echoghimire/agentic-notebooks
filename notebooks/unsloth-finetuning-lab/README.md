# Unsloth Fine-tuning Lab

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/unsloth-finetuning-lab/unsloth-finetuning-lab.ipynb)

The minimalist LLM factory: dataset → 4-bit LoRA fine-tune with [Unsloth](https://github.com/unslothai/unsloth) → chat with the result → export (adapter, merged 16-bit, GGUF) → download or push to the Hub.

- Secrets: `UNSLOTH_TUNNEL_TOKEN` (required for a public URL), `UNSLOTH_UI_PASSWORD` (recommended), `HF_TOKEN` (gated bases such as Llama, and Hub pushes)
- Accelerator: GPU T4 x2 (training on GPU 0; chat and exports on GPU 1). One GPU works; chat and exports then wait for training.
- Dataset formats: OpenAI `messages`, ShareGPT `conversations`, Alpaca `instruction/input/output`, `prompt/completion`, `question/answer`, raw `text`; JSONL, JSON or CSV; or any Hub dataset with those columns
- Training, chat and exports run in their own processes and survive a re-run of the server cell
- Regenerate the notebook: `python src/build_notebook.py`

## API

```
GET  /api/info
GET  /api/dataset?offset=&limit=&q=        POST /api/dataset/upload?format=jsonl|json|csv|auto&mode=append|replace (raw body)
POST /api/dataset/add {"records", "mode"}  POST /api/dataset/import_hf {"name", "split", "config", "max_rows", "mode"}
POST /api/dataset/delete {"indices"}       POST /api/dataset/clear      GET /api/dataset/download
POST /api/train {"name", "base_model", "epochs", "max_steps", "learning_rate", "lora_r", "lora_alpha",
                 "max_seq_length", "batch_size", "grad_accum", "val_frac"}
GET  /api/train/status                     POST /api/train/stop {"force": false}
GET  /api/runs                             GET /api/runs/<name>
POST /api/runs/<name>/export {"format": "merged_16bit"|"gguf_q4_k_m"|"gguf_q8_0"|"gguf_f16"}
GET  /api/runs/<name>/download/<adapter|format>   zip
POST /api/runs/<name>/push {"repo_id", "what", "private"}
POST /api/runs/<name>/delete
POST /api/chat {"model": "<hub id>"|"run:<name>", "messages", "max_new_tokens", "temperature"}   -> {"loading": true} while it loads
```

## MCP tools

`list_base_models`, `dataset_info`, `add_examples(records, mode?)`, `import_hf_dataset(name, split?, config?, max_rows?, mode?)`, `start_training(name?, base_model?, epochs?, max_steps?, learning_rate?, lora_r?, lora_alpha?, max_seq_length?, batch_size?, grad_accum?, val_frac?)`, `training_status(wait_seconds?)`, `stop_training(force?)`, `list_runs`, `get_run(name)`, `chat(model, messages, max_new_tokens?, temperature?)`, `export_run(name, format)`, `export_status(name)`, `push_to_hub(name, repo_id, what?, private?)`, `delete_run(name)`.

## Notes

- To use a GGUF export with Ollama: download it, then `ollama create my-model -f Modelfile` with `FROM ./unsloth.Q4_K_M.gguf` (Unsloth also writes a Modelfile next to some exports).
- 7–8B base models fit a T4 with `max_seq_length` ≤ 2048 and batch size 1–2.
- Licences: Unsloth Apache 2.0; each base model keeps its own licence (Llama 3.x community licence, Qwen2.5 Apache 2.0 for most sizes, Gemma terms, Mistral Apache 2.0).
