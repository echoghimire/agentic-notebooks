"""Fine-tunes a model with Unsloth (4-bit QLoRA). Launched by lab_server.py as its own process:
    python train_unsloth.py <run_dir>/job.json
It writes <run_dir>/status.json while it runs, the LoRA adapter to <run_dir>/adapter, and
metrics.json + samples.json at the end. SIGTERM stops training early; the adapter trained so far
is still saved.
"""
import json
import math
import os
import random
import signal
import sys
import time
import traceback

JOB = json.load(open(sys.argv[1], encoding="utf-8"))
RD = JOB["run_dir"]
ST = {"phase": "starting", "started": time.time()}
STOP = {"flag": False}
LOSSES, EVALS = [], []


def status(**kw):
    ST.update(kw, updated=time.time())
    tmp = os.path.join(RD, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ST, f)
    os.replace(tmp, os.path.join(RD, "status.json"))


def on_term(*_):
    STOP["flag"] = True
    print("Stop requested: finishing the current step, then saving the adapter.", flush=True)


signal.signal(signal.SIGTERM, on_term)


def sft_config(SFTConfig, n_train, n_eval):
    """Builds SFTConfig with whichever argument names this TRL version uses."""
    fields = set(getattr(SFTConfig, "__dataclass_fields__", {}))
    bs, ga = JOB["batch_size"], JOB["grad_accum"]
    steps_per_epoch = max(1, math.ceil(n_train / (bs * ga)))
    total = JOB["max_steps"] if JOB["max_steps"] > 0 else math.ceil(steps_per_epoch * JOB["epochs"])
    import torch
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = dict(output_dir=os.path.join(RD, "checkpoints"), per_device_train_batch_size=bs,
               gradient_accumulation_steps=ga, learning_rate=JOB["learning_rate"], num_train_epochs=JOB["epochs"],
               max_steps=JOB["max_steps"] if JOB["max_steps"] > 0 else -1, warmup_ratio=0.03,
               lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.01, logging_steps=1,
               save_strategy="no", report_to="none", seed=JOB["seed"], fp16=not bf16, bf16=bf16,
               dataset_text_field="text", packing=False, dataset_num_proc=2)
    cfg["max_seq_length" if "max_seq_length" in fields else "max_length"] = JOB["max_seq_length"]
    if n_eval:
        cfg["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "steps"
        cfg["eval_steps"] = max(1, total // 5)
        cfg["per_device_eval_batch_size"] = bs
    dropped = sorted(k for k in cfg if fields and k not in fields)
    if dropped:
        print("SFTConfig in this TRL version has no", dropped, flush=True)
    return SFTConfig(**{k: v for k, v in cfg.items() if not fields or k in fields}), total


def main():
    status(phase="loading model", base=JOB["base"])
    from unsloth import FastLanguageModel           # import unsloth first: it patches transformers and trl
    from unsloth.chat_templates import get_chat_template
    import inspect
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(model_name=JOB["base"], max_seq_length=JOB["max_seq_length"],
                                                   load_in_4bit=True, dtype=None, token=os.environ.get("HF_TOKEN"))
    if getattr(tok, "chat_template", None) is None:
        tok = get_chat_template(tok, chat_template="chatml")
    model = FastLanguageModel.get_peft_model(
        model, r=JOB["lora_r"], lora_alpha=JOB["lora_alpha"], lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=JOB["seed"])

    status(phase="preparing data")
    recs = [json.loads(line) for line in open(JOB["dataset"], encoding="utf-8") if line.strip()]
    eos = tok.eos_token or ""
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) if "messages" in r else r["text"] + eos
             for r in recs]
    order = list(range(len(texts)))
    random.Random(JOB["seed"]).shuffle(order)
    n_eval = min(200, int(len(texts) * JOB["val_frac"])) if len(texts) >= 20 else 0
    eval_idx, train_idx = order[:n_eval], order[n_eval:]
    train_ds = Dataset.from_dict({"text": [texts[i] for i in train_idx]})
    eval_ds = Dataset.from_dict({"text": [texts[i] for i in eval_idx]}) if n_eval else None
    args, total = sft_config(SFTConfig, len(train_idx), n_eval)

    t0 = time.time()

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" in logs:
                LOSSES.append([state.global_step, round(logs["loss"], 4)])
            if "eval_loss" in logs:
                EVALS.append([state.global_step, round(logs["eval_loss"], 4)])
            done = max(1, state.global_step)
            el = time.time() - t0
            status(phase="training", step=state.global_step, total_steps=state.max_steps or total,
                   epoch=round(state.epoch or 0, 3), loss=LOSSES[-1][1] if LOSSES else None,
                   eval_loss=EVALS[-1][1] if EVALS else None, lr=logs.get("learning_rate", ST.get("lr")),
                   losses=LOSSES[-1000:], evals=EVALS, elapsed=round(el),
                   eta=round(el / done * max(0, (state.max_steps or total) - state.global_step)))

        def on_step_end(self, args, state, control, **kw):
            if STOP["flag"]:
                control.should_training_stop = True
            return control

    params = inspect.signature(SFTTrainer.__init__).parameters
    kw = dict(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=args, callbacks=[Progress()])
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tok
    trainer = SFTTrainer(**kw)
    status(phase="training", step=0, total_steps=total, n_train=len(train_idx), n_eval=n_eval)
    result = trainer.train()

    status(phase="saving adapter")
    adapter = os.path.join(RD, "adapter")
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    metrics = {"train_loss": round(result.training_loss, 4) if result.training_loss else None,
               "final_loss": LOSSES[-1][1] if LOSSES else None,
               "eval_loss": EVALS[-1][1] if EVALS else None, "first_eval_loss": EVALS[0][1] if EVALS else None,
               "steps": trainer.state.global_step, "planned_steps": total, "stopped_early": STOP["flag"],
               "seconds": round(time.time() - t0), "n_train": len(train_idx), "n_eval": n_eval,
               "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2) if torch.cuda.is_available() else None}
    with open(os.path.join(RD, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    samples = []
    held_out = [recs[i] for i in eval_idx if "messages" in recs[i]][:3]
    if held_out and not STOP["flag"]:
        status(phase="writing sample answers")
        FastLanguageModel.for_inference(model)
        for r in held_out:
            msgs = r["messages"]
            last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
            prompt = msgs[:last]
            ids = tok.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True)["input_ids"].to(model.device)
            out = model.generate(input_ids=ids, max_new_tokens=256, do_sample=False)
            samples.append({"prompt": prompt[-1]["content"], "expected": msgs[last]["content"],
                            "model": tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()})
    with open(os.path.join(RD, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=1, ensure_ascii=False)
    status(phase="stopped" if STOP["flag"] else "done", metrics=metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        status(phase="error", error="%s: %s" % (type(e).__name__, e))
        sys.exit(1)
