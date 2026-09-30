"""Fine-tune a Laya checkpoint on a Laya Studio dataset.

Launched by the Studio UI as:  torchrun --standalone --nproc_per_node=N train_laya.py job.json
(or plain `python train_laya.py job.json` without a GPU).

The loss is the one from the official Laya Kaggle notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): a policy-gradient term on a
proper scoring reward plus soft cross-entropy, followed by per-type temperature calibration.
Additions here: 1..N GPUs, a document-level validation split, baseline/per-epoch accuracy,
a status.json the UI polls, and a load-and-predict smoke test of the saved model.
"""
import datetime
import json
import math
import os
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist

from laya_data import gold_to_target, load_jsonl

HUB_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
FWD_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
            "label": torch.tensor([it["label"] for it in items])}


def fit_one_temp(sel):
    """Temperature that minimises cross-entropy of softmax(logits / T) against the targets."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    # laya clamps temperatures to [0.5, 5.0] at load time and warns outside it
    return 1.0 if not math.isfinite(t) else min(5.0, max(0.5, t))


class Status:
    def __init__(self, path, enabled):
        self.path, self.enabled = path, enabled
        self.d = {"phase": "starting", "started": time.time()}

    def set(self, **kw):
        if not self.enabled:
            return
        self.d.update(kw)
        self.d["updated"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f)
        os.replace(tmp, self.path)


def evaluate(model, items, pad_id, device, use_amp, bs=16):
    """Accuracy (argmax == gold label) overall and per question id; also returns logits."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i: i + bs]
            b = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(*(b[k].to(device) for k in FWD_KEYS))
            l = logits.float().cpu()
            for r, it in enumerate(chunk):
                preds.append((it, l[r, : len(it["markers"])].tolist()))
    per_q, correct = {}, 0
    for it, z in preds:
        ok = int(max(range(len(z)), key=lambda j: z[j]) == it["label"])
        correct += ok
        d = per_q.setdefault(it["qid"], [0, 0])
        d[0] += ok
        d[1] += 1
    metrics = {"accuracy": round(correct / len(preds), 4) if preds else None, "n": len(preds),
               "per_question": {q: {"accuracy": round(c / n, 4), "n": n} for q, (c, n) in sorted(per_q.items())}}
    return metrics, preds


def main():
    job = json.load(open(sys.argv[1]))
    run_dir = job["run_dir"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(hours=3))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    is_main = rank == 0
    st = Status(os.path.join(run_dir, "status.json"), is_main)
    try:
        train(job, run_dir, world_size, rank, local_rank, device, use_cuda, distributed, is_main, st)
    except Exception:
        st.set(phase="error", error=traceback.format_exc()[-4000:])
        raise
    finally:
        if distributed:
            dist.destroy_process_group()


def train(job, run_dir, world_size, rank, local_rank, device, use_cuda, distributed, is_main, st):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_model, build_sequence, proper_reward

    def log(*a):
        if is_main:
            print(*a, flush=True)

    def barrier():
        if distributed:
            dist.barrier()

    t_start = time.time()
    base = job["base"]

    # ------------------------------------------------------------ base checkpoint
    st.set(phase="downloading", message="Fetching %s" % base)

    def resolve():
        return base if os.path.isdir(base) else snapshot_download(base, allow_patterns=HUB_FILES)

    if is_main:
        model_dir = resolve()
        _fix_tokenizer_config(model_dir)
    barrier()
    if not is_main:
        model_dir = resolve()
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    max_len = int(job.get("max_len") or cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    log("Base: %s | max_len=%d head_max_len=%d | world_size=%d | device=%s"
        % (base, max_len, head_max_len, world_size, device))

    # ------------------------------------------------------------ data
    st.set(phase="preparing", message="Tokenising dataset")
    records = load_jsonl(job["dataset"])
    order = list(range(len(records)))
    random.Random(int(job.get("seed", 42))).shuffle(order)
    val_frac = float(job.get("val_frac", 0.15))
    n_val = int(round(len(records) * val_frac)) if len(records) >= 10 and val_frac > 0 else 0
    val_ids = set(order[:n_val])       # split by document so no document leaks into both

    def build(ids):
        items, skipped = [], 0
        for i in ids:
            rec = records[i]
            for qid, g in rec["gold"].items():
                q = rec["questions"][qid]
                target, label = gold_to_target(q, g)
                iq = {"t": q["type"], "ins": q.get("instructions", ""), "crit": q.get("criteria")}
                seq, markers = build_sequence(tok, rec["state"], iq, max_len, head_max_len)
                if len(markers) != len(target):
                    skipped += 1       # options did not fit in head_max_len
                    continue
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]],
                              "target": target, "label": label, "qid": qid})
        return items, skipped

    train_items, sk1 = build([i for i in order if i not in val_ids])
    val_items, sk2 = build([i for i in order if i in val_ids])
    per_rank = len(train_items) // world_size
    if per_rank == 0:
        raise RuntimeError("Not enough training decisions (%d) for %d GPU(s)" % (len(train_items), world_size))
    # equal-length shards: unequal counts make DDP ranks wait on each other forever
    my_items = train_items[rank::world_size][:per_rank]
    log("Records: %d (%d train / %d val) | decisions: %d train, %d val | skipped: %d"
        % (len(records), len(records) - n_val, n_val, len(train_items), len(val_items), sk1 + sk2))

    # ------------------------------------------------------------ model
    st.set(phase="loading_model", message="Loading weights")
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if use_cuda:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    use_amp = use_cuda
    pad_id = tok.pad_token_id

    baseline = None
    if is_main and val_items:
        st.set(phase="baseline_eval", message="Scoring the base model on the validation set")
        baseline, _ = evaluate(model, val_items, pad_id, device, use_amp)
        log("Baseline val accuracy: %s" % baseline["accuracy"])
    barrier()

    model.train()
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank] if use_cuda else None, find_unused_parameters=True)
    else:
        net = model

    EPOCHS = int(job.get("epochs", 4))
    MICRO_BATCH = int(job.get("micro_batch", 8))
    GRAD_ACCUM = int(job.get("grad_accum", 4))
    GROUP_SIZE = 4
    LR_ENCODER = float(job.get("lr_encoder", 2.5e-5))
    LR_HEAD = float(job.get("lr_head", 1.0e-4))
    SIGMA_START, SIGMA_END = 0.4, 0.1

    enc_params = [p for n, p in net.named_parameters() if "encoder." in n]
    head_params = [p for n, p in net.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENCODER},
                                   {"params": head_params, "lr": LR_HEAD}], weight_decay=0.01)
    micro_per_epoch = math.ceil(len(my_items) / MICRO_BATCH)
    total_updates = math.ceil(micro_per_epoch / GRAD_ACCUM) * EPOCHS
    total_micro = micro_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist, done_micro, t0 = [], 0, time.time()
    recent = []
    st.set(phase="training", epoch=0, epochs=EPOCHS, step=0, total_steps=total_micro, baseline=baseline,
           message="Training on %d decisions x %d GPU(s)" % (len(my_items) * world_size, world_size))
    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches, accum_step = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx: b_idx + MICRO_BATCH]
            batch = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, act = net(*(batch[k].to(device) for k in FWD_KEYS))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)

            # G noisy logit samples (zero-mean over options) scored with a proper scoring rule
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            lv = loss.item() * GRAD_ACCUM
            epoch_loss += lv
            n_batches += 1
            done_micro += 1
            recent = (recent + [lv])[-25:]
            if is_main and (done_micro % 10 == 0 or done_micro == total_micro):
                el = time.time() - t0
                st.set(epoch=epoch + 1, step=done_micro, loss=round(sum(recent) / len(recent), 4),
                       reward=round(r.mean().item(), 4), lr=scheduler.get_last_lr()[0],
                       eta_s=int(el / done_micro * (total_micro - done_micro)))
            if is_main and done_micro % 50 == 0:
                log("epoch %d/%d step %d/%d loss %.4f" % (epoch + 1, EPOCHS, done_micro, total_micro, lv))

        barrier()
        entry = {"epoch": epoch + 1, "train_loss": round(epoch_loss / max(1, n_batches), 4)}
        if is_main and val_items:
            st.set(message="Validating after epoch %d" % (epoch + 1))
            m, _ = evaluate(model, val_items, pad_id, device, use_amp)
            entry["val_accuracy"] = m["accuracy"]
            model.train()
        if is_main:
            hist.append(entry)
            st.set(history=hist, message="Epoch %d/%d done" % (epoch + 1, EPOCHS))
            log("=== epoch %d/%d | loss %.4f | val acc %s | %.0fs ===" % (
                epoch + 1, EPOCHS, entry["train_loss"], entry.get("val_accuracy"), time.time() - t0))
        barrier()

    if not is_main:
        return

    # ------------------------------------------------------------ calibrate, evaluate, save
    del optimizer, scaler, scheduler
    if use_cuda:
        torch.cuda.empty_cache()
    st.set(phase="calibrating", message="Fitting confidence temperatures")
    final, val_preds = evaluate(model, val_items, pad_id, device, use_amp) if val_items else (None, [])
    calib = val_preds if len(val_preds) >= 30 else evaluate(
        model, train_items[:: max(1, len(train_items) // 400)][:400], pad_id, device, use_amp)[1]
    temps = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, it["target"]) for it, z in calib if it["qtype"] == qt]
        if sel:
            try:
                temps[qt] = round(fit_one_temp(sel), 4)
            except Exception as e:  # keep training result even if calibration fails
                log("temperature fit failed for type %d: %s" % (qt, e))
    log("Temperatures (choice, score, noul): %s" % temps)

    st.set(phase="saving", message="Saving model")
    out_dir = os.path.join(run_dir, "model")
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    _fix_tokenizer_config(out_dir)
    new_cfg = dict(cfg)
    new_cfg.update(fine_tuned=True, model_name=job["name"], base_model=base, temperature=temps,
                   max_len=max_len, head_max_len=head_max_len)
    new_cfg.pop("temperature_by_options", None)   # base buckets would override the new temperatures
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)

    # schema = every question seen in the data, so the Test tab can reuse it
    schema = {}
    for rec in records:
        schema.update(rec["questions"])
    with open(os.path.join(out_dir, "questions.json"), "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    smoke = None
    try:
        st.set(message="Loading the saved model to check it works")
        del model
        if use_cuda:
            torch.cuda.empty_cache()
        import laya
        agent = laya.Agent(out_dir, device=str(device))
        rec = records[order[0]]
        smoke = agent.predict(rec["state"], rec["questions"])["answers"]
    except Exception as e:
        smoke = {"error": str(e)}
        log("Smoke test failed: %s" % e)

    metrics = {
        "name": job["name"], "base": base, "created": time.time(),
        "duration_s": int(time.time() - t_start), "gpus": world_size,
        "records": len(records), "train_records": len(records) - n_val, "val_records": n_val,
        "train_decisions": len(train_items), "val_decisions": len(val_items), "skipped": sk1 + sk2,
        "hyperparams": {"epochs": EPOCHS, "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
                        "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD, "max_len": max_len},
        "baseline": baseline, "final": final, "history": hist, "temperatures": temps, "smoke_test": smoke,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    st.set(phase="done", message="Finished in %d min" % (metrics["duration_s"] // 60), metrics=metrics, eta_s=0)
    log("Done. Model saved to %s" % out_dir)


if __name__ == "__main__":
    main()
