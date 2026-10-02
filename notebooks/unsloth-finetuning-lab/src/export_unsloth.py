"""Exports a fine-tuned run. Launched by lab_server.py as its own process:
    python export_unsloth.py <export_dir>/job.json
Formats: merged_16bit (a full Hugging Face model), gguf_q4_k_m / gguf_q8_0 / gguf_f16 (for llama.cpp,
Ollama, LM Studio). GGUF builds llama.cpp on first use, which takes several minutes.
Writes <export_dir>/status.json; the exported files end up in <export_dir>/model.
"""
import glob
import json
import os
import shutil
import sys
import time
import traceback

JOB = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = JOB["export_dir"]


def status(**kw):
    st = json.load(open(os.path.join(OUT, "status.json"))) if os.path.exists(os.path.join(OUT, "status.json")) else {}
    st.update(kw, updated=time.time())
    with open(os.path.join(OUT, "status.json.tmp"), "w") as f:
        json.dump(st, f)
    os.replace(os.path.join(OUT, "status.json.tmp"), os.path.join(OUT, "status.json"))


def main():
    status(phase="loading model", started=time.time())
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_name=JOB["adapter"], max_seq_length=JOB["max_seq_length"],
                                                   load_in_4bit=True, dtype=None, token=os.environ.get("HF_TOKEN"))
    dest = os.path.join(OUT, "model")
    fmt = JOB["format"]
    if fmt == "merged_16bit":
        status(phase="merging LoRA into 16-bit weights")
        model.save_pretrained_merged(dest, tok, save_method="merged_16bit")
    elif fmt.startswith("gguf_"):
        status(phase="converting to GGUF %s (builds llama.cpp the first time)" % fmt[5:])
        model.save_pretrained_gguf(dest, tok, quantization_method=fmt[5:])
        # some Unsloth versions write next to the folder (dest + "_gguf") instead of into it
        os.makedirs(dest, exist_ok=True)
        for f in glob.glob(dest + "_gguf/*.gguf") + glob.glob(os.path.join(os.path.dirname(dest), "*.gguf")):
            shutil.move(f, os.path.join(dest, os.path.basename(f)))
        if not glob.glob(os.path.join(dest, "*.gguf")):
            raise RuntimeError("no .gguf file was produced; see the export log")
    else:
        raise ValueError("unknown format %s" % fmt)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(dest) for f in fn)
    status(phase="done", size=size, files=sorted(os.listdir(dest)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        status(phase="error", error="%s: %s" % (type(e).__name__, str(e)[:800]))
        sys.exit(1)
