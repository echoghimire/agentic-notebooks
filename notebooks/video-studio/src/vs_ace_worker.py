"""One-shot ACE-Step music generation (Apache 2.0), run in its own virtualenv because ACE-Step pins its own
transformers. The process exits when done, so the GPU memory is freed for rendering.

    python vs_ace_worker.py --out music.wav --prompt "soft ambient pad, slow, piano" --seconds 30 --device 1
"""
import argparse
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()

from acestep.pipeline_ace_step import ACEStepPipeline  # noqa: E402

# T4 has no fast bfloat16: float32 with CPU offload keeps it within 16 GB
pipe = ACEStepPipeline(checkpoint_dir=os.environ.get("ACE_CHECKPOINTS") or None, device_id=a.device, dtype="float32",
                       torch_compile=False, cpu_offload=True, overlapped_decode=True)
raw = a.out + ".raw.wav"
pipe(format="wav", audio_duration=a.seconds, prompt=a.prompt + ", instrumental, no vocals", lyrics="[instrumental]",
     infer_step=60, guidance_scale=15.0, scheduler_type="euler", cfg_type="apg", omega_scale=10.0,
     manual_seeds=[a.seed], save_path=raw)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", "44100", "-sample_fmt", "s16", a.out], check=True)
os.remove(raw)
print("ok", a.out)
