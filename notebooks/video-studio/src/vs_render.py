"""Renders one format of a job to MP4. Run as its own process (the server starts one per format, in parallel):
    python vs_render.py <job_dir> <landscape|reel>

Reads <job_dir>/plan.json (scenes with durations, images, captions; written by vs_server.py), writes one
HTML file per scene, seeks every frame in headless Chromium, pipes JPEG screenshots into ffmpeg, joins the
scenes and muxes <job_dir>/audio.wav. Progress goes to <job_dir>/<fmt>/progress.json.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import vs_scenes as V


def progress(out_dir, **kw):
    tmp = os.path.join(out_dir, "progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(kw, updated=time.time()), f)
    os.replace(tmp, os.path.join(out_dir, "progress.json"))


def ffmpeg(*args):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr[-1500:])


def main(job_dir, fmt):
    plan = json.load(open(os.path.join(job_dir, "plan.json"), encoding="utf-8"))
    out_dir = os.path.join(job_dir, fmt)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    fps = int(plan.get("fps", 30))
    w, h = V.SIZES[fmt]
    if plan.get("quality") == "720p":
        w, h = w * 2 // 3, h * 2 // 3
    scenes = plan["scenes"]
    total_frames = sum(max(1, round(s["dur"] * fps)) for s in scenes)
    total_dur = sum(s["dur"] for s in scenes)
    done, t_acc = 0, 0.0
    progress(out_dir, state="starting", frames=0, total_frames=total_frames)
    from playwright.sync_api import sync_playwright
    t0 = time.time()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None,
                                     args=["--no-sandbox", "--disable-dev-shm-usage", "--font-render-hinting=none"])
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        parts = []
        for i, sc in enumerate(scenes):
            caps = V.caption_chunks(sc["narration"], sc["narr_start"], sc["narr_dur"],
                                    5 if fmt == "reel" else 9) if plan.get("captions") else None
            doc = V.scene_html(sc, i, len(scenes), plan["story"], plan["style"], fmt, sc["dur"], sc.get("image"),
                               caps, 100 * t_acc / total_dur, 100 * (t_acc + sc["dur"]) / total_dur)
            t_acc += sc["dur"]
            path = os.path.join(out_dir, "scene_%02d.html" % i)
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
            page.goto("file://" + os.path.abspath(path))
            page.wait_for_function("window.__ready === true", timeout=60000)
            n = max(1, round(sc["dur"] * fps))
            mp4 = os.path.join(out_dir, "scene_%02d.mp4" % i)
            enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe",
                                    "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264",
                                    "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(fps), mp4],
                                   stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for k in range(n):
                    page.evaluate("t => window.__seek(t)", k / fps)
                    enc.stdin.write(page.screenshot(type="jpeg", quality=92))
                    done += 1
                    if done % 15 == 0:
                        el = time.time() - t0
                        progress(out_dir, state="rendering", scene=i + 1, scenes=len(scenes), frames=done,
                                 total_frames=total_frames, eta=round(el / done * (total_frames - done)))
                    if i == 0 and k == min(n - 1, int(1.8 * fps)):
                        page.screenshot(path=os.path.join(out_dir, "poster.jpg"), type="jpeg", quality=88)
            finally:
                enc.stdin.close()
                err = enc.stderr.read().decode("utf-8", "replace")
                if enc.wait() != 0:
                    raise RuntimeError("ffmpeg could not encode scene %d: %s" % (i + 1, err[-800:]))
            parts.append(mp4)
        browser.close()
    progress(out_dir, state="joining", frames=done, total_frames=total_frames)
    lst = os.path.join(out_dir, "parts.txt")
    with open(lst, "w") as f:
        f.writelines("file '%s'\n" % os.path.abspath(p) for p in parts)
    silent = os.path.join(out_dir, "silent.mp4")
    ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent)
    final = os.path.join(job_dir, "%s.mp4" % fmt)
    audio = os.path.join(job_dir, "audio.wav")
    if os.path.exists(audio):
        ffmpeg("-i", silent, "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-shortest", "-movflags", "+faststart", final + ".tmp.mp4")
    else:
        ffmpeg("-i", silent, "-c", "copy", "-movflags", "+faststart", final + ".tmp.mp4")
    os.replace(final + ".tmp.mp4", final)
    for p in parts + [silent, lst]:
        os.remove(p)
    if os.path.exists(os.path.join(out_dir, "poster.jpg")):
        shutil.copy(os.path.join(out_dir, "poster.jpg"), os.path.join(job_dir, "%s.jpg" % fmt))
    progress(out_dir, state="done", frames=done, total_frames=total_frames, seconds=round(time.time() - t0),
             size=os.path.getsize(final))


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as e:
        import traceback
        traceback.print_exc()
        d = os.path.join(sys.argv[1], sys.argv[2])
        os.makedirs(d, exist_ok=True)
        progress(d, state="error", error="%s: %s" % (type(e).__name__, str(e)[:800]))
        sys.exit(1)
