"""Animated HTML for each storyboard scene, in landscape (16:9) and reel (9:16).

Every animation is a CSS animation, so the renderer can seek it exactly: window.__seek(t) pauses all
animations at t seconds and updates captions, number count-ups and code typing. Frames are therefore
identical however slow the machine is. Sizes use vmin, so one template serves both orientations.
"""
import html
import json
import os
import re

STYLES = {
    "midnight": {"bg1": "#0b1020", "bg2": "#1b2550", "fg": "#f3f5ff", "muted": "#a9b3d6", "accent": "#7c9cff",
                 "accent2": "#ff7ac6", "card": "rgba(255,255,255,.07)", "font": "'Inter','Noto Sans',system-ui,sans-serif",
                 "head": "'Inter','Noto Sans',system-ui,sans-serif", "weight": 800},
    "paper": {"bg1": "#f6f1e7", "bg2": "#e9dfcc", "fg": "#1f1a14", "muted": "#6b5f50", "accent": "#d2552d",
              "accent2": "#2d6cd2", "card": "rgba(0,0,0,.05)", "font": "'Noto Serif','DejaVu Serif',Georgia,serif",
              "head": "'Noto Serif','DejaVu Serif',Georgia,serif", "weight": 700},
    "neon": {"bg1": "#07020f", "bg2": "#1a0638", "fg": "#f6f0ff", "muted": "#b9a6e0", "accent": "#00f0ff",
             "accent2": "#ff2bd6", "card": "rgba(0,240,255,.07)", "font": "'Inter','Noto Sans',system-ui,sans-serif",
             "head": "'Noto Sans Mono','DejaVu Sans Mono',monospace", "weight": 800},
    "swiss": {"bg1": "#ffffff", "bg2": "#f1f1f1", "fg": "#111111", "muted": "#555555", "accent": "#e3001b",
              "accent2": "#111111", "card": "#f4f4f4", "font": "'Helvetica Neue','Liberation Sans',Arial,sans-serif",
              "head": "'Helvetica Neue','Liberation Sans',Arial,sans-serif", "weight": 900},
}
SIZES = {"landscape": (1920, 1080), "reel": (1080, 1920)}

CSS = """
:root{--bg1:%(bg1)s;--bg2:%(bg2)s;--fg:%(fg)s;--muted:%(muted)s;--accent:%(accent)s;--accent2:%(accent2)s;
 --card:%(card)s;--font:%(font)s;--head:%(head)s;--dur:%(dur).3fs}
*{box-sizing:border-box}
html,body{margin:0;width:100%%;height:100%%;overflow:hidden;background:var(--bg1);color:var(--fg);font-family:var(--font)}
.stage{position:absolute;inset:0;overflow:hidden;background:radial-gradient(120%% 90%% at 20%% 10%%,var(--bg2),var(--bg1) 70%%)}
.blob{position:absolute;width:70vmin;height:70vmin;border-radius:50%%;filter:blur(9vmin);opacity:.45;animation:drift var(--dur) ease-in-out both}
.blob.a{background:var(--accent);left:-15vmin;top:-20vmin}.blob.b{background:var(--accent2);right:-20vmin;bottom:-25vmin;animation-direction:reverse}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(12vmin,8vmin) scale(1.15)}}
.bgimg{position:absolute;inset:-2%%;background-size:cover;background-position:center;animation:kb var(--dur) linear both}
@keyframes kb{from{transform:scale(1.02)}to{transform:scale(1.13) translate(-1.5%%,-1%%)}}
.shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.15),rgba(0,0,0,.72))}
.dim .shade{background:linear-gradient(180deg,rgba(0,0,0,.5),rgba(0,0,0,.72))}
.paper .shade,.swiss .shade{background:linear-gradient(180deg,rgba(255,255,255,.25),rgba(255,255,255,.88))}
.up{opacity:0;animation:up .9s cubic-bezier(.2,.75,.2,1) both}
@keyframes up{from{opacity:0;transform:translateY(5vmin)}to{opacity:1;transform:none}}
.pop{opacity:0;animation:pop .8s cubic-bezier(.2,.9,.25,1.2) both}
@keyframes pop{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:none}}
.veil{position:absolute;inset:0;background:var(--bg1);pointer-events:none;z-index:50;
 animation:veilin .45s ease-out both,veilout .45s ease-in calc(var(--dur) - .45s) both}
@keyframes veilin{from{opacity:1}to{opacity:0}}@keyframes veilout{from{opacity:0}to{opacity:1}}
.first .veil{animation:veilout .45s ease-in calc(var(--dur) - .45s) both;opacity:0}
.last .veil{animation:veilin .45s ease-out both}
.progress{position:absolute;left:0;bottom:0;height:.7vmin;background:var(--accent);z-index:40;animation:prog var(--dur) linear both}
@keyframes prog{from{width:%(p0).3f%%}to{width:%(p1).3f%%}}
.content{position:absolute;inset:0;padding:8vmin 9vmin;display:flex;flex-direction:column;justify-content:center;z-index:5}
h1,h2{font-family:var(--head);font-weight:%(weight)d;letter-spacing:-.02em;margin:0;line-height:1.05}
h1{font-size:9vmin}h2{font-size:6.6vmin;margin-bottom:4vmin}
.reel h1{font-size:9.5vmin}.reel h2{font-size:7vmin}
.kicker{font:700 2.6vmin var(--font);letter-spacing:.25em;text-transform:uppercase;color:var(--accent);margin-bottom:2.4vmin}
.chip{display:inline-block;align-self:flex-start;padding:1.1vmin 2.4vmin;border-radius:99vmin;background:var(--card);
 border:.25vmin solid color-mix(in srgb,var(--accent) 60%%,transparent);color:var(--fg);font:600 3vmin var(--font);margin-bottom:3.5vmin}
.tag{font-size:3.9vmin;color:var(--muted);margin:3.5vmin 0 0;max-width:80%%;line-height:1.3}
.reel .tag{max-width:100%%}
ul{list-style:none;margin:0;padding:0}
li{font-size:4.1vmin;line-height:1.3;margin:0 0 2.6vmin;padding-left:5.5vmin;position:relative}
li::before{content:"";position:absolute;left:0;top:1.35vmin;width:2.6vmin;height:2.6vmin;border-radius:.7vmin;background:var(--accent)}
.split{flex-direction:row;align-items:center;gap:6vmin}
.split .text{flex:1.15;min-width:0}.split .pic{flex:1;height:72%%;border-radius:3vmin;overflow:hidden;position:relative;
 box-shadow:0 3vmin 7vmin rgba(0,0,0,.35)}
.reel .split{flex-direction:column-reverse;justify-content:flex-end;padding-top:10vmin}
.reel .split .pic{flex:none;width:100%%;height:46%%}.reel .split .text{flex:none;width:100%%}
.pic .bgimg{inset:0}
.lower{justify-content:flex-end;padding-bottom:16vmin}
.lower h2{font-size:7.4vmin;text-shadow:0 .5vmin 3vmin rgba(0,0,0,.45);max-width:85%%}
.reel .lower{padding-bottom:30vmin}
.paper.dim .shade,.swiss.dim .shade{background:linear-gradient(180deg,rgba(255,255,255,.7),rgba(255,255,255,.9))}
.paper .lower h2,.swiss .lower h2{text-shadow:none}
.stats{display:flex;gap:4vmin;flex-wrap:wrap}.reel .stats{flex-direction:column}
.stat{flex:1;min-width:30vmin;max-width:100%%;overflow:hidden;background:var(--card);border-radius:3vmin;padding:4.5vmin 4vmin}
.stat b{display:block;font:%(weight)d 11vmin var(--head);color:var(--accent);letter-spacing:-.03em;line-height:1}
.stat b.word{font-size:6.4vmin;line-height:1.1;overflow-wrap:anywhere}
.stat span{display:block;font-size:3.2vmin;color:var(--muted);margin-top:1.6vmin}
.win{background:#0d1117;color:#e6edf3;border-radius:2.4vmin;overflow:hidden;box-shadow:0 3vmin 7vmin rgba(0,0,0,.35)}
.win .bar{height:5vmin;background:#161b22;display:flex;align-items:center;gap:1.4vmin;padding:0 2.4vmin}
.win .bar i{width:1.8vmin;height:1.8vmin;border-radius:50%%;background:#ff5f56}.win .bar i:nth-child(2){background:#ffbd2e}.win .bar i:nth-child(3){background:#27c93f}
pre{margin:0;padding:3.6vmin 4vmin;font:3.4vmin/1.45 'Noto Sans Mono','DejaVu Sans Mono',monospace;white-space:pre-wrap;word-break:break-word;min-height:24vmin}
pre .cur{display:inline-block;width:1.4vmin;height:3.2vmin;background:var(--accent);vertical-align:-.5vmin;margin-left:.3vmin}
.qmark{font:900 22vmin/0.7 Georgia,serif;color:var(--accent);height:12vmin}
.quote{font:500 5.6vmin/1.3 var(--head);max-width:92%%}
.reel .quote{font-size:6vmin}
.src{color:var(--muted);font-size:3vmin;margin-top:4vmin}
.center{align-items:center;text-align:center}.center .chip{align-self:center}.center .tag{margin-left:auto;margin-right:auto}
.url{font:600 3.4vmin 'Noto Sans Mono','DejaVu Sans Mono',monospace;color:var(--accent);margin-top:4vmin;word-break:break-all}
.cap{position:absolute;left:50%%;bottom:6.5vmin;transform:translateX(-50%%);z-index:30;max-width:86%%;text-align:center;
 font:700 3.6vmin/1.25 var(--font);color:#fff;background:rgba(0,0,0,.62);padding:1.3vmin 2.6vmin;border-radius:1.6vmin;
 opacity:0;transition:none}
.reel .cap{bottom:15vmin;font-size:4.6vmin}
.num{position:absolute;right:5vmin;top:4.5vmin;font:700 2.4vmin var(--font);color:var(--muted);z-index:20;letter-spacing:.15em}
"""

JS = """
const CAPS = %(caps)s, COUNTS = [...document.querySelectorAll('[data-count]')], TYPE = document.querySelector('[data-type]');
const capEl = document.querySelector('.cap');
function fmtNum(v, dec){ return dec ? v.toFixed(dec) : Math.round(v).toLocaleString('en-US'); }
window.__seek = function(t){
  document.getAnimations().forEach(a => { a.pause(); a.currentTime = t * 1000; });
  if (capEl) { const c = CAPS.find(c => t >= c[0] && t < c[1]); capEl.textContent = c ? c[2] : ''; capEl.style.opacity = c ? 1 : 0; }
  COUNTS.forEach((el, i) => {
    const target = parseFloat(el.dataset.count), dec = +el.dataset.dec, f = Math.min(1, Math.max(0, (t - 0.6 - i * 0.2) / 1.4));
    el.textContent = fmtNum(target * (1 - Math.pow(1 - f, 3)), dec) + el.dataset.suffix;
  });
  if (TYPE) { const full = TYPE.dataset.type, n = Math.floor(Math.max(0, t - 0.9) * %(cps)f);
    TYPE.textContent = full.slice(0, n); const cur = document.createElement('span'); cur.className = 'cur'; TYPE.appendChild(cur); }
};
window.__ready = false;
const urls = [...document.querySelectorAll('[data-bg]')].map(e => e.dataset.bg);
Promise.all([document.fonts.ready, ...urls.map(u => new Promise(r => { const im = new Image(); im.onload = im.onerror = r; im.src = u; }))])
  .then(() => { window.__seek(0); window.__ready = true; });
"""


def esc(s):
    return html.escape(str(s or ""), quote=True)


def caption_chunks(text, start, dur, max_words):
    """[[start, end, text], ...] spread over the narration's duration, weighted by word length."""
    words = (text or "").split()
    if not words or dur <= 0:
        return []
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?;:]$", w) and len(cur) >= 3:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    total = sum(len(c) for c in chunks)
    out, t = [], start
    for c in chunks:
        d = dur * len(c) / total
        out.append([round(t, 3), round(t + d, 3), c])
        t += d
    return out


def _num(value):
    """'12.3k' -> (12.3, 1, 'k'); 'Python' -> None."""
    m = re.fullmatch(r"\s*([\d,]+(?:\.\d+)?)\s*([kKmMbB%+]*)\s*", str(value or ""))
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    return float(raw), (len(raw.split(".")[1]) if "." in raw else 0), m.group(2)


def _bg(path, cls="bgimg"):
    if not path:
        return ""
    u = "file://" + os.path.abspath(path)
    return '<div class="%s" data-bg="%s" style="background-image:url(\'%s\')"></div>' % (cls, esc(u), esc(u))


def scene_html(scene, idx, n, story, style, fmt, dur, image, captions, p0, p1):
    st = dict(STYLES.get(style, STYLES["midnight"]), dur=dur, p0=p0, p1=p1)
    lay = scene["layout"]
    src = story.get("source") or {}
    label = {"github": "GitHub · " + str(src.get("title") or ""), "youtube": "YouTube", "pdf": "Document",
             "article": str((src.get("facts") or {}).get("site") or "Article")}.get(src.get("kind"), "")
    bg = '<div class="blob a"></div><div class="blob b"></div>'
    body = ""
    h = esc(scene.get("heading"))
    if lay == "title":
        bg += _bg(image) + ('<div class="shade"></div>' if image else "")
        body = '<div class="content center">%s<h1 class="up" style="animation-delay:.3s">%s</h1>%s</div>' % (
            '<div class="chip up" style="animation-delay:.15s">%s</div>' % esc(label) if label else "", h,
            '<p class="tag up" style="animation-delay:.65s">%s</p>' % esc(story.get("tagline")) if story.get("tagline") else "")
    elif lay == "bullets":
        lis = "".join('<li class="up" style="animation-delay:%.2fs">%s</li>' % (0.7 + i * 0.45, esc(b))
                      for i, b in enumerate(scene.get("bullets") or []))
        text = '<div class="text"><div class="kicker up" style="animation-delay:.1s">%02d</div><h2 class="up" ' \
               'style="animation-delay:.25s">%s</h2><ul>%s</ul></div>' % (idx, h, lis)
        pic = '<div class="pic pop" style="animation-delay:.35s">%s</div>' % _bg(image) if image else ""
        body = '<div class="content %s">%s%s</div>' % ("split" if image else "", text, pic)
    elif lay == "image":
        bg += _bg(image) + '<div class="shade"></div>'
        sub = (scene.get("bullets") or [""])[0]
        body = '<div class="content lower"><div class="kicker up" style="animation-delay:.2s">%02d</div><h2 class="up" ' \
               'style="animation-delay:.4s">%s</h2>%s</div>' % (idx, h, '<p class="tag up" style="animation-delay:.8s">%s</p>'
                                                                % esc(sub) if sub else "")
    elif lay == "stats":
        cards = ""
        for i, s in enumerate(scene.get("stats") or []):
            num = _num(s.get("value"))
            val = ('<b data-count="%s" data-dec="%d" data-suffix="%s">0</b>' % (num[0], num[1], esc(num[2])) if num
                   else '<b class="word">%s</b>' % esc(s.get("value")))
            cards += '<div class="stat pop" style="animation-delay:%.2fs">%s<span>%s</span></div>' % (
                0.4 + i * 0.2, val, esc(s.get("label")))
        body = '<div class="content"><h2 class="up" style="animation-delay:.15s">%s</h2><div class="stats">%s</div></div>' % (h, cards)
    elif lay == "code":
        code = scene.get("code") or ""
        body = '<div class="content"><h2 class="up" style="animation-delay:.15s">%s</h2><div class="win pop" ' \
               'style="animation-delay:.4s"><div class="bar"><i></i><i></i><i></i></div><pre data-type="%s"></pre></div></div>' % (h, esc(code))
    elif lay == "quote":
        body = '<div class="content"><div class="qmark up">&ldquo;</div><div class="quote up" style="animation-delay:.3s">%s</div>' \
               '<div class="src up" style="animation-delay:.9s">%s</div></div>' % (esc(scene.get("quote")), esc(src.get("title") or h))
    else:  # outro
        bg += _bg(image) + ('<div class="shade"></div>' if image else "")
        url = re.sub(r"^https?://(www\.)?", "", str(src.get("url") or ""))
        body = '<div class="content center"><h1 class="up" style="animation-delay:.2s">%s</h1>%s%s</div>' % (
            h, '<p class="tag up" style="animation-delay:.55s">%s</p>' % esc(scene.get("bullets", [""])[0] if scene.get("bullets") else story.get("tagline")),
            '<div class="url up" style="animation-delay:.9s">%s</div>' % esc(url) if url else "")
    cls = " ".join(filter(None, [fmt, style, "dim" if lay in ("title", "outro") else "", "first" if idx == 0 else "", "last" if idx == n - 1 else ""]))
    cps = max(18.0, len(scene.get("code") or "") / max(1.0, dur - 2.2))
    return ('<!doctype html><html class="%s"><head><meta charset="utf-8"><style>%s</style></head><body>'
            '<div class="stage">%s</div>%s<div class="num">%d / %d</div>%s<div class="progress"></div><div class="veil"></div>'
            '<script>%s</script></body></html>') % (
        cls, CSS % st, bg, body, idx + 1, n, '<div class="cap"></div>' if captions else "",
        JS % {"caps": json.dumps(captions or []), "cps": cps})
