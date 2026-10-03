"""Animated HTML for each storyboard scene, in landscape (16:9) and reel (9:16).

Every animation is a CSS animation, so the renderer can seek it exactly: window.__seek(t) pauses all
animations at t seconds and updates captions, count-ups and code typing. Frames are identical however slow
the machine is. Sizes use vmin, so one template serves both orientations.

Photos (from the source page, or generated) get two layers: a blurred, darkened copy that fills the frame
and the photo itself, cropped when its shape suits the frame and letterboxed over the blur when it does not,
then colour-graded, vignetted and slowly pushed in. Styles: broadcast (news), documentary (film grain, light
leaks), midnight, paper, neon, swiss. Labels follow the video's language.
"""
import html
import json
import os
import re
from string import Template

DEVA = "'Noto Sans Devanagari','Mukta','Noto Sans',sans-serif"
STYLES = {
    "broadcast": {"bg1": "#0a0d14", "bg2": "#18233b", "fg": "#ffffff", "muted": "#c4cad6", "accent": "#e1261c",
                  "accent2": "#ffcc00", "card": "rgba(255,255,255,.08)", "panel": "rgba(8,12,22,.86)",
                  "font": "'Inter','Noto Sans'," + DEVA, "head": "'Inter','Noto Sans'," + DEVA, "weight": "800"},
    "documentary": {"bg1": "#0d0b09", "bg2": "#2a2219", "fg": "#f5efe6", "muted": "#cbbfae", "accent": "#e9b872",
                    "accent2": "#c4572e", "card": "rgba(255,240,220,.07)", "panel": "rgba(14,11,8,.72)",
                    "font": "'Noto Serif','DejaVu Serif','Noto Serif Devanagari'," + DEVA,
                    "head": "'Noto Serif','DejaVu Serif','Noto Serif Devanagari'," + DEVA, "weight": "700"},
    "midnight": {"bg1": "#0b1020", "bg2": "#1b2550", "fg": "#f3f5ff", "muted": "#a9b3d6", "accent": "#7c9cff",
                 "accent2": "#ff7ac6", "card": "rgba(255,255,255,.07)", "panel": "rgba(11,16,32,.78)",
                 "font": "'Inter','Noto Sans'," + DEVA, "head": "'Inter','Noto Sans'," + DEVA, "weight": "800"},
    "paper": {"bg1": "#f6f1e7", "bg2": "#e9dfcc", "fg": "#1f1a14", "muted": "#6b5f50", "accent": "#d2552d",
              "accent2": "#2d6cd2", "card": "rgba(0,0,0,.05)", "panel": "rgba(246,241,231,.9)",
              "font": "'Noto Serif','DejaVu Serif','Noto Serif Devanagari'," + DEVA,
              "head": "'Noto Serif','DejaVu Serif','Noto Serif Devanagari'," + DEVA, "weight": "700"},
    "neon": {"bg1": "#07020f", "bg2": "#1a0638", "fg": "#f6f0ff", "muted": "#b9a6e0", "accent": "#00f0ff",
             "accent2": "#ff2bd6", "card": "rgba(0,240,255,.07)", "panel": "rgba(7,2,15,.8)",
             "font": "'Inter','Noto Sans'," + DEVA, "head": "'Noto Sans Mono','DejaVu Sans Mono'," + DEVA, "weight": "800"},
    "swiss": {"bg1": "#ffffff", "bg2": "#f1f1f1", "fg": "#111111", "muted": "#555555", "accent": "#e3001b",
              "accent2": "#111111", "card": "#f4f4f4", "panel": "rgba(255,255,255,.92)",
              "font": "'Helvetica Neue','Liberation Sans',Arial," + DEVA, "head": "'Helvetica Neue','Liberation Sans',Arial," + DEVA,
              "weight": "900"},
}
LIGHT = {"paper", "swiss"}
SIZES = {"landscape": (1920, 1080), "reel": (1080, 1920)}
LABELS = {
    "en": {"latest": "LATEST", "numbers": "By the numbers", "source": "Source", "photo": "Photo", "thanks": "Thanks for watching"},
    "ne": {"latest": "ताजा समाचार", "numbers": "तथ्यांकमा", "source": "स्रोत", "photo": "तस्बिर", "thanks": "हेर्नुभएकोमा धन्यवाद"},
    "hi": {"latest": "ताज़ा ख़बर", "numbers": "आँकड़ों में", "source": "स्रोत", "photo": "फ़ोटो", "thanks": "देखने के लिए धन्यवाद"},
    "es": {"latest": "ÚLTIMA HORA", "numbers": "En cifras", "source": "Fuente", "photo": "Foto", "thanks": "Gracias por ver"},
    "fr": {"latest": "DERNIÈRE MINUTE", "numbers": "En chiffres", "source": "Source", "photo": "Photo", "thanks": "Merci d'avoir regardé"},
}
DEVANAGARI_LANGS = {"ne", "hi", "mr", "sa", "mai", "bho", "new"}
GRAIN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'>"
         "<feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/></filter>"
         "<rect width='100%25' height='100%25' filter='url(%23n)' opacity='.55'/></svg>")

CSS = Template("""
:root{--bg1:$bg1;--bg2:$bg2;--fg:$fg;--muted:$muted;--accent:$accent;--accent2:$accent2;--card:$card;--panel:$panel;
 --font:$font;--head:$head;--dur:${dur}s}
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg1);color:var(--fg);font-family:var(--font)}
.stage{position:absolute;inset:0;overflow:hidden;background:radial-gradient(120% 90% at 20% 10%,var(--bg2),var(--bg1) 70%)}
.blob{position:absolute;width:70vmin;height:70vmin;border-radius:50%;filter:blur(9vmin);opacity:.42;animation:drift var(--dur) ease-in-out both}
.blob.a{background:var(--accent);left:-15vmin;top:-20vmin}.blob.b{background:var(--accent2);right:-20vmin;bottom:-25vmin;animation-direction:reverse}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(12vmin,8vmin) scale(1.15)}}
/* photo layers */
.ph{position:absolute;inset:0;overflow:hidden}
.ph .fill{position:absolute;inset:-6%;background-size:cover;background-position:center;filter:blur(5vmin) brightness(.55) saturate(1.2)}
.ph .img{position:absolute;inset:0;background-repeat:no-repeat;background-position:center;filter:contrast(1.06) saturate(1.06);
 animation:$kb var(--dur) cubic-bezier(.3,.1,.3,1) both}
.ph .img{background-size:cover}
.ph .img.band{box-shadow:0 2vmin 7vmin rgba(0,0,0,.55);animation-name:kbband!important}
.reel .ph .img.band{inset:auto 0 auto 0;top:13vmin}
.landscape .ph .img.band{inset:0 0 0 auto}
@keyframes kbband{from{transform:scale(1)}to{transform:scale(1.05)}}
@keyframes kbin{from{transform:scale(1.04)}to{transform:scale(1.16)}}
@keyframes kbout{from{transform:scale(1.16)}to{transform:scale(1.04)}}
@keyframes kbleft{from{transform:scale(1.14) translateX(2.5%)}to{transform:scale(1.14) translateX(-2.5%)}}
@keyframes kbright{from{transform:scale(1.14) translateX(-2.5%)}to{transform:scale(1.14) translateX(2.5%)}}
.ph .vig{position:absolute;inset:0;background:radial-gradient(120% 95% at 50% 45%,transparent 55%,rgba(0,0,0,.55))}
.ph .shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.05) 35%,rgba(0,0,0,.78))}
.light .ph .shade{background:linear-gradient(180deg,rgba(255,255,255,0) 40%,rgba(255,255,255,.9))}
.dim .ph .shade{background:linear-gradient(180deg,rgba(0,0,0,.45),rgba(0,0,0,.75))}
.light.dim .ph .shade{background:linear-gradient(180deg,rgba(255,255,255,.65),rgba(255,255,255,.9))}
/* text over a photo, in every style: a dark scrim under the text, white type with a soft shadow, and a
   frosted plate behind centred headlines. Light styles switch to this too: dark type on photos is unreadable. */
html.onphoto{--fg:#fff;--muted:rgba(255,255,255,.88)}
html.onphoto .ph .shade{background:linear-gradient(180deg,rgba(0,0,0,.35) 0%,rgba(0,0,0,0) 22%,rgba(0,0,0,.25) 48%,rgba(0,0,0,.82) 78%,rgba(0,0,0,.92))}
html.onphoto.dim .ph .shade{background:linear-gradient(180deg,rgba(0,0,0,.5),rgba(0,0,0,.38) 45%,rgba(0,0,0,.8))}
html.onphoto h1,html.onphoto h2,html.onphoto .tag,html.onphoto .quote,html.onphoto li{text-shadow:0 .2vmin .4vmin rgba(0,0,0,.6),0 .6vmin 3vmin rgba(0,0,0,.5)}
html.onphoto .kicker{color:#fff;background:var(--accent);display:inline-block;align-self:flex-start;padding:.7vmin 1.6vmin;border-radius:.5vmin;text-shadow:none}
html.onphoto .center .kicker{align-self:center}
html.onphoto .chip{background:rgba(0,0,0,.5);color:#fff}
html.onphoto .url{color:#fff;text-shadow:0 .2vmin 1vmin rgba(0,0,0,.7)}
.plate{background:rgba(10,12,18,.55);backdrop-filter:blur(1.8vmin) saturate(1.15);-webkit-backdrop-filter:blur(1.8vmin);
 border:.15vmin solid rgba(255,255,255,.12);border-radius:2.6vmin;padding:4.5vmin 5.5vmin;max-width:100%;
 box-shadow:0 3vmin 8vmin rgba(0,0,0,.35)}
.center .plate{display:flex;flex-direction:column;align-items:center}
/* band photos: the text goes beside (landscape) or below (reel) the picture, on the darkened blur */
html.onphoto.pc.reel .ph .shade{background:linear-gradient(180deg,rgba(0,0,0,.5),rgba(0,0,0,0) 13vmin,rgba(0,0,0,0) calc(var(--below) - 8vmin),rgba(0,0,0,.72) var(--below),rgba(0,0,0,.85))}
html.onphoto.pc.landscape .ph .shade{background:linear-gradient(90deg,rgba(0,0,0,.82),rgba(0,0,0,.62) 45%,rgba(0,0,0,.08) 62%)}
.pc.reel .content.lower,.pc.reel .content.center,.pc.reel .content.qt{justify-content:center;padding-top:var(--below,82vmin);padding-bottom:44vmin}
.pc.reel h1{font-size:8.2vmin}.pc.reel h2,.pc.reel .lower h2{font-size:6.8vmin;margin-bottom:2vmin}.pc.reel .tag{font-size:3.7vmin;margin-top:2.4vmin}
.pc.reel .plate{padding:3.6vmin 4.4vmin}
.pc.reel .l3{bottom:auto;top:var(--below,82vmin)}
.pc.landscape .content{padding-right:var(--beside,92vmin);padding-bottom:12vmin}
.pc.landscape .center{align-items:flex-start;text-align:left}.pc.landscape .center .plate{align-items:flex-start}
.pc.landscape .center .tag{margin-left:0}.pc.landscape .lower h2{max-width:100%}
.pc.landscape .l3{right:var(--beside,92vmin)}
.credit{position:absolute;right:3vmin;bottom:3vmin;z-index:25;font:500 1.7vmin var(--font);color:rgba(255,255,255,.75);
 text-shadow:0 .2vmin .6vmin rgba(0,0,0,.6)}
.reel .credit,.broadcast .credit{bottom:auto;top:9vmin}
.broadcast:not(.reel) .credit{top:5vmin}
/* motion */
.up{opacity:0;animation:up .9s cubic-bezier(.2,.75,.2,1) both}
@keyframes up{from{opacity:0;transform:translateY(5vmin)}to{opacity:1;transform:none}}
.pop{opacity:0;animation:pop .8s cubic-bezier(.2,.9,.25,1.2) both}
@keyframes pop{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:none}}
.w{display:inline-block;opacity:0;animation:wd .7s cubic-bezier(.2,.75,.2,1) both;white-space:pre}
@keyframes wd{from{opacity:0;transform:translateY(60%);filter:blur(.6vmin)}to{opacity:1;transform:none;filter:none}}
.slide{opacity:0;animation:sl .7s cubic-bezier(.2,.8,.2,1) both}
@keyframes sl{from{opacity:0;transform:translateX(-6vmin)}to{opacity:1;transform:none}}
/* transitions */
.veil{position:absolute;inset:0;background:var(--bg1);pointer-events:none;z-index:60;
 animation:veilin .45s ease-out both,veilout .45s ease-in calc(var(--dur) - .45s) both}
@keyframes veilin{from{opacity:1}to{opacity:0}}@keyframes veilout{from{opacity:0}to{opacity:1}}
.first .veil{animation:veilout .45s ease-in calc(var(--dur) - .45s) both;opacity:0}
.last .veil{animation:veilin .45s ease-out both}
.wipe{position:absolute;inset:0;z-index:61;pointer-events:none;background:var(--accent);
 animation:wipein .5s cubic-bezier(.7,0,.3,1) both,wipeout .4s cubic-bezier(.7,0,.3,1) calc(var(--dur) - .4s) both}
@keyframes wipein{from{transform:translateX(0)}to{transform:translateX(101%)}}
@keyframes wipeout{from{transform:translateX(-101%)}to{transform:translateX(0)}}
.first .wipe{animation:wipeout .4s cubic-bezier(.7,0,.3,1) calc(var(--dur) - .4s) both;transform:translateX(-101%)}
.last .wipe{animation:wipein .5s cubic-bezier(.7,0,.3,1) both}
.broadcast .veil{display:none}
.progress{position:absolute;left:0;bottom:0;height:.7vmin;background:var(--accent);z-index:40;animation:prog var(--dur) linear both}
@keyframes prog{from{width:${p0}%}to{width:${p1}%}}
/* texture */
.grain{position:absolute;inset:-50%;z-index:45;pointer-events:none;opacity:.16;background-image:url("$grain");
 animation:grain .5s steps(5) infinite;mix-blend-mode:overlay}
@keyframes grain{0%{transform:translate(0,0)}20%{transform:translate(-3%,2%)}40%{transform:translate(2%,-3%)}60%{transform:translate(-2%,-1%)}80%{transform:translate(3%,3%)}}
.leak{position:absolute;inset:0;z-index:44;pointer-events:none;mix-blend-mode:screen;
 background:radial-gradient(40% 55% at 0% 30%,rgba(255,150,60,.55),transparent 70%),radial-gradient(35% 45% at 100% 80%,rgba(255,80,120,.35),transparent 70%);
 animation:leak var(--dur) ease-in-out both}
@keyframes leak{0%{opacity:0}25%{opacity:.85}60%{opacity:.25}100%{opacity:.6}}
.bars:before,.bars:after{content:"";position:absolute;left:0;right:0;height:7vmin;background:#000;z-index:46}
.bars:before{top:0}.bars:after{bottom:0}.reel .bars:before,.reel .bars:after{display:none}
/* type */
.content{position:absolute;inset:0;padding:8vmin 9vmin;display:flex;flex-direction:column;justify-content:center;z-index:5}
h1,h2{font-family:var(--head);font-weight:$weight;letter-spacing:-.015em;margin:0;line-height:1.08}
.deva h1,.deva h2{line-height:1.32;letter-spacing:0}
h1{font-size:8.6vmin}h2{font-size:6.4vmin;margin-bottom:4vmin}
.reel h1{font-size:9.4vmin}.reel h2{font-size:7vmin}
.kicker{font:700 2.6vmin var(--font);letter-spacing:.22em;text-transform:uppercase;color:var(--accent);margin-bottom:2.4vmin}
.deva .kicker{letter-spacing:.04em}
.chip{display:inline-block;align-self:flex-start;padding:1.1vmin 2.4vmin;border-radius:99vmin;background:var(--card);
 border:.25vmin solid color-mix(in srgb,var(--accent) 60%,transparent);color:var(--fg);font:600 3vmin var(--font);margin-bottom:3.5vmin}
.tag{font-size:3.9vmin;color:var(--muted);margin:3.5vmin 0 0;max-width:80%;line-height:1.35}
.reel .tag{max-width:100%}
ul{list-style:none;margin:0;padding:0}
li{font-size:4.1vmin;line-height:1.35;margin:0 0 2.6vmin;padding-left:5.5vmin;position:relative}
li::before{content:"";position:absolute;left:0;top:1.5vmin;width:2.4vmin;height:2.4vmin;border-radius:.6vmin;background:var(--accent)}
.split{flex-direction:row;align-items:center;gap:6vmin}
.split .text{flex:1.15;min-width:0}.split .pic{flex:1;height:74%;border-radius:2.4vmin;overflow:hidden;position:relative;
 box-shadow:0 3vmin 7vmin rgba(0,0,0,.35)}
.reel .split{flex-direction:column-reverse;justify-content:flex-end;padding-top:14vmin}
.reel .split .pic{flex:none;width:100%;height:44vmin;min-height:44vmin}.reel .split .text{flex:none;width:100%}
.lower{justify-content:flex-end;padding-bottom:24vmin}
.lower h2{font-size:7vmin;text-shadow:0 .5vmin 3vmin rgba(0,0,0,.5);max-width:88%}
.reel .lower{padding-bottom:58vmin}
.reel .content:not(.lower):not(.center){padding-bottom:46vmin;justify-content:flex-start;padding-top:16vmin}
.reel .stat{padding:3vmin 4vmin}.reel .stat b{font-size:9vmin}
.light .lower h2{text-shadow:none}
.stats{display:flex;gap:4vmin;flex-wrap:wrap}.reel .stats{flex-direction:column}
.stat{flex:1;min-width:30vmin;max-width:100%;overflow:hidden;background:var(--card);border-radius:3vmin;padding:4.5vmin 4vmin}
.stat b{display:block;font:$weight 11vmin var(--head);color:var(--accent);letter-spacing:-.03em;line-height:1.05}
.stat b.word{font-size:6.4vmin;line-height:1.2;overflow-wrap:anywhere}
.stat span{display:block;font-size:3.2vmin;color:var(--muted);margin-top:1.6vmin}
.win{background:#0d1117;color:#e6edf3;border-radius:2.4vmin;overflow:hidden;box-shadow:0 3vmin 7vmin rgba(0,0,0,.35)}
.win .bar{height:5vmin;background:#161b22;display:flex;align-items:center;gap:1.4vmin;padding:0 2.4vmin}
.win .bar i{width:1.8vmin;height:1.8vmin;border-radius:50%;background:#ff5f56}.win .bar i:nth-child(2){background:#ffbd2e}.win .bar i:nth-child(3){background:#27c93f}
pre{margin:0;padding:3.6vmin 4vmin;font:3.4vmin/1.45 'Noto Sans Mono','DejaVu Sans Mono',monospace;white-space:pre-wrap;word-break:break-word;min-height:24vmin}
pre .cur{display:inline-block;width:1.4vmin;height:3.2vmin;background:var(--accent);vertical-align:-.5vmin;margin-left:.3vmin}
.qmark{font:900 22vmin/0.7 Georgia,serif;color:var(--accent);height:12vmin}
.quote{font:500 5.4vmin/1.35 var(--head);max-width:92%}
.reel .quote{font-size:6vmin}
.src{color:var(--muted);font-size:3vmin;margin-top:4vmin}
.center{align-items:center;text-align:center}.center .chip{align-self:center}.center .tag{margin-left:auto;margin-right:auto}
.url{font:600 3.2vmin 'Noto Sans Mono','DejaVu Sans Mono',monospace;color:var(--accent);margin-top:4vmin;word-break:break-all}
.cap{position:absolute;left:50%;bottom:7vmin;transform:translateX(-50%);z-index:50;max-width:86%;text-align:center;
 font:700 3.6vmin/1.35 var(--font);color:#fff;background:rgba(0,0,0,.66);padding:1.3vmin 2.6vmin;border-radius:1.6vmin;opacity:0}
.reel .cap{bottom:30vmin;font-size:4.6vmin;max-width:84%}
.broadcast:not(.reel) .cap{bottom:11vmin}.broadcast.reel .cap{bottom:34vmin}
.num{position:absolute;right:5vmin;top:4.5vmin;font:700 2.4vmin var(--font);color:var(--muted);z-index:20;letter-spacing:.15em}
/* broadcast furniture */
.bug{position:absolute;left:5vmin;top:4.5vmin;z-index:30;display:flex;gap:1.2vmin;align-items:center}
.bug .live{background:var(--accent);color:#fff;font:800 2.5vmin var(--font);padding:.8vmin 1.8vmin;border-radius:.6vmin;letter-spacing:.06em}
.bug .site{background:rgba(0,0,0,.55);color:#fff;font:600 2.4vmin var(--font);padding:.8vmin 1.6vmin;border-radius:.6vmin}
.l3{position:absolute;left:5vmin;right:5vmin;bottom:25vmin;z-index:20}
.reel .l3{bottom:62vmin}
.l3 .k{display:inline-block;background:var(--accent);color:#fff;font:800 2.8vmin var(--font);padding:1vmin 2vmin;border-radius:.6vmin .6vmin 0 0}
.l3 .h{background:var(--panel);color:#fff;font:$weight 5.6vmin/1.3 var(--head);padding:2vmin 2.8vmin;border-left:1.1vmin solid var(--accent);
 border-radius:0 .8vmin .8vmin .8vmin}
.reel .l3 .h{font-size:6.2vmin}
.l3 .s{background:rgba(255,255,255,.92);color:#111;font:600 3vmin/1.35 var(--font);padding:1.2vmin 2.8vmin;display:inline-block;margin-top:.8vmin;border-radius:.6vmin}
.ticker{position:absolute;left:0;right:0;bottom:0;height:8vmin;z-index:35;display:flex;align-items:center;overflow:hidden;
 background:var(--accent2);color:#111;font:700 3vmin var(--font)}
.reel .ticker{bottom:20vmin}
.ticker .lab{flex:none;height:100%;display:flex;align-items:center;padding:0 2.4vmin;background:#111;color:#fff;z-index:2}
.ticker .run{white-space:nowrap;padding-left:4vmin;animation:tick var(--dur) linear both}
@keyframes tick{from{transform:translateX(30%)}to{transform:translateX(-60%)}}
""")

JS = """
const CAPS = __CAPS__, COUNTS = [...document.querySelectorAll('[data-count]')], TYPE = document.querySelector('[data-type]');
const capEl = document.querySelector('.cap');
function fmtNum(v, dec){ return dec ? v.toFixed(dec) : Math.round(v).toLocaleString('en-US'); }
window.__seek = function(t){
  document.getAnimations().forEach(a => { a.pause(); a.currentTime = t * 1000; });
  if (capEl) { const c = CAPS.find(c => t >= c[0] && t < c[1]); capEl.textContent = c ? c[2] : ''; capEl.style.opacity = c ? 1 : 0; }
  COUNTS.forEach((el, i) => {
    const target = parseFloat(el.dataset.count), dec = +el.dataset.dec, f = Math.min(1, Math.max(0, (t - 0.6 - i * 0.2) / 1.4));
    el.textContent = fmtNum(target * (1 - Math.pow(1 - f, 3)), dec) + el.dataset.suffix;
  });
  if (TYPE) { const full = TYPE.dataset.type, n = Math.floor(Math.max(0, t - 0.9) * __CPS__);
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
    """[[start, end, text], ...] spread over the narration's duration, weighted by length."""
    words = (text or "").split()
    if not words or dur <= 0:
        return []
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?;:।॥]$", w) and len(cur) >= 3:
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
    """'12.3k' -> (12.3, 1, 'k'); 'Python' -> None. Devanagari digits are kept as words."""
    m = re.fullmatch(r"\s*([\d,]+(?:\.\d+)?)\s*([kKmMbB%+]*)\s*", str(value or ""))
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    return float(raw), (len(raw.split(".")[1]) if "." in raw else 0), m.group(2)


def _url(path):
    return "file://" + os.path.abspath(path)


def photo_layer(ph, idx, fmt="landscape"):
    """ph: {"path", "fit": "cover"|"contain", "w", "h", "credit"}; idx picks the camera move. A photo whose shape does
    not suit the frame becomes a band: across the top of a reel (text goes below it) or down the right of a landscape
    frame (text goes left of it), so text never sits on the busy part of the picture."""
    if not ph or not ph.get("path"):
        return ""
    u = esc(_url(ph["path"]))
    cls, size = "img", ""
    if ph.get("fit") == "contain":
        pa = ph.get("w", 1) / float(ph.get("h", 1) or 1)
        cls = "img band"
        size = ("height:%.1fvmin;" % max(46, min(70, 100 / pa)) if fmt == "reel"
                else "width:%.1fvmin;" % max(56, min(100, 100 * pa)))
    return ('<div class="ph"><div class="fill" style="background-image:url(\'%s\')"></div>'
            '<div class="%s" data-bg="%s" style="%sbackground-image:url(\'%s\');animation-name:%s"></div>'
            '<div class="vig"></div><div class="shade"></div></div>') % (
        u, cls, u, size, u, ("kbin", "kbleft", "kbout", "kbright")[idx % 4])


def words(text, start=0.25, step=0.07, limit=1.6):
    """Heading split into words that rise in one after another (kinetic type)."""
    ws = str(text or "").split()
    step = min(step, limit / max(1, len(ws)))
    return "".join('<span class="w" style="animation-delay:%.2fs">%s</span>%s' % (start + i * step, esc(w), " " if i < len(ws) - 1 else "")
                   for i, w in enumerate(ws))


def scene_html(scene, idx, n, story, style, fmt, dur, photo, captions, p0, p1):
    style = style if style in STYLES else "midnight"
    lang = (story.get("language") or "en").split("-")[0]
    L = dict(LABELS["en"], **LABELS.get(lang, {}))
    src = story.get("source") or {}
    facts = src.get("facts") or {}
    site = str(facts.get("site") or "")
    lay = {"image": "photo"}.get(scene["layout"], scene["layout"])
    h = scene.get("heading") or ""
    broadcast = style == "broadcast"
    bg = '<div class="blob a"></div><div class="blob b"></div>'
    credit = ""
    if photo and photo.get("credit"):
        credit = '<div class="credit">%s: %s</div>' % (esc(L["photo"]), esc(photo["credit"]))
    kicker = scene.get("kicker") or ""
    body = ""
    if lay in ("title", "headline"):
        bg += photo_layer(photo, idx, fmt)
        if broadcast:
            body = ('<div class="l3"><div class="k slide" style="animation-delay:.2s">%s</div><div class="h slide" style="animation-delay:.35s">%s</div>%s</div>') % (
                esc(kicker or L["latest"]), words(h, .5), '<div class="s up" style="animation-delay:1.1s">%s</div>' % esc(story.get("tagline")) if story.get("tagline") else "")
        else:
            label = {"github": "GitHub · " + str(src.get("title") or ""), "youtube": "YouTube", "pdf": "PDF"}.get(src.get("kind"), site)
            body = ('<div class="content center"><div class="plate up">%s<h1>%s</h1>%s</div></div>' if photo else
                    '<div class="content center">%s<h1>%s</h1>%s</div>') % (
                '<div class="chip up" style="animation-delay:.15s">%s</div>' % esc(kicker or label) if (kicker or label) else "", words(h, .3),
                '<p class="tag up" style="animation-delay:.9s">%s</p>' % esc(story.get("tagline")) if story.get("tagline") else "")
    elif lay == "bullets":
        lis = "".join('<li class="up" style="animation-delay:%.2fs">%s</li>' % (0.8 + i * 0.45, esc(b))
                      for i, b in enumerate(scene.get("bullets") or []))
        text = '<div class="text"><div class="kicker up" style="animation-delay:.1s">%s</div><h2>%s</h2><ul>%s</ul></div>' % (
            esc(kicker) or "%02d" % idx, words(h, .25), lis)
        pic = '<div class="pic pop" style="animation-delay:.35s">%s</div>' % photo_layer(dict(photo, fit="cover"), idx) if photo else ""
        body = '<div class="content %s">%s%s</div>' % ("split" if photo else "", text, pic)
    elif lay == "photo":
        bg += photo_layer(photo, idx, fmt)
        sub = (scene.get("bullets") or [""])[0]
        if broadcast:
            body = '<div class="l3"><div class="k slide" style="animation-delay:.2s">%s</div><div class="h slide" style="animation-delay:.3s">%s</div>%s</div>' % (
                esc(kicker or site or L["latest"]), words(h, .45), '<div class="s up" style="animation-delay:1s">%s</div>' % esc(sub) if sub else "")
        else:
            body = '<div class="content lower"><div class="kicker up" style="animation-delay:.2s">%s</div><h2>%s</h2>%s</div>' % (
                esc(kicker) or "%02d" % idx, words(h, .4), '<p class="tag up" style="animation-delay:1s">%s</p>' % esc(sub) if sub else "")
    elif lay == "stats":
        cards = ""
        for i, s in enumerate(scene.get("stats") or []):
            num = _num(s.get("value"))
            val = ('<b data-count="%s" data-dec="%d" data-suffix="%s">0</b>' % (num[0], num[1], esc(num[2])) if num
                   else '<b class="word">%s</b>' % esc(s.get("value")))
            cards += '<div class="stat pop" style="animation-delay:%.2fs">%s<span>%s</span></div>' % (0.4 + i * 0.2, val, esc(s.get("label")))
        body = '<div class="content"><h2>%s</h2><div class="stats">%s</div></div>' % (words(h or L["numbers"], .15), cards)
    elif lay == "code":
        body = ('<div class="content"><h2>%s</h2><div class="win pop" style="animation-delay:.4s"><div class="bar"><i></i><i></i><i></i></div>'
                '<pre data-type="%s"></pre></div></div>') % (words(h, .15), esc(scene.get("code")))
    elif lay == "quote":
        bg += photo_layer(photo, idx, fmt)
        body = ('<div class="content qt"><div class="qmark up">&ldquo;</div><div class="quote up" style="animation-delay:.3s">%s</div>'
                '<div class="src up" style="animation-delay:.9s">%s</div></div>') % (esc(scene.get("quote")), esc(scene.get("heading") or src.get("title")))
    else:  # outro
        bg += photo_layer(photo, idx, fmt)
        url = re.sub(r"^https?://(www\.)?", "", str(src.get("url") or "")).rstrip("/")
        if src.get("kind") != "github":
            url = url.split("/")[0]                       # articles: just the site
        tag = (scene.get("bullets") or [""])[0] if scene.get("bullets") else (story.get("tagline") or "")
        body = ('<div class="content center"><div class="plate up"><h1>%s</h1>%s%s</div></div>' if photo else
                '<div class="content center"><h1>%s</h1>%s%s</div>') % (
            words(h or L["thanks"], .2), '<p class="tag up" style="animation-delay:.8s">%s</p>' % esc(tag) if tag else "",
            '<div class="url up" style="animation-delay:1.1s">%s: %s</div>' % (esc(L["source"]), esc(url[:70])) if url else "")
    furniture = ""
    if broadcast:
        furniture = '<div class="bug"><span class="live">%s</span>%s</div>' % (esc(L["latest"]), '<span class="site">%s</span>' % esc(site) if site else "")
        if lay not in ("title", "headline", "outro"):
            furniture += '<div class="ticker"><div class="lab">%s</div><div class="run">%s</div></div>' % (
                esc(site or L["latest"]), esc(" • ".join(x for x in (story.get("title"), story.get("tagline")) if x)))
    texture = ""
    if style == "documentary":
        texture = '<div class="leak"></div><div class="grain"></div>'
    elif style in ("broadcast", "midnight", "neon") and photo:
        texture = '<div class="grain" style="opacity:.07"></div>'
    onphoto = bool(photo) and lay in ("title", "headline", "photo", "quote", "outro")
    pc = onphoto and photo.get("fit") == "contain"
    var = ""
    if pc:                                              # where the text starts, beside or below the band photo
        pa = photo.get("w", 1) / float(photo.get("h", 1) or 1)
        var = ("--below:%.1fvmin" % (13 + max(46, min(70, 100 / pa)) + 6) if fmt == "reel"
               else "--beside:%.1fvmin" % (max(56, min(100, 100 * pa)) + 8))
    cls = " ".join(filter(None, [fmt, style, "onphoto" if onphoto else "", "pc" if pc else "", "light" if style in LIGHT else "", "deva" if lang in DEVANAGARI_LANGS else "",
                                 "dim" if lay in ("title", "outro", "quote") and not broadcast else "",
                                 "bars" if style == "documentary" else "", "first" if idx == 0 else "", "last" if idx == n - 1 else ""]))
    st = dict(STYLES[style], dur="%.3f" % dur, p0="%.3f" % p0, p1="%.3f" % p1, grain=GRAIN, kb="kbin")
    cps = max(18.0, len(scene.get("code") or "") / max(1.0, dur - 2.2))
    transition = '<div class="wipe"></div>' if broadcast else '<div class="veil"></div>'
    num = '<div class="num">%d / %d</div>' % (idx + 1, n) if not broadcast else ""
    return ('<!doctype html><html class="%s" style="%s" lang="%s"><head><meta charset="utf-8"><style>%s</style></head><body>'
            '<div class="stage">%s</div>%s%s%s%s%s%s<div class="progress"></div>%s'
            '<script>%s</script></body></html>') % (
        cls, var, esc(lang), CSS.substitute(st), bg, body, furniture, credit, texture, num,
        '<div class="cap"></div>' if captions else "", transition,
        JS.replace("__CAPS__", json.dumps(captions or [], ensure_ascii=False)).replace("__CPS__", "%.2f" % cps))
