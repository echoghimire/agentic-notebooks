"""Turns any input into source material (text, facts, photos, language) for the storyboard writer.

fetch_source(text) accepts:
- a GitHub repository URL      -> description, stats, topics, languages, file tree, README, README screenshots
- a YouTube URL                -> title, channel, description and subtitles (yt-dlp, no video download), thumbnail
- a PDF link                   -> text and embedded images (pymupdf)
- any other web page / article -> main text (trafilatura when installed), title, date, photos with captions;
                                  falls back to a headless-browser fetch for sites that block plain downloads
- anything else                -> treated as a plain prompt
Returns {"kind", "url", "title", "description", "text", "facts", "images", "language", "instructions"}.
images: [{"url", "alt", "caption", "score"}] best first (downloading and checking them is vs_server's job).
"""
import html
import html.parser
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
MAX_TEXT = 14000
TIMEOUT = 30
NE_MARKERS = ("छ", "छन्", "गरेको", "भएको", "हुन्छ", "गर्न", "पनि", "थियो", "भने", "रहेको", "गरी", "लागि", "जना", "सम्म", "बाट")
HI_MARKERS = ("है", "हैं", "में", "का", "की", "के", "और", "था", "थी", "गया", "किया", "रहा", "लिए", "साथ", "बाद")
BAD_IMG = re.compile(r"logo|icon|avatar|sprite|placeholder|blank|pixel|spacer|tracking|gravatar|emoji|badge|banner|"
                     r"/ads?[/_-]|advert|sponsor|facebook\.com|twitter\.com|\.svg(\?|$)|\.gif(\?|$)|data:image", re.I)


def _get(url, headers=None, limit=25 << 20):
    req = urllib.request.Request(url, headers=dict({"User-Agent": UA, "Accept-Language": "ne,en;q=0.8,hi;q=0.6",
                                                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}, **(headers or {})))
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = r.read(limit + 1)
        if len(data) > limit:
            raise ValueError("the page is larger than %d MB" % (limit >> 20))
        return data, r.headers.get("Content-Type", ""), r.geturl()


def browser_fetch(url):
    """Renders the page in headless Chromium (for sites that block scripts or need JavaScript)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None, args=["--no-sandbox"])
        page = b.new_page(user_agent=UA, locale="ne-NP", viewport={"width": 1366, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        page.mouse.wheel(0, 6000)                       # trigger lazy-loaded images
        page.wait_for_timeout(1200)
        raw, final = page.content(), page.url
        b.close()
    return raw, final


def _clip(text, n=MAX_TEXT):
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text or "")).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def _human(n):
    n = int(n or 0)
    return "%.1fk" % (n / 1000) if n >= 1000 else str(n)


def detect_language(text, hint=None):
    """Two-letter code from the script (Devanagari / Latin / ...) plus Nepali-vs-Hindi word markers."""
    sample = (text or "")[:6000]
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return (hint or "en")[:2]
    deva = sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)
    if deva > 0.3:
        words = re.findall(r"[ऀ-ॿ]+", sample)
        ne = sum(1 for w in words if w in NE_MARKERS)
        hi = sum(1 for w in words if w in HI_MARKERS)
        return "hi" if hi > ne * 1.3 else "ne"
    for lo, hi_, code in (("ঀ", "৿", "bn"), ("஀", "௿", "ta"), ("؀", "ۿ", "ur"),
                          ("一", "鿿", "zh"), ("぀", "ヿ", "ja")):
        if sum(1 for c in letters if lo <= c <= hi_) / len(letters) > 0.3:
            return code
    h = (hint or "").lower()[:2]
    return h if h and h.isalpha() and h not in ("ne", "hi") else "en"


# ---------------------------------------------------------------------- GitHub
GH = re.compile(r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:[/?#].*)?$")


def github(url):
    m = GH.match(url)
    owner, repo = m.group(1), m.group(2)
    hdr = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        hdr["Authorization"] = "Bearer " + token
    api = "https://api.github.com/repos/%s/%s" % (owner, repo)
    try:
        info = json.loads(_get(api, hdr)[0])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError("GitHub repository %s/%s not found (or private)" % (owner, repo))
        if e.code == 403:
            raise ValueError("GitHub API rate limit reached; add a GITHUB_TOKEN secret or try again later")
        raise
    langs = {}
    try:
        langs = json.loads(_get(api + "/languages", hdr)[0])
    except Exception:
        pass
    readme = ""
    try:
        readme = _get(api + "/readme", dict(hdr, Accept="application/vnd.github.raw"))[0].decode("utf-8", "replace")
    except Exception:
        pass
    tree = []
    try:
        tree = [("%s/" % x["name"]) if x["type"] == "dir" else x["name"]
                for x in json.loads(_get(api + "/contents", hdr)[0])][:40]
    except Exception:
        pass
    branch = info.get("default_branch") or "HEAD"
    images = [{"url": "https://opengraph.githubassets.com/1/%s/%s" % (owner, repo), "alt": info.get("full_name"),
               "caption": "", "score": 5}]
    for m2 in re.finditer(r"!\[([^\]]*)\]\(([^)\s]+)|<img[^>]+src=[\"']([^\"']+)", readme):
        src = m2.group(2) or m2.group(3)
        if not src or BAD_IMG.search(src) or "shields.io" in src or "badge" in src:
            continue
        if not re.match(r"^https?://", src):
            src = "https://raw.githubusercontent.com/%s/%s/%s/%s" % (owner, repo, branch, src.lstrip("./"))
        images.append({"url": src, "alt": (m2.group(1) or "")[:120], "caption": "", "score": 3})
    readme = re.sub(r"<!--.*?-->", "", readme, flags=re.S)
    readme = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", readme)
    readme = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", readme)
    readme = re.sub(r"<img[^>]*>", "", readme, flags=re.I)
    total = sum(langs.values()) or 1
    top_langs = ["%s %d%%" % (k, round(100 * v / total)) for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:4]]
    facts = {"name": info.get("full_name"), "site": "GitHub", "stars": _human(info.get("stargazers_count")),
             "forks": _human(info.get("forks_count")), "language": info.get("language"), "languages": top_langs,
             "license": (info.get("license") or {}).get("spdx_id"), "topics": info.get("topics", [])[:8],
             "homepage": info.get("homepage") or None, "created": (info.get("created_at") or "")[:10],
             "updated": (info.get("pushed_at") or "")[:10], "open_issues": info.get("open_issues_count")}
    text = "Repository: %s\nDescription: %s\nStars: %s, forks: %s\nLanguages: %s\nTopics: %s\nLicense: %s\n" \
           "Top-level files: %s\n\nREADME:\n%s" % (
               info.get("full_name"), info.get("description") or "", facts["stars"], facts["forks"],
               ", ".join(top_langs), ", ".join(facts["topics"]), facts["license"], ", ".join(tree), readme)
    return {"kind": "github", "url": info.get("html_url", url), "title": info.get("full_name") or repo,
            "description": info.get("description") or "", "text": _clip(text), "facts": facts, "images": images[:10],
            "language": detect_language(readme, "en")}


# ---------------------------------------------------------------------- YouTube
YT = re.compile(r"^https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch|shorts/|live/)|youtu\.be/)", re.I)


def _vtt_text(vtt):
    lines, last = [], None
    for line in vtt.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", html.unescape(line))
        if line != last:
            lines.append(line)
            last = line
    return " ".join(lines)


def youtube(url):
    if not shutil.which("yt-dlp"):
        raise ValueError("YouTube links need yt-dlp (installed by the notebook's install cell)")
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run(["yt-dlp", "--skip-download", "--write-info-json", "--write-auto-subs", "--write-subs",
                            "--sub-langs", "ne.*,ne,hi.*,en.*,en", "--sub-format", "vtt", "--no-playlist", "-o",
                            os.path.join(d, "v.%(ext)s"), url], capture_output=True, text=True, timeout=180)
        info_path = os.path.join(d, "v.info.json")
        if not os.path.exists(info_path):
            raise ValueError("could not read that YouTube video: %s" % (p.stderr.strip().splitlines() or ["?"])[-1])
        info = json.load(open(info_path, encoding="utf-8"))
        subs = ""
        for f in sorted(os.listdir(d)):
            if f.endswith(".vtt"):
                subs = _vtt_text(open(os.path.join(d, f), encoding="utf-8", errors="replace").read())
                break
    facts = {"channel": info.get("channel") or info.get("uploader"), "site": "YouTube", "views": _human(info.get("view_count")),
             "duration": "%d:%02d" % divmod(int(info.get("duration") or 0), 60), "date": info.get("upload_date")}
    text = "Video: %s\nChannel: %s\n\nDescription:\n%s\n\nTranscript:\n%s" % (
        info.get("title"), facts["channel"], info.get("description") or "", subs or "(no subtitles)")
    thumbs = sorted((t for t in info.get("thumbnails") or [] if t.get("width")), key=lambda t: -t["width"])
    images = [{"url": (thumbs[0]["url"] if thumbs else info.get("thumbnail")), "alt": info.get("title"), "caption": "", "score": 4}]
    return {"kind": "youtube", "url": url, "title": info.get("title") or "YouTube video",
            "description": (info.get("description") or "")[:300], "text": _clip(text), "facts": facts,
            "images": [i for i in images if i["url"]], "language": detect_language(text, info.get("language"))}


# ---------------------------------------------------------------------- web pages
class _Page(html.parser.HTMLParser):
    """One pass over the page: metadata, JSON-LD, readable text (fallback) and candidate photos."""
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "button", "iframe",
            "template", "select"}
    BLOCK = {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
             "tr", "td", "main", "figcaption", "dd", "dt"}
    ARTICLE_HINT = re.compile(r"(^|[\s_-])(content|entry|post|story|article|news|detail|single|body|main)", re.I)

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.out, self.skip, self.meta, self.title, self.in_title, self.lang = [], 0, {}, "", False, ""
        self.ld, self.in_ld, self.imgs, self.depth_article, self.in_cap, self.cap = [], False, [], 0, False, ""
        self.stack = []

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "html":
            self.lang = a.get("lang", "")
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])
        elif tag == "title":
            self.in_title = True
        elif tag == "script" and "ld+json" in a.get("type", ""):
            self.in_ld = True
            self.ld.append("")
            return
        if tag in ("article", "main", "figure") or self.ARTICLE_HINT.search(a.get("class", "") + " " + a.get("id", "")):
            self.depth_article += 1
            self.stack.append(tag)
        else:
            self.stack.append(None)
        if tag == "figcaption":
            self.in_cap, self.cap = True, ""
        if tag in ("img", "source"):
            src = a.get("data-src") or a.get("data-lazy-src") or a.get("data-original") or a.get("data-full-url") or ""
            srcset = a.get("data-srcset") or a.get("srcset") or ""
            if srcset:
                best = max((p.strip().split(" ") for p in srcset.split(",") if p.strip()),
                           key=lambda p: int(re.sub(r"\D", "", p[1]) or 0) if len(p) > 1 else 0, default=None)
                if best and (not src or len(best) > 1):
                    src = best[0]
            src = src or a.get("src", "")
            if src and not src.startswith("data:"):
                w = int(re.sub(r"\D", "", a.get("width", "")) or 0)
                self.imgs.append({"url": urllib.parse.urljoin(self.base, src), "alt": a.get("alt", "")[:200], "caption": "",
                                  "score": (2 if self.depth_article else 0) + (1 if len(a.get("alt", "")) > 15 else 0)
                                  - (3 if 0 < w < 300 else 0)})
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
            if tag in ("h1", "h2", "h3"):
                self.out.append("## ")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self.in_ld:
            self.in_ld = False
            return
        if tag == "figcaption":
            self.in_cap = False
            if self.imgs and not self.imgs[-1]["caption"]:
                self.imgs[-1]["caption"] = " ".join(self.cap.split())[:240]
        if self.stack:
            if self.stack.pop():
                self.depth_article = max(0, self.depth_article - 1)
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.in_ld:
            self.ld[-1] += data
        elif self.in_title:
            self.title += data
        else:
            if self.in_cap:
                self.cap += data
            if not self.skip:
                self.out.append(data)


def _ld_items(blobs):
    for b in blobs:
        try:
            d = json.loads(b.strip())
        except ValueError:
            continue
        stack = [d]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, dict):
                if "@graph" in x:
                    stack.extend(x["@graph"] if isinstance(x["@graph"], list) else [x["@graph"]])
                yield x


def _ld_images(item):
    im = item.get("image")
    for v in (im if isinstance(im, list) else [im]):
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict) and v.get("url"):
            yield v["url"]


def wp_original(url):
    """WordPress thumbnails (photo-300x200.jpg) -> the full-size original (photo.jpg)."""
    return re.sub(r"-\d{2,4}x\d{2,4}(?=\.(jpe?g|png|webp)(\?|$))", "", url, flags=re.I)


def harvest_images(p, final, trafi_images=()):
    cands = []
    for key, score in (("og:image", 6), ("og:image:url", 6), ("twitter:image", 5), ("twitter:image:src", 5), ("image", 4)):
        if p.meta.get(key):
            cands.append({"url": urllib.parse.urljoin(final, p.meta[key]), "alt": p.meta.get("og:image:alt", ""), "caption": "", "score": score})
    for item in _ld_items(p.ld):
        if str(item.get("@type", "")).lower() in ("newsarticle", "article", "blogposting", "reportagenewsarticle", "webpage"):
            for u in _ld_images(item):
                cands.append({"url": urllib.parse.urljoin(final, u), "alt": item.get("headline", "")[:200], "caption": "", "score": 5})
    for u in trafi_images:
        cands.append({"url": urllib.parse.urljoin(final, u), "alt": "", "caption": "", "score": 4})
    cands += p.imgs
    seen, out = {}, []
    for c in cands:
        c["url"] = wp_original(html.unescape(c["url"]).strip())
        if not re.match(r"^https?://", c["url"]) or BAD_IMG.search(c["url"]):
            continue
        key = re.sub(r"[?#].*$", "", c["url"])
        if key in seen:
            prev = seen[key]
            prev["score"] = max(prev["score"], c["score"]) + 1
            prev["caption"] = prev["caption"] or c["caption"]
            prev["alt"] = prev["alt"] or c["alt"]
            continue
        seen[key] = c
        out.append(c)
    out = [c for c in out if c["score"] >= 0]          # tiny or off-article pictures (sidebars, related news)
    out.sort(key=lambda c: -c["score"])
    return out[:14]


def _trafilatura(raw, url):
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        doc = trafilatura.bare_extraction(raw, url=url, with_metadata=True, include_images=True, include_comments=False,
                                          include_tables=True, favor_precision=True)
    except TypeError:
        doc = trafilatura.bare_extraction(raw, url=url, include_images=True, include_comments=False)
    if doc is None:
        return None
    if not isinstance(doc, dict):
        doc = doc.as_dict() if hasattr(doc, "as_dict") else vars(doc)
    return doc


def webpage(url):
    raw = final = ctype = None
    try:
        data, ctype, final = _get(url)
        if "pdf" in ctype.lower() or final.lower().split("?")[0].endswith(".pdf") or data[:5] == b"%PDF-":
            return pdf(final, data)
        if ctype and not any(t in ctype.lower() for t in ("html", "xml", "text")):
            raise ValueError("unsupported content type %r at that link" % ctype)
        enc = re.search(r"charset=([\w-]+)", ctype or "")
        raw = data.decode(enc.group(1) if enc else "utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        blocked = e
    else:
        blocked = None
    text, doc, p = "", None, None
    if raw:
        doc = _trafilatura(raw, final)
        p = _Page(final)
        p.feed(raw)
        p.close()
        text = (doc or {}).get("text") or ""
    if len(text.split()) < 60:                         # blocked, JavaScript-only, or a poor extraction: render it
        try:
            raw2, final2 = browser_fetch(url)
            doc2 = _trafilatura(raw2, final2)
            p2 = _Page(final2)
            p2.feed(raw2)
            p2.close()
            t2 = (doc2 or {}).get("text") or ""
            if len(t2.split()) > len(text.split()):
                raw, final, doc, p, text = raw2, final2, doc2, p2, t2
        except Exception as e:
            if raw is None:
                raise ValueError("could not open that link (%s; the browser fallback also failed: %s)" % (
                    blocked or "no content", str(e)[:200]))
    if not text and p is not None:                     # no trafilatura: our own readable-text pass
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in "".join(p.out).splitlines()]
        text = "\n".join(ln for ln in lines if ln.startswith("## ") or len(ln.split()) >= 6)
    if len(text.split()) < 30:
        raise ValueError("that page has almost no readable text (it may need a login)")
    doc = doc or {}
    meta = p.meta if p else {}
    title = doc.get("title") or meta.get("og:title") or " ".join((p.title if p else "").split()) or urllib.parse.urlparse(final).netloc
    desc = doc.get("description") or meta.get("og:description") or meta.get("description") or ""
    site = doc.get("sitename") or meta.get("og:site_name") or urllib.parse.urlparse(final).netloc.replace("www.", "")
    date = doc.get("date") or meta.get("article:published_time", "")[:10]
    trafi_imgs = [u for u in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)] + ([doc["image"]] if doc.get("image") else [])
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    lang = detect_language(title + "\n" + text, (p.lang if p else "") or doc.get("language"))
    return {"kind": "article", "url": final, "title": html.unescape(title)[:200], "description": html.unescape(desc)[:400],
            "text": _clip("Title: %s\nSite: %s\nDate: %s\nAuthor: %s\nSummary: %s\n\n%s" % (
                title, site, date, doc.get("author") or "", desc, text)),
            "facts": {"site": site, "date": date, "author": doc.get("author"), "words": len(text.split())},
            "images": harvest_images(p, final, trafi_imgs) if p else [], "language": lang}


def pdf(url, data):
    try:
        import pymupdf
    except ImportError:
        raise ValueError("PDF links need pymupdf (installed by the notebook's install cell)")
    doc = pymupdf.open(stream=data, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    title = (doc.metadata or {}).get("title") or os.path.basename(urllib.parse.urlparse(url).path) or "Document"
    images = []
    for page in doc:
        for im in page.get_images(full=True)[:3]:
            try:
                info = doc.extract_image(im[0])
            except Exception:
                continue
            if info and info.get("width", 0) >= 500 and info.get("height", 0) >= 300:
                images.append({"bytes": info["image"], "ext": info.get("ext", "png"), "alt": "", "caption": "", "score": 3})
        if len(images) >= 6:
            break
    pages = len(doc)
    doc.close()
    if len(text.split()) < 40:
        raise ValueError("that PDF has no text layer (scanned?)")
    return {"kind": "pdf", "url": url, "title": title[:200], "description": "", "text": _clip(text),
            "facts": {"pages": pages, "site": urllib.parse.urlparse(url).netloc.replace("www.", "")},
            "images": images, "language": detect_language(text)}


# ---------------------------------------------------------------------- entry point
def fetch_source(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("give a link or describe the video")
    first = text.split()[0]
    if re.match(r"^https?://", first):
        url, extra = first, text[len(first):].strip()
        if GH.match(url):
            src = github(url)
        elif YT.match(url):
            src = youtube(url)
        else:
            src = webpage(url)
        src["instructions"] = extra
        return src
    return {"kind": "prompt", "url": None, "title": text[:80], "description": "", "text": _clip(text),
            "facts": {}, "images": [], "language": detect_language(text), "instructions": ""}
