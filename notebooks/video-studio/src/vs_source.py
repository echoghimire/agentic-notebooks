"""Turns any input into source material for the storyboard writer (standard library only).

fetch_source(text) accepts:
- a GitHub repository URL      -> description, stats, topics, languages, file tree, README
- a YouTube URL                -> title, channel, description and subtitles (via yt-dlp, no video download)
- a PDF link                   -> extracted text (pymupdf)
- any other web page / article -> title, description and readable text
- anything else                -> treated as a plain prompt
Returns {"kind", "url", "title", "text", "facts", "image_url"}.
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

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 video-studio"
MAX_TEXT = 14000
TIMEOUT = 30


def _get(url, headers=None, limit=20 << 20):
    req = urllib.request.Request(url, headers=dict({"User-Agent": UA}, **(headers or {})))
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = r.read(limit + 1)
        if len(data) > limit:
            raise ValueError("the page is larger than %d MB" % (limit >> 20))
        return data, r.headers.get("Content-Type", ""), r.geturl()


def _clip(text, n=MAX_TEXT):
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text or "")).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def _human(n):
    n = int(n or 0)
    return "%.1fk" % (n / 1000) if n >= 1000 else str(n)


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
    readme = re.sub(r"<!--.*?-->", "", readme, flags=re.S)
    readme = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", readme)                 # images
    readme = re.sub(r"<img[^>]*>", "", readme, flags=re.I)
    readme = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", readme)     # badges
    total = sum(langs.values()) or 1
    top_langs = ["%s %d%%" % (k, round(100 * v / total)) for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:4]]
    facts = {"name": info.get("full_name"), "stars": _human(info.get("stargazers_count")),
             "forks": _human(info.get("forks_count")), "language": info.get("language"),
             "languages": top_langs, "license": (info.get("license") or {}).get("spdx_id"),
             "topics": info.get("topics", [])[:8], "homepage": info.get("homepage") or None,
             "created": (info.get("created_at") or "")[:10], "updated": (info.get("pushed_at") or "")[:10],
             "open_issues": info.get("open_issues_count")}
    text = "Repository: %s\nDescription: %s\nStars: %s, forks: %s\nLanguages: %s\nTopics: %s\nLicense: %s\n" \
           "Top-level files: %s\n\nREADME:\n%s" % (
               info.get("full_name"), info.get("description") or "", facts["stars"], facts["forks"],
               ", ".join(top_langs), ", ".join(facts["topics"]), facts["license"], ", ".join(tree), readme)
    return {"kind": "github", "url": info.get("html_url", url), "title": info.get("full_name") or repo,
            "description": info.get("description") or "", "text": _clip(text), "facts": facts,
            "image_url": (info.get("owner") or {}).get("avatar_url")}


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
                            "--sub-langs", "en.*,en", "--sub-format", "vtt", "--no-playlist", "-o",
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
    facts = {"channel": info.get("channel") or info.get("uploader"), "views": _human(info.get("view_count")),
             "duration": "%d:%02d" % divmod(int(info.get("duration") or 0), 60), "date": info.get("upload_date")}
    text = "Video: %s\nChannel: %s\n\nDescription:\n%s\n\nTranscript:\n%s" % (
        info.get("title"), facts["channel"], info.get("description") or "", subs or "(no subtitles)")
    return {"kind": "youtube", "url": url, "title": info.get("title") or "YouTube video",
            "description": (info.get("description") or "")[:300], "text": _clip(text), "facts": facts,
            "image_url": info.get("thumbnail")}


# ---------------------------------------------------------------------- web pages
class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "button",
            "iframe", "template", "select"}
    BLOCK = {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
             "tr", "td", "main", "figcaption", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.meta, self.title, self.in_title = [], 0, {}, "", False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key in ("og:title", "og:description", "description", "og:image", "twitter:image", "og:site_name"):
                self.meta.setdefault(key, a.get("content") or "")
        elif tag == "title":
            self.in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
            if tag in ("h1", "h2", "h3"):
                self.out.append("## ")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def page_text(raw):
    p = _Text()
    p.feed(raw)
    p.close()
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in "".join(p.out).splitlines()]
    # keep paragraphs and headings; drop short menu-like fragments
    keep = [ln for ln in lines if ln.startswith("## ") or len(ln.split()) >= 6]
    return p, "\n".join(keep)


def webpage(url):
    data, ctype, final = _get(url)
    if "pdf" in ctype.lower() or final.lower().split("?")[0].endswith(".pdf") or data[:5] == b"%PDF-":
        return pdf(final, data)
    if not any(t in ctype.lower() for t in ("html", "xml", "text")) and ctype:
        raise ValueError("unsupported content type %r at that link" % ctype)
    enc = re.search(r"charset=([\w-]+)", ctype)
    raw = data.decode(enc.group(1) if enc else "utf-8", "replace")
    p, text = page_text(raw)
    if "text/plain" in ctype.lower():
        text = raw
    title = p.meta.get("og:title") or " ".join(p.title.split()) or urllib.parse.urlparse(final).netloc
    desc = p.meta.get("og:description") or p.meta.get("description") or ""
    if len(text.split()) < 40:
        raise ValueError("that page has almost no readable text (it may need JavaScript or a login)")
    site = p.meta.get("og:site_name") or urllib.parse.urlparse(final).netloc.replace("www.", "")
    image = p.meta.get("og:image") or p.meta.get("twitter:image")
    return {"kind": "article", "url": final, "title": html.unescape(title)[:200], "description": html.unescape(desc)[:400],
            "text": _clip("Title: %s\nSite: %s\nSummary: %s\n\n%s" % (title, site, desc, text)),
            "facts": {"site": site, "words": len(text.split())},
            "image_url": urllib.parse.urljoin(final, image) if image else None}


def pdf(url, data):
    try:
        import pymupdf
    except ImportError:
        raise ValueError("PDF links need pymupdf (installed by the notebook's install cell)")
    doc = pymupdf.open(stream=data, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    title = (doc.metadata or {}).get("title") or os.path.basename(urllib.parse.urlparse(url).path) or "Document"
    pages = len(doc)
    doc.close()
    if len(text.split()) < 40:
        raise ValueError("that PDF has no text layer (scanned?)")
    return {"kind": "pdf", "url": url, "title": title[:200], "description": "", "text": _clip(text),
            "facts": {"pages": pages}, "image_url": None}


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
        src["instructions"] = extra             # text after the link: "focus on the install steps", ...
        return src
    return {"kind": "prompt", "url": None, "title": text[:80], "description": "", "text": _clip(text),
            "facts": {}, "image_url": None, "instructions": ""}
