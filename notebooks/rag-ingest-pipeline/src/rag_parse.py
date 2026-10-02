"""Document parsing and chunking for the RAG ingest pipeline (no GPU libraries).

parse_file() turns a file into blocks: [{"text", "page", "heading"}] in reading order.
chunk_blocks() packs blocks into overlapping chunks that keep their page range and heading.
PDF needs pymupdf, DOCX needs python-docx; everything else is the standard library.
"""
import html.parser
import json
import os
import re

KINDS = {".pdf": "pdf", ".docx": "docx", ".html": "html", ".htm": "html", ".xhtml": "html",
         ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown", ".txt": "text", ".rst": "text",
         ".csv": "text", ".tsv": "text", ".json": "text", ".jsonl": "text", ".xml": "text", ".log": "text",
         ".ipynb": "notebook", ".py": "code", ".js": "code", ".ts": "code", ".tsx": "code", ".jsx": "code",
         ".java": "code", ".go": "code", ".rs": "code", ".c": "code", ".h": "code", ".cpp": "code", ".hpp": "code",
         ".cs": "code", ".rb": "code", ".php": "code", ".kt": "code", ".swift": "code", ".scala": "code",
         ".sql": "code", ".sh": "code", ".yaml": "code", ".yml": "code", ".toml": "code", ".ini": "code"}
SUPPORTED = tuple(KINDS)


def kind_of(name, content_type=""):
    ext = os.path.splitext(name.lower().split("?")[0])[1]
    if ext in KINDS:
        return KINDS[ext]
    ct = (content_type or "").lower()
    for key, kind in (("pdf", "pdf"), ("html", "html"), ("wordprocessingml", "docx"), ("markdown", "markdown"),
                      ("json", "text"), ("text/", "text")):
        if key in ct:
            return kind
    return None


def _paras(text, page=None, heading=None, markdown=False):
    blocks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if markdown:
            m = re.match(r"^(#{1,6})\s+(.+)$", para.splitlines()[0])
            if m:
                heading = m.group(2).strip()
        blocks.append({"text": para, "page": page, "heading": heading})
    return blocks, heading


# ---------------------------------------------------------------------- formats
def _pdf(path):
    import pymupdf
    doc = pymupdf.open(path)
    blocks, empty = [], 0
    title = (doc.metadata or {}).get("title") or ""
    for i, page in enumerate(doc, 1):
        text = page.get_text("text")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # join hyphenated line breaks
        text = re.sub(r"(?<![\n.!?:])\n(?!\n)", " ", text)     # unwrap lines inside paragraphs
        if not text.strip():
            empty += 1
            continue
        b, _ = _paras(text, page=i)
        blocks += b
    pages = len(doc)
    doc.close()
    warn = ["%d of %d pages have no text layer (scanned?); run OCR first to include them" % (empty, pages)] if empty else []
    return title, blocks, pages, warn


def _docx(path):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(path)
    blocks, heading = [], None
    title = d.core_properties.title or ""
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                heading = text
                title = title or (text if style.lower() == "title" else "")
            blocks.append({"text": text, "page": None, "heading": heading})
        elif tag == "tbl":
            rows = []
            for row in Table(child, d).rows:
                cells = []
                for c in row.cells:
                    t = " ".join(c.text.split())
                    if not cells or cells[-1] != t:                    # merged cells repeat
                        cells.append(t)
                rows.append(" | ".join(cells))
            if rows:
                blocks.append({"text": "\n".join(rows), "page": None, "heading": heading})
    return title, blocks, None, []


class _HTML(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
    BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
             "blockquote", "td", "th", "dd", "dt", "table", "ul", "ol", "main", "header", "footer", "nav", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.buf, self.skip, self.title, self.in_title, self.heading, self.hbuf = [], [], 0, "", False, None, None

    def flush(self):
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.buf)).strip()
        self.buf = []
        if text:
            self.blocks.append({"text": text, "page": None, "heading": self.heading})

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.flush()
            if re.fullmatch(r"h[1-6]", tag):
                self.hbuf = []

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            if re.fullmatch(r"h[1-6]", tag) and self.hbuf is not None:
                self.heading = " ".join("".join(self.hbuf).split()) or self.heading
                self.hbuf = None
            self.flush()

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.buf.append(data)
            if self.hbuf is not None:
                self.hbuf.append(data)


def parse_html(text):
    p = _HTML()
    p.feed(text)
    p.close()
    p.flush()
    # glue tiny fragments (menu items, table cells) into paragraphs
    blocks = []
    for b in p.blocks:
        if blocks and len(blocks[-1]["text"]) < 200 and blocks[-1]["heading"] == b["heading"]:
            blocks[-1]["text"] += "\n" + b["text"]
        else:
            blocks.append(b)
    return " ".join(p.title.split()), blocks


def _notebook(text):
    nb = json.loads(text)
    blocks, heading = [], None
    for c in nb.get("cells", []):
        src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
        if c.get("cell_type") == "markdown":
            b, heading = _paras(src, heading=heading, markdown=True)
            blocks += b
        elif src.strip():
            blocks.append({"text": src.strip(), "page": None, "heading": heading})
    return blocks


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def parse_file(path, name=None, content_type=""):
    """Returns {"title", "kind", "blocks", "pages", "warnings"}."""
    name = name or os.path.basename(path)
    kind = kind_of(name, content_type)
    if kind is None:
        raise ValueError("unsupported file type: %s" % name)
    title, pages, warn = "", None, []
    if kind == "pdf":
        title, blocks, pages, warn = _pdf(path)
    elif kind == "docx":
        title, blocks, pages, warn = _docx(path)
    elif kind == "html":
        title, blocks = parse_html(read_text(path))
    elif kind == "notebook":
        blocks = _notebook(read_text(path))
    elif kind == "code":
        blocks, _ = _paras(read_text(path), heading=name)
    else:
        blocks, _ = _paras(read_text(path), markdown=(kind == "markdown"))
    if kind == "markdown" and not title:
        m = re.search(r"^#\s+(.+)$", read_text(path), re.M)
        title = m.group(1).strip() if m else ""
    return {"title": (title or os.path.splitext(name)[0]).strip()[:200], "kind": kind, "blocks": blocks,
            "pages": pages, "warnings": warn}


# ---------------------------------------------------------------------- chunking
def _pieces(text, size):
    if len(text) <= size:
        return [text]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?。！？])\s+|\n", text):
        while len(sent) > size:                                   # a single huge "sentence"
            cut = sent.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            out.append((cur + " " + sent[:cut]).strip() if cur else sent[:cut])
            cur, sent = "", sent[cut:].strip()
        if len(cur) + len(sent) + 1 > size and cur:
            out.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        out.append(cur)
    return out


def _tail(text, n):
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    t = text[-n:]
    sp = t.find(" ")
    return t[sp + 1:] if 0 <= sp < n // 2 else t


def chunk_blocks(blocks, size=1200, overlap=150):
    """Packs blocks into chunks of about ``size`` characters, overlapping by ``overlap``."""
    size, overlap = max(200, int(size)), max(0, min(int(overlap), int(size) // 2))
    chunks, cur = [], None

    def emit():
        if cur and cur["text"].strip():
            chunks.append(dict(cur, text=cur["text"].strip()))
    for b in blocks:
        for piece in _pieces(b["text"], size - overlap if overlap else size):
            if cur and len(cur["text"]) + len(piece) + 2 <= size and cur["heading"] == b["heading"]:
                cur["text"] += "\n\n" + piece
                if b["page"] is not None:
                    cur["page_end"] = b["page"]
                    cur["page_start"] = cur["page_start"] or b["page"]
                continue
            emit()
            carry = _tail(cur["text"], overlap) if cur and cur["heading"] == b["heading"] else ""
            cur = {"text": (carry + "\n\n" + piece) if carry else piece, "heading": b["heading"],
                   "page_start": b["page"], "page_end": b["page"]}
    emit()
    return chunks


def embed_text(chunk, title):
    """What gets embedded: the chunk with its document title and heading as context."""
    ctx = " - ".join(x for x in (title, chunk.get("heading")) if x)
    return (ctx + "\n" + chunk["text"]) if ctx else chunk["text"]
