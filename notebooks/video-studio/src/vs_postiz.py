"""Posting finished videos to social channels through Postiz (open source social scheduler, https://postiz.com).

Postiz holds the connections to YouTube, TikTok, Instagram, Facebook, X, LinkedIn, Threads, Bluesky... The studio
only needs a Postiz API key (Postiz -> Settings -> Public API), kept in the POSTIZ_API_KEY Kaggle secret, and the API
address: https://api.postiz.com/public/v1 for Postiz cloud, or https://<your-postiz>/api/public/v1 when self-hosted.

Public API used (see apps/backend/src/public-api in gitroomhq/postiz-app):
  GET  /integrations      connected channels: id, name, identifier (youtube, tiktok, instagram...), picture, disabled
  POST /upload            multipart "file" -> media {id, path}
  POST /posts             {type: now|schedule|draft, date, shortLink, tags, posts: [{integration: {id},
                           value: [{content, image: [{id, path}]}], settings: {__type, ...per platform}}]}
Postiz cloud allows about 30 API requests an hour, so channels are cached and each format is uploaded once per job.
"""
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid

API = os.environ.get("POSTIZ_API_URL", "https://api.postiz.com/public/v1").rstrip("/")
KEY = os.environ.get("POSTIZ_API_KEY", "")
VERTICAL = {"tiktok", "tiktok-business", "instagram", "instagram-standalone", "youtube"}   # reels / shorts by default
_cache = {}


def configured():
    return bool(KEY)


def _req(method, path, body=None, headers=None, timeout=60):
    if not KEY:
        raise ValueError("Postiz is not connected: add your Postiz API key as the POSTIZ_API_KEY Kaggle secret")
    h = {"Authorization": KEY, "Accept": "application/json"}
    h.update(headers or {})
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        try:
            j = json.loads(detail)
            detail = j.get("msg") or j.get("message") or j.get("error") or detail
            if isinstance(detail, dict):
                detail = detail.get("error") or json.dumps(detail)
        except ValueError:
            pass
        raise ValueError("Postiz said %d: %s" % (e.code, detail))
    return json.loads(raw) if raw.strip() else {}


def channels(refresh=False):
    """Connected channels (cached for 10 minutes: the API is rate-limited)."""
    c = _cache.get("ch")
    if c and not refresh and time.time() - c[0] < 600:
        return c[1]
    out = [{"id": i["id"], "name": i.get("name") or i.get("identifier"), "identifier": i.get("identifier", ""),
            "picture": i.get("picture"), "profile": i.get("profile"),
            "format": "reel" if i.get("identifier") in VERTICAL else "landscape"}
           for i in _req("GET", "/integrations") if not i.get("disabled")]
    _cache["ch"] = (time.time(), out)
    return out


def upload(path):
    """Uploads a video (multipart field "file"); returns Postiz's media {id, path}."""
    boundary = "----vs" + uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "video/mp4"
    with open(path, "rb") as f:
        data = f.read()
    body = (("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
             % (boundary, name, ctype)).encode() + data + ("\r\n--%s--\r\n" % boundary).encode())
    media = _req("POST", "/upload", body, {"Content-Type": "multipart/form-data; boundary=" + boundary}, timeout=900)
    if not media.get("id") or not media.get("path"):
        raise ValueError("Postiz upload returned no media: %s" % str(media)[:200])
    return {"id": media["id"], "path": media["path"]}


def settings_for(identifier, title, privacy="public"):
    """The per-platform settings Postiz requires, filled with safe defaults (made-with-AI disclosed)."""
    t = (title or "Video")[:100]
    if identifier == "youtube":
        return {"__type": "youtube", "title": t, "type": privacy if privacy in ("public", "private", "unlisted") else "public"}
    if identifier in ("tiktok", "tiktok-business"):
        return {"__type": identifier, "title": t[:90],
                "privacy_level": {"private": "SELF_ONLY"}.get(privacy, "PUBLIC_TO_EVERYONE"), "duet": False,
                "stitch": False, "comment": True, "autoAddMusic": "no", "brand_content_toggle": False,
                "brand_organic_toggle": False, "video_made_with_ai": True, "content_posting_method": "DIRECT_POST"}
    if identifier in ("instagram", "instagram-standalone"):
        return {"__type": identifier, "post_type": "post"}
    if identifier == "facebook":
        return {"__type": "facebook", "post_type": "post", "title": t}
    if identifier == "x":
        return {"__type": "x", "who_can_reply_post": "everyone", "made_with_ai": True}
    if identifier in ("linkedin", "linkedin-page"):
        return {"__type": identifier, "post_as_images_carousel": False}
    return {"__type": identifier}


def caption(story, src, brand=None, limit=2000):
    """A ready caption: title, summary, source credit, the channel's handle and a few hashtags."""
    site = ((src or {}).get("facts") or {}).get("site") or ""
    lang = story.get("language") or "en"
    src_word = {"ne": "स्रोत", "hi": "स्रोत"}.get(lang, "Source")
    tags = {"news": "#news", "tech": "#tech", "education": "#learn", "story": "#story", "promo": "#new"}.get(story.get("category"), "")
    if lang == "ne":
        tags = (tags + " #Nepal #नेपाल").strip()
    lines = [story.get("title") or "", story.get("tagline") or ""]
    if site and (src or {}).get("kind") != "prompt":
        lines.append("%s: %s" % (src_word, site))
    if brand and brand.get("handle"):
        lines.append(brand["handle"])
    if tags:
        lines.append(tags)
    return "\n\n".join(x for x in lines if x)[:limit]


def post(files, targets, text, title, when=None, privacy="public", media_cache=None):
    """files: {"landscape": path, "reel": path}; targets: [{"id", "identifier", "format"?}].
    Uploads each needed format once, then creates one Postiz post per channel in a single request.
    when: None = now, else an ISO date-time (UTC) to schedule. Returns Postiz's response."""
    media_cache = {} if media_cache is None else media_cache
    items = []
    for t in targets:
        fmt = t.get("format") or ("reel" if t.get("identifier") in VERTICAL else "landscape")
        if fmt not in files:
            fmt = next(iter(files))
        if fmt not in media_cache:
            media_cache[fmt] = upload(files[fmt])
        items.append({"integration": {"id": t["id"]}, "value": [{"content": text, "image": [media_cache[fmt]]}],
                      "settings": settings_for(t.get("identifier", ""), title, privacy)})
    if not items:
        raise ValueError("pick at least one channel")
    body = {"type": "schedule" if when else "now", "shortLink": False, "tags": [], "posts": items,
            "date": when or time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())}
    return _req("POST", "/posts", body), media_cache
