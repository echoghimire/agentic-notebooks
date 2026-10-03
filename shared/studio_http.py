"""studio_http: the small web kit behind the newer agentic-notebooks apps.

build_notebook.py copies this file into each notebook's app folder, so every notebook stays
self-contained: nothing is imported from GitHub at run time. Standard library only.

An app built on it gets:
- routes with regex paths, JSON in and out, file (with byte ranges), zip and chunked responses;
- one password for everything: browsers get a login page and a signed session cookie (no browser pop-up);
  agents send ``Authorization: Bearer <password>``, ``X-Access-Token: <password>`` or HTTP Basic.
  ``GET /health`` stays open;
- an MCP server at ``POST /mcp`` (Streamable HTTP transport, plain JSON responses, no sessions),
  and the same tools as REST: ``GET /mcp/tools`` and ``POST /mcp/tools/<name>`` with the arguments as JSON;
- an optional reverse proxy (HTTP and WebSocket) for every path no route matches;
- IPv4 and IPv6 listeners on one port (cloudflared may dial [::1] for "localhost").

The notebook cells use the helpers at the bottom: kaggle_secret(), spawn() / stop_process() with pid
files (so re-running a cell, or the whole notebook after a kernel restart, never leaves a second copy
running), wait_http(), tail(), start_tunnel() and keep_alive(). Logs go to files, never to the notebook.
"""
import base64
import hashlib
import hmac
import html as _html
import http.client
import json
import logging
import mimetypes
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

MCP_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
       "transfer-encoding", "upgrade"}
SECRET_HEADERS = {"authorization", "x-access-token"}
SESSION_COOKIE = "studio_session"
SESSION_DAYS = 30
PID_DIR = "/kaggle/working/.pids" if os.path.isdir("/kaggle/working") else "/tmp/agentic-notebooks-pids"


# ====================================================================== responses
class HTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Response:
    def __init__(self, body=b"", status=200, ctype="text/plain; charset=utf-8", headers=None):
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.status, self.ctype, self.headers = status, ctype, dict(headers or {})


class FileResponse:
    """A file from disk. ``name`` makes it a download. Single byte ranges work (audio/video seeking)."""

    def __init__(self, path, name=None, ctype=None, cache="no-cache"):
        if not os.path.isfile(path):
            raise HTTPError(404, "file not found")
        self.path, self.name, self.cache = path, name, cache
        self.ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"


class StreamResponse:
    """Chunked response from an iterable of bytes or str."""

    def __init__(self, chunks, ctype="text/plain; charset=utf-8", headers=None):
        self.chunks, self.ctype, self.headers = chunks, ctype, dict(headers or {})


class ZipResponse:
    """Streams ``folder`` (or a list of (path, arcname) pairs) as a zip download."""

    def __init__(self, files, name):
        if isinstance(files, str):
            root = os.path.dirname(os.path.abspath(files))
            pairs = []
            for dp, dn, fn in os.walk(files):
                dn.sort()
                for f in sorted(fn):
                    if not f.endswith((".tmp", ".part")):
                        pairs.append((os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), root)))
            files = pairs
        self.files, self.name = files, name


class ToolContent(list):
    """Return this from an MCP tool to send raw content blocks, for example an image."""


def text_block(text):
    return {"type": "text", "text": text}


def image_block(data, mime="image/png"):
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": mime}


def file_logger(path, name="app"):
    """A logger that writes to ``path`` only (nothing reaches the notebook output)."""
    log = logging.getLogger(name)
    path = os.path.abspath(path)
    if not any(getattr(h, "baseFilename", None) == path for h in log.handlers):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


def _disposition(name):
    ascii_name = re.sub(r'[^A-Za-z0-9_.() -]+', "_", name) or "download"
    return "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (ascii_name, quote(name))


# ====================================================================== requests
class Request:
    def __init__(self, handler, params, max_body):
        self.handler = handler
        u = urlparse(handler.path)
        self.method = handler.command
        self.path = u.path
        self.query = {k: v[-1] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
        self.params = params
        self.headers = handler.headers
        self.max_body = max_body
        self.consumed = False

    @property
    def length(self):
        try:
            return int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise HTTPError(400, "bad Content-Length")

    @property
    def chunked(self):
        return "chunked" in (self.headers.get("Transfer-Encoding") or "").lower()

    def has_body(self):
        return self.chunked or self.length > 0

    def iter_body(self, limit=None):
        """Yields the request body in pieces (Content-Length or chunked), enforcing ``limit`` bytes."""
        limit = self.max_body if limit is None else limit
        too_big = HTTPError(413, "request body too large (limit %d MB)" % (limit // 2 ** 20))
        rf = self.handler.rfile
        self.consumed = True
        if self.chunked:
            total = 0
            while True:
                line = rf.readline(1024)
                try:
                    size = int(line.split(b";")[0].strip() or b"0", 16)
                except ValueError:
                    raise HTTPError(400, "bad chunked body")
                if size == 0:
                    while rf.readline(1024) not in (b"\r\n", b"\n", b""):
                        pass
                    return
                total += size
                if total > limit:
                    raise too_big
                left = size
                while left:
                    data = rf.read(min(left, 1 << 20))
                    if not data:
                        raise HTTPError(400, "request body was cut off")
                    left -= len(data)
                    yield data
                rf.readline(8)
        else:
            left = self.length
            if left > limit:
                raise too_big
            while left:
                data = rf.read(min(left, 1 << 20))
                if not data:
                    raise HTTPError(400, "request body was cut off")
                left -= len(data)
                yield data

    def body(self):
        return b"".join(self.iter_body())

    def json(self):
        raw = self.body()
        if not raw.strip():
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            raise HTTPError(400, "body is not valid JSON")

    def save_body(self, path, limit):
        """Writes the raw body to ``path`` (atomically) and returns its size."""
        if not self.has_body():
            raise HTTPError(411, "send the file as the raw request body")
        tmp = path + ".part"
        size = 0
        try:
            with open(tmp, "wb") as f:
                for data in self.iter_body(limit):
                    f.write(data)
                    size += len(data)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        os.replace(tmp, path)
        return size

    def arg(self, name, default=None, cast=str):
        v = self.query.get(name)
        if v in (None, ""):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            raise HTTPError(400, "bad value for %s" % name)


# ====================================================================== app
def _strip_session(cookie):
    """Never forward our session cookie to a proxied backend."""
    return "; ".join(p.strip() for p in (cookie or "").split(";")
                     if p.strip() and p.strip().partition("=")[0] != SESSION_COOKIE)


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign in · {{NAME}}</title>
<style>
:root{--bg:#f4f5f8;--card:#fff;--text:#14171c;--muted:#636b77;--line:#dfe2e8;--accent:#2f6fec;--bad:#c4372f}
@media (prefers-color-scheme:dark){:root{--bg:#0d0f13;--card:#161a20;--text:#e8eaee;--muted:#98a0ab;--line:#2a2f37;--accent:#5b8ff5;--bad:#ef6a60}}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:radial-gradient(1200px 600px at 10% -10%,color-mix(in srgb,var(--accent) 18%,transparent),transparent),var(--bg);
color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans","Noto Sans Devanagari",sans-serif;
display:flex;align-items:center;justify-content:center;padding:16px}
.card{width:100%;max-width:380px;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:30px 28px;
box-shadow:0 20px 60px rgba(0,0,0,.12)}
.logo{width:44px;height:44px;border-radius:12px;background:var(--accent);display:flex;align-items:center;justify-content:center;
color:#fff;font-weight:800;font-size:20px;margin-bottom:18px}
h1{font-size:20px;margin:0 0 4px}p{color:var(--muted);margin:0 0 22px;font-size:14px}
label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px}
.field{position:relative}
input{width:100%;font:inherit;color:var(--text);background:transparent;border:1px solid var(--line);border-radius:10px;padding:11px 44px 11px 12px}
input:focus{outline:2px solid color-mix(in srgb,var(--accent) 45%,transparent);border-color:var(--accent)}
.eye{position:absolute;right:6px;top:50%;transform:translateY(-50%);border:0;background:none;color:var(--muted);cursor:pointer;padding:6px;font-size:13px}
button.go{width:100%;margin-top:16px;border:0;border-radius:10px;padding:12px;background:var(--accent);color:#fff;font:600 15px system-ui;cursor:pointer}
button.go:disabled{opacity:.6}.err{color:var(--bad);font-size:13px;min-height:20px;margin-top:10px}
.hint{font-size:12px;color:var(--muted);margin-top:18px;border-top:1px solid var(--line);padding-top:14px}
</style></head><body>
<form class="card" id="f" method="post" action="/login">
<div class="logo">&#9656;</div><h1>{{NAME}}</h1><p>Sign in with the password from your notebook.</p>
<label for="pw">Password</label><div class="field"><input id="pw" name="password" type="password" autocomplete="current-password" autofocus required>
<button class="eye" type="button" id="eye">Show</button></div>
<button class="go" id="go">Sign in</button><div class="err" id="err"></div>
<div class="hint">It is the <b>*_UI_PASSWORD</b> Kaggle secret, or the password the notebook printed.</div>
</form>
<script>
const f=document.getElementById('f'),pw=document.getElementById('pw'),err=document.getElementById('err'),go=document.getElementById('go');
document.getElementById('eye').onclick=e=>{pw.type=pw.type==='password'?'text':'password';e.target.textContent=pw.type==='password'?'Show':'Hide';};
f.onsubmit=async e=>{e.preventDefault();go.disabled=true;err.textContent='';
 try{const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw.value})});
  if(r.ok){location.reload();return;} const d=await r.json().catch(()=>({}));err.textContent=d.error||('Sign-in failed ('+r.status+')');}
 catch(x){err.textContent='Network error: '+x;} go.disabled=false;pw.select();};
</script></body></html>"""


def _rpc_error(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


class App:
    def __init__(self, name, password=None, version="1.0", instructions="", proxy=None, log=None,
                 max_body=64 << 20):
        self.name, self.version, self.instructions = name, version, instructions
        self.password = password or None
        self.proxy = proxy.rstrip("/") if proxy else None
        self.log = log or logging.getLogger(name)
        self.max_body = max_body
        self.routes, self.tools, self.servers = [], {}, []
        self._fails = {}

        self.route("GET", "/health", auth=False)(lambda req: {"ok": True})
        self.route("POST", "/login", auth=False)(self._login)
        self.route("GET", "/logout", auth=False)(self._logout)
        self.route("POST", "/logout", auth=False)(self._logout)
        self.route("POST", "/mcp")(self._mcp_http)
        self.route("GET", "/mcp")(lambda req: Response(
            "This MCP server answers POST requests (Streamable HTTP transport, JSON responses).", 405,
            headers={"Allow": "POST"}))
        self.route("DELETE", "/mcp")(lambda req: Response(b"", 405, headers={"Allow": "POST"}))
        self.route("GET", "/mcp/tools")(lambda req: {"tools": self.tool_list()})
        self.route("POST", r"/mcp/tools/(?P<name>[A-Za-z0-9_\-]+)")(self._tool_http)

    # ------------------------------------------------------------------ registration
    def route(self, method, pattern, auth=True):
        rx = re.compile(pattern)

        def deco(fn):
            self.routes.append((method.upper(), rx, fn, auth))
            return fn
        return deco

    def page(self, path, file):
        self.route("GET", re.escape(path))(lambda req: FileResponse(file, ctype="text/html; charset=utf-8"))

    def static(self, prefix, folder):
        root = os.path.realpath(folder)

        def serve(req):
            full = os.path.realpath(os.path.join(root, req.params["rel"]))
            if not full.startswith(root + os.sep):
                raise HTTPError(404, "not found")
            return FileResponse(full)
        self.route("GET", re.escape(prefix) + r"(?P<rel>.+)")(serve)

    def tool(self, name, description, properties=None, required=()):
        """Registers an MCP tool. ``properties`` is a JSON Schema properties object."""
        def deco(fn):
            self.tools[name] = {"fn": fn, "schema": {
                "name": name, "description": description,
                "inputSchema": {"type": "object", "properties": properties or {}, "required": list(required)}}}
            return fn
        return deco

    def tool_list(self):
        return [t["schema"] for t in self.tools.values()]

    def call_tool(self, name, args):
        """Runs a tool. Returns (content_blocks, is_error)."""
        t = self.tools[name]
        if not isinstance(args, dict):
            return [text_block("arguments must be a JSON object")], True
        try:
            out = t["fn"](**args)
        except TypeError as e:
            if "argument" in str(e):
                return [text_block("bad arguments for %s: %s" % (name, e))], True
            self.log.error("tool %s failed\n%s", name, traceback.format_exc())
            return [text_block("TypeError: %s" % e)], True
        except HTTPError as e:
            return [text_block(str(e))], True
        except (ValueError, KeyError) as e:
            return [text_block(str(e).strip("'\""))], True
        except Exception as e:
            self.log.error("tool %s failed\n%s", name, traceback.format_exc())
            return [text_block("%s: %s" % (type(e).__name__, e))], True
        if isinstance(out, ToolContent):
            return list(out), False
        if isinstance(out, str):
            return [text_block(out)], False
        return [text_block(json.dumps(out, indent=1, ensure_ascii=False, default=str))], False

    # ------------------------------------------------------------------ MCP
    def _mcp_http(self, req):
        try:
            msg = json.loads(req.body().decode("utf-8") or "null")
        except ValueError:
            return _rpc_error(None, -32700, "parse error")
        if isinstance(msg, list):
            out = [r for r in map(self._rpc, msg) if r is not None]
            return out if out else Response(b"", 202)
        r = self._rpc(msg)
        return r if r is not None else Response(b"", 202)

    def _rpc(self, m):
        if not isinstance(m, dict) or m.get("jsonrpc") != "2.0":
            return _rpc_error(m.get("id") if isinstance(m, dict) else None, -32600, "invalid request")
        if "method" not in m or "id" not in m:
            return None            # a notification, or a response to a request we never send
        mid, method, params = m["id"], m["method"], m.get("params") or {}
        try:
            if method == "initialize":
                v = params.get("protocolVersion")
                result = {"protocolVersion": v if v in MCP_VERSIONS else MCP_VERSIONS[0],
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": self.name, "version": self.version}}
                if self.instructions:
                    result["instructions"] = self.instructions
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.tool_list()}
            elif method == "tools/call":
                name = params.get("name")
                if name not in self.tools:
                    return _rpc_error(mid, -32602, "unknown tool: %s" % name)
                content, err = self.call_tool(name, params.get("arguments") or {})
                result = {"content": content, "isError": err}
            elif method == "resources/list":
                result = {"resources": []}
            elif method == "resources/templates/list":
                result = {"resourceTemplates": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            else:
                return _rpc_error(mid, -32601, "method not found: %s" % method)
        except Exception as e:
            self.log.error("mcp %s failed\n%s", method, traceback.format_exc())
            return _rpc_error(mid, -32603, str(e))
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _tool_http(self, req):
        name = req.params["name"]
        if name not in self.tools:
            raise HTTPError(404, "unknown tool: %s" % name)
        content, err = self.call_tool(name, req.json())
        if len(content) == 1 and content[0]["type"] == "text":
            try:
                data = json.loads(content[0]["text"])
            except ValueError:
                data = content[0]["text"]
            body = {"error": data} if err else data
        else:
            body = {"content": content, "isError": err}
        return Response(json.dumps(body, ensure_ascii=False, default=str), 400 if err else 200,
                        "application/json; charset=utf-8")

    # ------------------------------------------------------------------ auth
    def _key(self):
        return hashlib.sha256(("studio-session:" + self.password).encode("utf-8")).digest()

    def make_session(self, days=SESSION_DAYS):
        exp = str(int(time.time() + days * 86400))
        return "%s.%s" % (exp, hmac.new(self._key(), exp.encode(), hashlib.sha256).hexdigest()[:40])

    def _session_ok(self, headers):
        for part in (headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == SESSION_COOKIE:
                exp, _, sig = v.partition(".")
                if exp.isdigit() and int(exp) > time.time():
                    want = hmac.new(self._key(), exp.encode(), hashlib.sha256).hexdigest()[:40]
                    if hmac.compare_digest(sig.encode(), want.encode()):
                        return True
        return False

    def _cookie(self, headers, value, max_age):
        secure = (headers.get("X-Forwarded-Proto") or "").lower() == "https" or '"https"' in (headers.get("Cf-Visitor") or "")
        return "%s=%s; Path=/; Max-Age=%d; HttpOnly; SameSite=Lax%s" % (
            SESSION_COOKIE, value, max_age, "; Secure" if secure else "")

    def _login(self, req):
        if not self.password:
            return {"ok": True}
        ip = req.headers.get("Cf-Connecting-Ip") or req.handler.client_address[0]
        now = time.time()
        fails = [t for t in self._fails.get(ip, []) if now - t < 300]
        if len(fails) >= 8:
            raise HTTPError(429, "too many attempts; wait a few minutes")
        if "json" in (req.headers.get("Content-Type") or ""):
            given = str(req.json().get("password") or "")
        else:
            given = parse_qs(req.body().decode("utf-8", "replace")).get("password", [""])[0]
        if not hmac.compare_digest(given.encode("utf-8"), self.password.encode("utf-8")):
            fails.append(now)
            self._fails[ip] = fails
            time.sleep(0.5)
            raise HTTPError(401, "wrong password")
        self._fails.pop(ip, None)
        return Response(json.dumps({"ok": True}), 200, "application/json; charset=utf-8",
                        {"Set-Cookie": self._cookie(req.headers, self.make_session(), SESSION_DAYS * 86400)})

    def _logout(self, req):
        return Response(b"", 303, "text/plain", {"Location": "/", "Set-Cookie": self._cookie(req.headers, "", 0)})

    def authorized(self, headers):
        pw = self.password
        if not pw:
            return True
        if self._session_ok(headers):
            return True

        def ok(given):
            return bool(given) and hmac.compare_digest(given.encode("utf-8"), pw.encode("utf-8"))
        if ok(headers.get("X-Access-Token", "")):
            return True
        h = headers.get("Authorization", "")
        if h[:6].lower() == "basic ":
            try:
                given = base64.b64decode(h[6:].strip()).decode("utf-8").partition(":")[2]
            except Exception:
                given = ""
            return ok(given)
        if h[:7].lower() == "bearer ":
            return ok(h[7:].strip())
        return False

    def _deny(self, h):
        """Pages get the login form; everything else a JSON 401 (no browser pop-up)."""
        if h.command in ("GET", "HEAD") and "text/html" in (h.headers.get("Accept") or ""):
            return self._respond(h, Response(LOGIN_PAGE.replace("{{NAME}}", _html.escape(self.name)), 401,
                                             "text/html; charset=utf-8"))
        self._respond(h, Response(json.dumps({"error": "login required: sign in on the page, or send "
                                              "'Authorization: Bearer <password>'"}), 401,
                                  "application/json; charset=utf-8"))

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, h):
        h._sent = False
        req = None
        try:
            path = urlparse(h.path).path
            method = "GET" if h.command == "HEAD" else h.command
            found, path_known = None, False
            for m, rx, fn, auth in self.routes:
                mt = rx.fullmatch(path)
                if mt:
                    path_known = True
                    if m == method:
                        found = (fn, auth, mt.groupdict())
                        break
            if found is None:
                if path_known:
                    raise HTTPError(405, "method not allowed")
                if not self.proxy:
                    raise HTTPError(404, "not found")
                if not self.authorized(h.headers):
                    return self._deny(h)
                return self._proxy(h)
            fn, auth, params = found
            if auth and not self.authorized(h.headers):
                return self._deny(h)
            req = Request(h, params, self.max_body)
            self._respond(h, fn(req))
        except (BrokenPipeError, ConnectionResetError):
            h.close_connection = True
        except HTTPError as e:
            self._fail(h, e.status, str(e))
        except (ValueError, KeyError) as e:
            msg = ("missing field %s" % e) if isinstance(e, KeyError) else str(e)
            self._fail(h, 400, msg)
        except Exception as e:
            self.log.error("%s %s failed\n%s", h.command, h.path, traceback.format_exc())
            self._fail(h, 500, "%s: %s" % (type(e).__name__, e))
        finally:
            body_left = (req is None or not req.consumed) and (
                "chunked" in (h.headers.get("Transfer-Encoding") or "").lower()
                or (h.headers.get("Content-Length") or "0").strip() not in ("", "0"))
            if body_left:
                h.close_connection = True   # an unread body would corrupt the next request on this connection

    def _fail(self, h, status, message):
        if h._sent:
            h.close_connection = True
            return
        try:
            self._respond(h, Response(json.dumps({"error": message}), status, "application/json; charset=utf-8"))
        except OSError:
            h.close_connection = True

    def _start(self, h, status, headers):
        h.send_response(status)
        h._sent = True
        for k, v in headers:
            h.send_header(k, v)
        h.end_headers()

    def _respond(self, h, out):
        head = h.command == "HEAD"
        if out is None:
            out = Response(b"", 204)
        elif isinstance(out, (dict, list)):
            out = Response(json.dumps(out, ensure_ascii=False, default=str), 200, "application/json; charset=utf-8")
        elif isinstance(out, str):
            out = Response(out)
        elif isinstance(out, bytes):
            out = Response(out, 200, "application/octet-stream")

        if isinstance(out, Response):
            hdrs = {"Content-Type": out.ctype, "Cache-Control": "no-store"}
            hdrs.update(out.headers)
            if out.status not in (204, 304):
                hdrs["Content-Length"] = str(len(out.body))
            self._start(h, out.status, hdrs.items())
            if not head and out.status not in (204, 304) and out.body:
                h.wfile.write(out.body)
        elif isinstance(out, FileResponse):
            self._send_file(h, out, head)
        elif isinstance(out, StreamResponse):
            self._start(h, 200, list({"Content-Type": out.ctype, "Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no", **out.headers}.items())
                        + [("Transfer-Encoding", "chunked")])
            if head:
                h.wfile.write(b"0\r\n\r\n")
                return
            for chunk in out.chunks:
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if chunk:
                    h.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    h.wfile.flush()
            h.wfile.write(b"0\r\n\r\n")
        elif isinstance(out, ZipResponse):
            self._send_zip(h, out, head)
        else:
            raise TypeError("handler returned %r" % type(out))

    def _send_file(self, h, f, head):
        size = os.path.getsize(f.path)
        start, end, status = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", (h.headers.get("Range") or "").strip())
        if m and size and (m.group(1) or m.group(2)):
            a, b = m.groups()
            if a == "":
                start = max(0, size - int(b))
            else:
                start, end = int(a), min(int(b), size - 1) if b else size - 1
            if start > end or start >= size:
                return self._start(h, 416, [("Content-Range", "bytes */%d" % size), ("Content-Length", "0")])
            status = 206
        hdrs = [("Content-Type", f.ctype), ("Content-Length", str(end - start + 1)), ("Accept-Ranges", "bytes"),
                ("Cache-Control", f.cache)]
        if status == 206:
            hdrs.append(("Content-Range", "bytes %d-%d/%d" % (start, end, size)))
        if f.name:
            hdrs.append(("Content-Disposition", _disposition(f.name)))
        self._start(h, status, hdrs)
        if head:
            return
        with open(f.path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                data = fh.read(min(left, 1 << 20))
                if not data:
                    break
                h.wfile.write(data)
                left -= len(data)

    def _send_zip(self, h, z, head):
        self._start(h, 200, [("Content-Type", "application/zip"), ("Content-Disposition", _disposition(z.name)),
                             ("Cache-Control", "no-store"), ("Transfer-Encoding", "chunked")])
        w = h.wfile

        class Chunks:
            def write(self, data):
                if data:
                    w.write(b"%x\r\n" % len(data) + bytes(data) + b"\r\n")
                return len(data)

            def flush(self):
                w.flush()

        if not head:
            with zipfile.ZipFile(Chunks(), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for path, arc in z.files:
                    if isinstance(path, bytes):
                        zf.writestr(arc, path)
                    elif os.path.isfile(path):
                        zf.write(path, arc, compress_type=zipfile.ZIP_STORED if path.endswith(
                            (".safetensors", ".gguf", ".bin", ".npy", ".png", ".jpg", ".mp4", ".zip", ".wav"))
                            else zipfile.ZIP_DEFLATED)
        w.write(b"0\r\n\r\n")
        w.flush()

    # ------------------------------------------------------------------ reverse proxy
    def _proxy(self, h):
        pu = urlparse(self.proxy)
        host, port = pu.hostname, pu.port or 80
        if ("upgrade" in (h.headers.get("Connection") or "").lower()
                and (h.headers.get("Upgrade") or "").lower() == "websocket"):
            return self._proxy_ws(h, host, port)
        req = Request(h, {}, self.max_body)
        headers = [(k, _strip_session(v) if k.lower() == "cookie" else v) for k, v in h.headers.items()
                   if k.lower() not in HOP and k.lower() not in SECRET_HEADERS and k.lower() != "content-length"]
        headers = [(k, v) for k, v in headers if v]
        headers.append(("X-Forwarded-For", h.client_address[0]))
        body = req.body() if req.has_body() else None
        if body is not None:
            headers.append(("Content-Length", str(len(body))))
        conn = http.client.HTTPConnection(host, port, timeout=900)
        try:
            # the original Host header is kept: some backends (ComfyUI) compare it with Origin
            conn.putrequest(h.command, h.path, skip_host="host" in {k.lower() for k, v in headers},
                            skip_accept_encoding=True)
            for k, v in headers:
                conn.putheader(k, v)
            conn.endheaders(body)
            r = conn.getresponse()
        except OSError as e:
            conn.close()
            raise HTTPError(502, "backend not reachable (%s)" % type(e).__name__)
        try:
            hdrs = [(k, v) for k, v in r.getheaders() if k.lower() not in HOP and k.lower() != "content-length"]
            cl = r.getheader("Content-Length")
            chunked = "chunked" in (r.getheader("Transfer-Encoding") or "").lower()
            if h.command == "HEAD" or r.status in (204, 304) or r.status < 200:
                return self._start(h, r.status, hdrs + ([("Content-Length", cl)] if cl else []))
            if cl is not None and not chunked:
                self._start(h, r.status, hdrs + [("Content-Length", cl)])
                while True:
                    data = r.read(1 << 16)
                    if not data:
                        break
                    h.wfile.write(data)
            else:
                self._start(h, r.status, hdrs + [("Transfer-Encoding", "chunked")])
                while True:
                    data = r.read1(1 << 16)
                    if not data:
                        break
                    h.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                    h.wfile.flush()
                h.wfile.write(b"0\r\n\r\n")
        finally:
            conn.close()

    def _proxy_ws(self, h, host, port):
        try:
            up = socket.create_connection((host, port), timeout=10)
        except OSError as e:
            raise HTTPError(502, "backend not reachable (%s)" % type(e).__name__)
        lines = ["%s %s HTTP/1.1" % (h.command, h.path)]
        lines += ["%s: %s" % (k, _strip_session(v) if k.lower() == "cookie" else v) for k, v in h.headers.items()
                  if k.lower() not in SECRET_HEADERS and (k.lower() != "cookie" or _strip_session(v))]
        h._sent = True
        h.close_connection = True
        client = h.connection
        try:
            up.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
            up.settimeout(None)
            client.settimeout(None)
            while True:
                ready, _, broken = select.select([client, up], [], [client, up], 600)
                if broken:
                    break
                for s in ready:
                    data = s.recv(1 << 16)
                    if not data:
                        return
                    (up if s is client else client).sendall(data)
        except OSError:
            pass
        finally:
            up.close()

    # ------------------------------------------------------------------ serving
    def handler_class(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = re.sub(r"\W+", "", app.name) or "studio"

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                app.dispatch(self)

            do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_GET
        return Handler

    def start(self, port):
        """Listens on 0.0.0.0 and [::] in background threads. Calling it again restarts cleanly."""
        self.stop()
        handler = self.handler_class()
        for cls, host in ((_V4Server, "0.0.0.0"), (_V6Server, "::")):
            try:
                srv = cls((host, port), handler)
            except OSError as e:
                self.log.warning("could not listen on [%s]:%d: %s", host, port, e)
                continue
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.servers.append(srv)
        if not self.servers:
            raise RuntimeError("nothing is listening on port %d" % port)
        self.log.info("%s listening on port %d", self.name, port)

    def stop(self):
        while self.servers:
            srv = self.servers.pop()
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass

    def serve_forever(self, port):
        """For app processes started with spawn(): serve until SIGTERM."""
        done = threading.Event()
        signal.signal(signal.SIGTERM, lambda *a: done.set())
        signal.signal(signal.SIGINT, lambda *a: done.set())
        self.start(port)
        done.wait()
        self.log.info("stopping")
        self.stop()


class _V4Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


class _V6Server(_V4Server):
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


# ====================================================================== JSON files and jobs
def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    """Atomic write, so readers in other threads or processes never see half a file."""
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def safe_name(s, n=60):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s or "")).strip("-.")[:n]


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        gpus = []
        for line in out.strip().splitlines():
            i, n, u, t, g = [x.strip() for x in line.split(",")]
            gpus.append({"index": int(i), "name": n, "mem_used_mb": int(u), "mem_total_mb": int(t), "util": int(g)})
        return gpus
    except Exception:
        return []


# ====================================================================== notebook helpers
def kaggle_secret(name):
    """Value of a Kaggle secret, or None when it is missing or not attached to this notebook.
    Outside Kaggle (local runs, Docker) it reads the environment variable of the same name."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return os.environ.get(name) or None
    try:
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None


def ui_password(secret_name):
    pw = kaggle_secret(secret_name)
    if pw:
        print("UI password: from the %s secret. Browser login: any username + that password." % secret_name)
        return pw
    import secrets
    pw = secrets.token_urlsafe(9)
    print("No %s secret, so this session's password is: %s" % (secret_name, pw))
    print("Browser login: any username + that password. Add the secret to keep a fixed password.")
    return pw


def _pidfile(name):
    return os.path.join(PID_DIR, safe_name(name) + ".json")


def _alive(pid, exe=None):
    try:
        with open("/proc/%d/stat" % pid) as f:
            if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        if exe:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                return exe.encode() in f.read()
        return True
    except (OSError, IndexError):
        return False


def process_alive(name):
    info = read_json(_pidfile(name))
    return bool(info) and _alive(info["pid"], info.get("exe"))


def stop_process(name, timeout=20):
    """Stops a process spawn() started under ``name`` (and its children). Returns True if one was running."""
    info = read_json(_pidfile(name))
    if not info:
        return False
    pid, exe = info["pid"], info.get("exe")
    running = _alive(pid, exe)
    if running:
        for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5)):
            try:
                os.killpg(pid, sig)
            except OSError:
                break
            t = time.time()
            while time.time() - t < wait and _alive(pid, exe):
                time.sleep(0.2)
            if not _alive(pid, exe):
                break
    try:
        os.remove(_pidfile(name))
    except OSError:
        pass
    return running


def spawn(name, cmd, log_path, env=None, cwd=None):
    """Starts ``cmd`` in the background with its output appended to ``log_path``.
    A process started earlier under the same name is stopped first, so re-running a cell is safe."""
    stop_process(name)
    os.makedirs(PID_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "ab") as log:
        log.write(("\n===== %s: starting %s =====\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), name)).encode())
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, cwd=cwd, start_new_session=True)
    write_json(_pidfile(name), {"pid": proc.pid, "exe": os.path.basename(cmd[0]), "log": log_path,
                                "started": time.time()})
    return proc


def wait_http(url, timeout=120, name=None, headers=None):
    """Waits until ``url`` answers (any HTTP status). Returns False on timeout or if process ``name`` died."""
    t = time.time()
    while time.time() - t < timeout:
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=3)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            pass
        if name and not process_alive(name):
            return False
        time.sleep(1)
    return False


def tail(path, n=30):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64 * 1024))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])
    except OSError:
        return "(no log yet at %s)" % path


def ensure_cloudflared(path="/kaggle/working/cloudflared"):
    if not os.path.exists(path):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
        os.chmod(path, 0o755)
    return path


def start_tunnel(secret_name, port=7860, log_path="/kaggle/working/logs/cloudflared.log",
                 binary="/kaggle/working/cloudflared"):
    """Starts cloudflared with the token from ``secret_name`` (passed as TUNNEL_TOKEN, never printed).
    A missing secret is not an error: the app keeps running inside the session, without a public URL."""
    stop_process("cloudflared")
    token = kaggle_secret(secret_name)
    if not token:
        print("No public URL: there is no Kaggle secret named %s attached to this notebook.\n"
              "The app is running inside this session anyway. To publish it:\n"
              "  1. Cloudflare Zero Trust -> Networks -> Tunnels: create a tunnel and add a public hostname\n"
              "     pointing to http://localhost:%d\n"
              "  2. Kaggle: Add-ons -> Secrets -> add %s with the tunnel token, and tick it for this notebook\n"
              "  3. Re-run this cell." % (secret_name, port, secret_name))
        return None
    ensure_cloudflared(binary)
    offset = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    proc = spawn("cloudflared", [binary, "tunnel", "--no-autoupdate", "run"], log_path,
                 env=dict(os.environ, TUNNEL_TOKEN=token))

    def new_log():
        with open(log_path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    for _ in range(45):
        time.sleep(1)
        if "Registered tunnel connection" in new_log():
            print("Tunnel connected. Open the public hostname you routed to http://localhost:%d" % port)
            return proc
        if proc.poll() is not None:
            print("cloudflared stopped (exit code %s). Last log lines:" % proc.returncode)
            print("\n".join(new_log().splitlines()[-15:]))
            print("Check that %s holds the token of a tunnel that still exists." % secret_name)
            return None
    print("Tunnel not registered yet; it keeps trying in the background. Last log lines:")
    print("\n".join(new_log().splitlines()[-15:]))
    return proc


def keep_alive(port, names, extra=None, interval=60):
    """The last cell: prints one status line a minute so the Kaggle session stays active.
    Interrupting it only stops the printing; the app keeps running."""
    try:
        while True:
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5)
                parts = ["ui=up"]
            except Exception:
                parts = ["ui=DOWN"]
            for n in names:
                if n == "cloudflared" and not os.path.exists(_pidfile(n)):
                    parts.append("tunnel=off")
                else:
                    parts.append("%s=%s" % (n, "up" if process_alive(n) else "DOWN"))
            gpus = gpu_info()
            if gpus:
                parts.append("GPU " + ", ".join("%d: %.1f/%.0f GB %d%%" % (g["index"], g["mem_used_mb"] / 1024,
                                                                           g["mem_total_mb"] / 1024, g["util"])
                                                for g in gpus))
            if extra:
                try:
                    s = extra()
                    if s:
                        parts.append(s)
                except Exception as e:
                    parts.append("status error: %s" % e)
            print(time.strftime("%H:%M:%S"), " ".join(parts), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped watching. Everything keeps running in the background; re-run this cell to watch again.")


def api_get(port, path, password, timeout=10):
    """GET a JSON endpoint of a local app (used by keep-alive status lines)."""
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 headers={"Authorization": "Bearer %s" % password} if password else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))
