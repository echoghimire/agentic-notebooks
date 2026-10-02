"""Gateway on port 7860: the one thing the Cloudflare Tunnel exposes.

- Password-protects everything (browser: HTTP Basic, any username; agents and MCP
  clients: ``Authorization: Bearer <password>`` or ``X-Access-Token: <password>``).
- Serves the built OpenShorts dashboard (SPA fallback to index.html).
- Streams /api, /videos, /thumbnails, /gallery, /video, /mcp, /health to the
  OpenShorts backend on 127.0.0.1:8000 (uploads and range requests included).
- Mounts the B-roll page and API at /broll.

Standard library + starlette/httpx/uvicorn (already installed with OpenShorts).
"""
import base64
import hmac
import os
import socket
import threading

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Mount, Route

BACKEND = os.environ.get("OPENSHORTS_BACKEND", "http://127.0.0.1:8000")
PROXY_PREFIXES = ("/api", "/videos", "/thumbnails", "/gallery", "/video", "/mcp", "/health")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
CFG = {"password": None, "dist": None}
_client = None


def _client_get():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=BACKEND, timeout=httpx.Timeout(None, connect=10.0), follow_redirects=False)
    return _client


# ------------------------------------------------------------------ auth
def _token_ok(given):
    pw = CFG["password"]
    return bool(given) and hmac.compare_digest(given.encode(), pw.encode())


def authorized(request):
    """Returns (ok, strip_authorization_header)."""
    if not CFG["password"]:
        return True, False
    if _token_ok(request.headers.get("x-access-token", "")):
        return True, False
    h = request.headers.get("authorization", "")
    if h.startswith("Basic "):
        try:
            _, _, given = base64.b64decode(h[6:]).decode("utf-8").partition(":")
        except Exception:
            given = ""
        if _token_ok(given):
            return True, True
    if h.startswith("Bearer ") and _token_ok(h[7:].strip()):
        return True, True
    return False, False


def denied(request):
    if request.url.path.startswith(("/api", "/mcp", "/broll/api")):
        return JSONResponse({"error": "unauthorized: send Authorization: Bearer <password> or X-Access-Token"},
                            status_code=401, headers={"WWW-Authenticate": 'Basic realm="OpenShorts Studio"'})
    return Response("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="OpenShorts Studio"'})


class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        if request.url.path in ("/healthz",):
            return await self.app(scope, receive, send)
        ok, strip = authorized(request)
        if not ok:
            return await denied(request)(scope, receive, send)
        if strip:   # never forward our password to OpenShorts
            scope = dict(scope)
            scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"authorization"]
        await self.app(scope, receive, send)


# ------------------------------------------------------------------ proxy
async def proxy(request: Request):
    client = _client_get()
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP]
    headers.append(("x-forwarded-proto", request.headers.get("x-forwarded-proto", request.url.scheme)))
    url = httpx.URL(path=request.url.path, query=request.url.query.encode())
    body = request.stream() if request.method not in ("GET", "HEAD", "OPTIONS") else None
    try:
        req = client.build_request(request.method, url, headers=headers, content=body)
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"error": "OpenShorts backend is not reachable: %s" % e.__class__.__name__}, status_code=502)
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP and k.lower() != "content-encoding"}

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=out_headers)


async def render_unavailable(request):
    return JSONResponse({"error": "The Remotion renderer is not running in this notebook."}, status_code=503)


async def healthz(request):
    return JSONResponse({"ok": True})


# ------------------------------------------------------------------ dashboard
async def dashboard(request: Request):
    path = request.url.path
    if path.startswith(PROXY_PREFIXES):
        return await proxy(request)
    dist = CFG["dist"]
    if not dist or not os.path.isdir(dist):
        return PlainTextResponse("Dashboard not built. Run the install cell, or use /broll.", status_code=503)
    rel = os.path.normpath(path.lstrip("/")) if path != "/" else "index.html"
    full = os.path.realpath(os.path.join(dist, rel))
    if not full.startswith(os.path.realpath(dist) + os.sep):
        return PlainTextResponse("Not found", status_code=404)
    if os.path.isdir(full):
        full = os.path.join(full, "index.html")
    if os.path.isfile(full):
        cache = "public, max-age=31536000, immutable" if "/assets/" in full else "no-cache"
        return FileResponse(full, headers={"Cache-Control": cache})
    return FileResponse(os.path.join(dist, "index.html"), headers={"Cache-Control": "no-cache"})


def build_app(broll_app=None):
    routes = [Route("/healthz", healthz)]
    if broll_app is not None:
        routes.append(Mount("/broll", app=broll_app))
    routes += [Route("/render/{rest:path}", render_unavailable, methods=["GET", "POST", "PUT", "DELETE"]),
               Route("/{rest:path}", dashboard, methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])]
    app = Starlette(routes=routes)
    app.add_middleware(AuthMiddleware)
    return app


# ------------------------------------------------------------------ server
SERVERS = []


def start(port=7860, password=None, dist=None, broll_app=None):
    """Serve on IPv4 and IPv6 (cloudflared may dial [::1]) in background threads."""
    stop()
    CFG.update(password=password or None, dist=dist)
    app = build_app(broll_app)
    for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        # Bind the sockets here: uvicorn's own "::" socket is dual-stack and collides with 0.0.0.0.
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((host, port))
        except OSError as e:
            print("Could not listen on [%s]:%d -> %s" % (host, port, e))
            continue
        cfg = uvicorn.Config(app, log_level="warning", proxy_headers=True,
                             forwarded_allow_ips="*", timeout_keep_alive=30)
        server = uvicorn.Server(cfg)
        server.install_signal_handlers = lambda: None   # running inside a notebook thread
        t = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        t.start()
        SERVERS.append((server, t, sock))
    if not SERVERS:
        raise RuntimeError("nothing is listening on port %d" % port)
    return app


def stop():
    while SERVERS:
        server, t, sock = SERVERS.pop()
        server.should_exit = True
        t.join(timeout=5)
        try:
            sock.close()
        except OSError:
            pass
