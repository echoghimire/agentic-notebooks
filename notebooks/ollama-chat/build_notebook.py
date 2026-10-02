import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": src,
                  "outputs": [], "execution_count": None})

md("""# Ollama on Kaggle, exposed via Cloudflare Tunnel

Run the cells top to bottom (or **Run All**). Nothing blocks until the last cell.

- Ollama API: `127.0.0.1:11434`
- Chat UI: port `7860` (no Gradio; uses Python's standard library, so there is nothing to install)
- Public URL: whatever hostname your tunnel routes to `http://localhost:7860` (for example `chat.example.com`)

**Before running:** add your tunnel token as a Kaggle secret named `OLLAMA_TUNNEL_TOKEN` (Add-ons → Secrets) and turn on Internet in the session settings.
The old name `CF_TUNNEL_TOKEN` still works.

**Optional:** add a secret named `OLLAMA_UI_PASSWORD` to require a login (any username + that password). Without it, anyone who knows your hostname can use the chat and your session's GPU.""")

code("""# 1. Install Ollama and cloudflared
!apt-get -qq update > /dev/null && apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
!curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared
!./cloudflared --version
print("Installation complete.")""")

code("""# 2. Start Ollama in the background and pull the model
import os, subprocess, time, requests

MODEL_NAME = "llama3.2"
OLLAMA_URL = "http://127.0.0.1:11434"

os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"

# Log to a file instead of an unread PIPE (a full pipe buffer can freeze the server)
ollama_log = open("ollama.log", "a")
ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=ollama_log, stderr=subprocess.STDOUT)

for _ in range(60):
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("Ollama did not start - check ollama.log")
print("Ollama is up on", OLLAMA_URL)

# The original notebook never pulled the model, so every chat request would fail
subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
print(f"Model '{MODEL_NAME}' ready.")""")

code(r'''# 3. Lightweight chat UI + proxy on port 7860 (standard library only, runs in a background thread)
import base64, hmac, json, socket, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

PORT = 7860

def kaggle_secret(name):
    """Read a Kaggle secret; returns None if it doesn't exist or isn't attached to this notebook."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None

PASSWORD = kaggle_secret("OLLAMA_UI_PASSWORD")   # optional; without it the UI is open to anyone with the URL
print("UI login:", "any username + OLLAMA_UI_PASSWORD" if PASSWORD else "off (no OLLAMA_UI_PASSWORD secret)")

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ollama Chat</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#1d2129;--muted:#6b7280;--me:#2563eb;--border:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--panel:#181b21;--text:#e6e8eb;--muted:#9aa1ab;--me:#3b82f6;--border:#2a2f37}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100dvh}
header{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
header b{font-size:16px} header span{color:var(--muted);font-size:13px}
#log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px;max-width:860px;width:100%;margin:0 auto}
.msg{padding:10px 14px;border-radius:12px;max-width:85%;white-space:pre-wrap;word-wrap:break-word}
.user{align-self:flex-end;background:var(--me);color:#fff}
.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--border)}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--border);max-width:860px;width:100%;margin:0 auto}
textarea{flex:1;resize:none;padding:10px;border-radius:10px;border:1px solid var(--border);background:var(--panel);color:var(--text);font:inherit;min-height:44px;max-height:160px}
button{padding:0 16px;border:0;border-radius:10px;background:var(--me);color:#fff;font:inherit;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
#clear{background:transparent;color:var(--muted);border:1px solid var(--border)}
</style></head><body>
<header><b>Ollama Chat</b><span id="model"></span></header>
<div id="log"></div>
<form id="f">
  <textarea id="q" rows="1" placeholder="Type a message... (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send">Send</button><button type="button" id="clear">Clear</button>
</form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),send=document.getElementById('send');
let history=[];
fetch('/health').then(r=>r.json()).then(d=>document.getElementById('model').textContent=d.model).catch(()=>{});
function add(role,text){const d=document.createElement('div');d.className='msg '+(role==='user'?'user':'bot');d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit();}});
document.getElementById('clear').onclick=()=>{history=[];log.innerHTML='';};
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text||send.disabled)return;
  q.value='';add('user',text);history.push({role:'user',content:text});
  const bubble=add('bot','...');send.disabled=true;let reply='';
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});
    const reader=r.body.getReader(),dec=new TextDecoder();
    while(true){const {done,value}=await reader.read();if(done)break;reply+=dec.decode(value,{stream:true});bubble.textContent=reply;log.scrollTop=log.scrollHeight;}
    history.push({role:'assistant',content:reply});
  }catch(err){bubble.textContent='Error: '+err;}
  send.disabled=false;q.focus();
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep the notebook output quiet

    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        if not PASSWORD:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(h[6:]).decode("utf-8").partition(":")
                if hmac.compare_digest(given.encode(), PASSWORD.encode()):
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Ollama Chat", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _chunk(self, text):
        data = text.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path != "/health" and not self._authed():   # /health stays open for the keep-alive cell
            return
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "model": MODEL_NAME}), "application/json")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if not self._authed():
            return
        if self.path != "/api/chat":
            return self._send(404, "Not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", 0))
            messages = json.loads(self.rfile.read(length) or b"{}").get("messages", [])
        except Exception:
            return self._send(400, "Bad request", "text/plain")

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            with requests.post(f"{OLLAMA_URL}/api/chat",
                               json={"model": MODEL_NAME, "messages": messages, "stream": True},
                               stream=True, timeout=(5, 600)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    tok = d.get("message", {}).get("content", "")
                    if tok:
                        self._chunk(tok)
                    if d.get("done"):
                        break
        except (BrokenPipeError, ConnectionResetError):
            return  # browser closed the tab mid-reply
        except Exception as e:
            try:
                self._chunk(f"\n[Error: {e}]")
            except Exception:
                return
        try:
            self.wfile.write(b"0\r\n\r\n")
        except Exception:
            pass


class V4Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class V6Server(V4Server):
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


# Shut down previous instances if this cell is re-run
for srv in globals().get("chat_servers", []):
    try:
        srv.shutdown(); srv.server_close()
    except Exception:
        pass

# cloudflared resolved "localhost" to [::1] in the old log ("connection refused"),
# so listen on BOTH IPv4 and IPv6 loopback/any on the same port.
chat_servers = []
for cls, addr in ((V4Server, "0.0.0.0"), (V6Server, "::")):
    try:
        srv = cls((addr, PORT), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        chat_servers.append(srv)
    except OSError as e:
        print(f"Could not listen on {addr}:{PORT} -> {e}")
assert chat_servers, f"Nothing is listening on port {PORT}"

for host in ("127.0.0.1", "[::1]"):
    try:
        print(host, "->", requests.get(f"http://{host}:{PORT}/health", timeout=3).json())
    except Exception as e:
        print(host, "-> FAILED:", e)''')

code("""# 4. Start the Cloudflare tunnel in the background (does NOT block the notebook)
TOKEN = kaggle_secret("OLLAMA_TUNNEL_TOKEN") or kaggle_secret("CF_TUNNEL_TOKEN")   # CF_TUNNEL_TOKEN: old name
assert TOKEN, "Add your tunnel token as a Kaggle secret named OLLAMA_TUNNEL_TOKEN"

if "tunnel_proc" in globals() and tunnel_proc.poll() is None:
    tunnel_proc.terminate()

tunnel_log = open("cloudflared.log", "a")
# Token is passed via env var so it never appears in the notebook output or process list
tunnel_proc = subprocess.Popen(["./cloudflared", "tunnel", "--no-autoupdate", "run"],
                               stdout=tunnel_log, stderr=subprocess.STDOUT,
                               env={**os.environ, "TUNNEL_TOKEN": TOKEN})

for _ in range(30):
    time.sleep(1)
    if "Registered tunnel connection" in open("cloudflared.log").read():
        print("Tunnel connected. Open your public hostname (the hostname you set up in Cloudflare)")
        break
else:
    print("Tunnel not registered yet - last log lines:")
    print("".join(open("cloudflared.log").readlines()[-15:]))""")

code("""# 5. Keep-alive monitor. Leave this running; interrupt it to stop watching (services keep running).
import datetime
while True:
    status = {
        "ollama": "up" if ollama_proc.poll() is None else f"DOWN ({ollama_proc.returncode})",
        "tunnel": "up" if tunnel_proc.poll() is None else f"DOWN ({tunnel_proc.returncode})",
    }
    try:
        requests.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
        status["ui"] = "up"
    except Exception:
        status["ui"] = "DOWN"
    print(datetime.datetime.now().strftime("%H:%M:%S"), status, flush=True)
    time.sleep(60)""")

nb = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
        "kaggle": {"accelerator": "none", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
    },
    "nbformat": 4, "nbformat_minor": 4, "cells": cells,
}
with open("ollama-chat.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("ok")
