// Shared helpers for the agentic-notebooks app pages (served at /static/kit.js).
// The browser already holds the HTTP Basic login, so fetch() calls need no token.
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : String(kid));
  return e;
}

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.json !== undefined || opts.body ? "POST" : "GET"), headers: {} };
  if (opts.json !== undefined) { init.body = JSON.stringify(opts.json); init.headers["Content-Type"] = "application/json"; }
  else if (opts.body) init.body = opts.body;
  const r = await fetch(path, init);
  const ct = r.headers.get("Content-Type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || r.status + " " + r.statusText);
  return data;
}

function toast(msg, kind = "") {
  let box = $("#toasts");
  if (!box) { box = el("div", { id: "toasts" }); document.body.append(box); }
  const t = el("div", { class: "toast " + kind }, msg);
  box.append(t);
  setTimeout(() => t.remove(), kind === "bad" ? 9000 : 4500);
}

async function guard(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg, "ok"); return r; }
  catch (e) { toast(e.message || String(e), "bad"); throw e; }
}

function setupTabs(onChange) {
  const btns = $$("nav.tabs button[data-tab]");
  const show = (name) => {
    btns.forEach(b => b.classList.toggle("on", b.dataset.tab === name));
    $$("[data-pane]").forEach(p => p.classList.toggle("on", p.dataset.pane === name));
    try { history.replaceState(null, "", "#" + name); } catch (e) {}
    if (onChange) onChange(name);
  };
  btns.forEach(b => b.addEventListener("click", () => show(b.dataset.tab)));
  const first = location.hash.slice(1);
  show(btns.some(b => b.dataset.tab === first) ? first : btns[0].dataset.tab);
  return show;
}

function fmtBytes(n) {
  if (n == null) return "";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(1) : n) + " " + u[i];
}

function fmtDur(s) {
  if (s == null || isNaN(s)) return "";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(sec).padStart(2, "0");
}

function fmtAgo(ts) {
  if (!ts) return "";
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "just now";
  if (d < 3600) return Math.floor(d / 60) + " min ago";
  if (d < 86400) return Math.floor(d / 3600) + " h ago";
  return new Date(ts * 1000).toLocaleString();
}

function badge(state) {
  const kind = { done: "ok", ready: "ok", running: "warn", queued: "", loading: "warn", error: "bad", stopped: "bad" }[state] || "";
  return el("span", { class: "badge " + kind }, state || "?");
}

function progressBar(frac) {
  return el("div", { class: "bar" }, el("i", { style: { width: Math.round(Math.max(0, Math.min(1, frac || 0)) * 100) + "%" } }));
}

function every(ms, fn) {
  let stopped = false;
  const tick = async () => { if (stopped) return; try { await fn(); } catch (e) {} if (!stopped) setTimeout(tick, ms); };
  tick();
  return () => { stopped = true; };
}

function dropZone(zone, input, onFiles) {
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files.length) onFiles([...input.files]); input.value = ""; });
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("over"); if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]); });
}

function mcpHelp(name) {
  const url = location.origin + "/mcp";
  return "claude mcp add --transport http " + name + " " + url + " --header \"Authorization: Bearer <password>\"";
}
