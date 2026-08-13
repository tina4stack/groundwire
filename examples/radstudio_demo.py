#!/usr/bin/env python3
"""
RAD Studio grounding demo — two side-by-side deepseek chats, WITHOUT vs WITH
groundwire. One composer drives both: the left chat answers from the bare model,
the right chat answers after groundwire retrieves and injects RAD Studio source.
Multi-turn; shows the exact chunks that were grounded each turn.

Deploy-first, add-corpus-later: starts with an EMPTY index. Add the corpus after
deployment via the drop folder + Reindex, or by uploading files / a .zip in-page.

    GW_LLM_URL=http://192.168.88.99:11457 GW_LLM_MODEL=tina4-deepseek-ft \
    python3 examples/radstudio_demo.py --corpus ~/radstudio-corpus --port 8899

Point radstudiohelp.andrevanzuydam.com at the host running this (it must reach
the deepseek endpoint). Stdlib only + groundwire.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from groundwire import Groundwire, ChatMemory

LLM_URL = os.environ.get("GW_LLM_URL", "http://192.168.88.99:11457").rstrip("/")
LLM_MODEL = os.environ.get("GW_LLM_MODEL", "tina4-deepseek-ft")
K = int(os.environ.get("GW_K", "5"))
# Reader window is ~8K tokens. Cap injected context so there's room for a full
# answer (else vLLM shrinks max_tokens to fit and the reply gets chopped).
CTX_CHARS = int(os.environ.get("GW_CTX_CHARS", "16000"))   # ~4k tokens
MAX_TOKENS = int(os.environ.get("GW_MAX_TOKENS", "900"))
EXTS = (
    # Delphi / Pascal
    ".pas", ".dpr", ".dpk", ".dproj", ".inc", ".fmx", ".dfm", ".lpr", ".lfm",
    # C / C++ (C++Builder)
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".cbproj",
    # docs / markup / plain text
    ".md", ".txt", ".rst", ".html", ".htm", ".xml", ".json",
    ".yaml", ".yml", ".ini", ".cfg", ".sql", ".cs",
)

CORPUS_DIR = ""
_LOCK = threading.Lock()
GW = Groundwire(memory="sqlite_fts", k=K)
STATS = {"files": 0, "bytes": 0}

SYS_PLAIN = ("You are a RAD Studio / Delphi expert. Answer questions about the RAD "
             "Studio libraries concisely and correctly. If unsure, say so.")
SYS_GROUNDED = ("You are a RAD Studio / Delphi expert. Answer using the RAD Studio "
                "source/docs provided in the latest message. Quote exact identifiers "
                "and signatures. If the answer isn't in the provided context, say so.")


def deepseek(messages, max_tokens=MAX_TOKENS) -> tuple[str, float]:
    payload = {"model": LLM_MODEL, "temperature": 0, "max_tokens": max_tokens,
               "messages": messages}
    req = urllib.request.Request(LLM_URL + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        text = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        text = f"[reader error: {e}]"
    return text, time.time() - t0


_MSPEC = {}


def model_spec() -> dict:
    """Reader spec (id, context length, HF root) — fetched once from /v1/models."""
    if _MSPEC:
        return dict(_MSPEC)
    try:
        with urllib.request.urlopen(LLM_URL + "/v1/models", timeout=5) as r:
            d = json.loads(r.read())["data"][0]
        _MSPEC.update(id=d.get("id") or LLM_MODEL, ctx=d.get("max_model_len"),
                      root=(d.get("root") or "").split("/")[-1] or None)
    except Exception:
        _MSPEC.update(id=LLM_MODEL, ctx=None, root=None)
    return dict(_MSPEC)


HIST_K = int(os.environ.get("GW_HIST_K", "3"))   # relevant prior turns B retrieves


def answer(question: str, turns: list) -> dict:
    """Before/after over CHAT HISTORY + source.
      A (without): the FULL raw history is dumped into the prompt as chat turns —
        it grows unbounded and a small 8K reader degrades as the thread lengthens.
      B (with): the history is INDEXED; only the turns relevant to the new question
        are retrieved and injected alongside the relevant RAD Studio source —
        bounded and clean no matter how long the conversation.
    `turns` = prior [{q, wo, w}] the client carries (stateless server)."""
    with _LOCK:
        gw, have = GW, STATS["files"]
    src = gw.retrieve(question, k=K) if have else []
    sources = [{"title": gw.sources.get(cid, "?"), "text": t} for cid, t, _ in src]

    # A — naive: replay the ENTIRE history as messages (the degrading baseline)
    wo_msgs = [{"role": "system", "content": SYS_PLAIN}]
    for t in turns:
        wo_msgs.append({"role": "user", "content": t.get("q", "")})
        wo_msgs.append({"role": "assistant", "content": t.get("wo", "")})
    wo_msgs.append({"role": "user", "content": question})
    without, t_wo = deepseek(wo_msgs)

    # B — groundwire: index the history, retrieve only the RELEVANT prior turns
    # (groundwire.ChatMemory — the library's retrieved-conversational-memory primitive)
    hist_used = []
    if turns:
        cm = ChatMemory(k=HIST_K).add_turns((t.get("q", ""), t.get("w", "")) for t in turns)
        for _, text, _ in cm.retrieve(question):
            hist_used.append({"title": "earlier", "text": text})

    if sources or hist_used:
        parts = []
        if hist_used:
            parts.append("Relevant earlier conversation:\n"
                         + "\n\n".join(h["text"] for h in hist_used))
        if sources:
            ctx = "\n\n".join(f"### {s['title']}\n{s['text']}" for s in sources)[:CTX_CHARS]
            parts.append("RAD Studio source:\n" + ctx)
        w_user = "\n\n".join(parts) + f"\n\nQuestion: {question}"
        with_, t_w = deepseek([{"role": "system", "content": SYS_GROUNDED},
                               {"role": "user", "content": w_user}])
    else:
        with_, t_w = "[no corpus loaded yet — add RAD Studio source below, then Reindex]", 0.0

    return {"without": without, "with": with_, "sources": sources,
            "history_used": hist_used, "t_without": round(t_wo, 1),
            "t_with": round(t_w, 1), "k": len(sources), "kh": len(hist_used),
            "turns": len(turns) + 1}


def reindex() -> dict:
    gw = Groundwire(memory="sqlite_fts", k=K)
    files = nbytes = 0
    for root, _, names in os.walk(CORPUS_DIR):
        for fn in names:
            if not fn.lower().endswith(EXTS):
                continue
            path = os.path.join(root, fn)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if not text.strip():
                continue
            gw.ingest_code(text, title=os.path.relpath(path, CORPUS_DIR), max_lines=60)
            files += 1
            nbytes += len(text)
    global GW, STATS
    with _LOCK:
        GW, STATS = gw, {"files": files, "bytes": nbytes}
    print(f"reindexed: {files} files · {nbytes/1e6:.1f} MB")
    return {"files": files, "bytes": nbytes}


def _safe_member(name: str) -> str | None:
    name = name.replace("\\", "/")
    if name.startswith("/") or ".." in name.split("/"):
        return None
    return name


def save_upload(name: str, data: bytes) -> dict:
    base = os.path.basename(name.replace("\\", "/")) or "upload.bin"
    dest = os.path.join(CORPUS_DIR, base)
    with open(dest, "wb") as f:
        f.write(data)
    extracted = 0
    if base.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(dest) as z:
                for m in z.namelist():
                    safe = _safe_member(m)
                    if not safe or m.endswith("/") or not safe.lower().endswith(EXTS):
                        continue
                    target = os.path.join(CORPUS_DIR, safe)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(m) as src, open(target, "wb") as out:
                        out.write(src.read())
                    extracted += 1
        except zipfile.BadZipFile:
            return {"error": "bad zip", "saved": base}
        os.remove(dest)
    return {"saved": base, "extracted": extracted}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAD Studio Help — grounded</title>
<style>
:root{--bg:#f5f1e8;--panel:#fffdf8;--ink:#26211d;--muted:#8a8178;--line:#e6ded0;
  --rust:#c15831;--code:#2b2521;--code-ink:#f0e9df;--u:#e9e0d0}
@media(prefers-color-scheme:dark){:root{--bg:#1c1a18;--panel:#252220;--ink:#ece5da;
  --muted:#9a9186;--line:#39332d;--rust:#c96a41;--code:#141210;--code-ink:#e8e0d5;--u:#332c25}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{margin:0;font-size:18px}.badge{background:var(--rust);color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;font-weight:700}
.st{color:var(--muted);font-size:13px}.st b{color:var(--ink)}
header .sp{flex:1}
input[type=text]{padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink);font:inherit}
button{padding:9px 14px;border:none;border-radius:9px;background:var(--rust);color:#fff;font-weight:700;cursor:pointer}
button.ghost{background:transparent;color:var(--rust);border:1px solid var(--rust)}button:disabled{opacity:.5}
.cols{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);overflow:hidden}
@media(max-width:740px){.cols{grid-template-columns:1fr;overflow:auto}}
.chat{background:var(--bg);display:flex;flex-direction:column;overflow:hidden;min-height:0}
.chat .hd{padding:8px 16px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
  border-bottom:1px solid var(--line);background:var(--panel)}
.chat.wo .hd{color:var(--muted)}.chat.w .hd{color:var(--rust);font-weight:700}
.msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.b{max-width:92%;padding:9px 12px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.b.user{align-self:flex-end;background:var(--u)}
.b.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--line)}
.b code{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.b pre{background:var(--code);color:var(--code-ink);padding:10px;border-radius:8px;overflow:auto;font-size:13px;margin:6px 0 0}
.b .m{color:var(--muted);font-size:11px;margin-top:6px}
.src{align-self:flex-start;max-width:96%;font-size:11px;color:var(--muted)}
.src summary{cursor:pointer;color:var(--rust)}
.src .t{color:var(--rust);font-family:ui-monospace,monospace}
.src pre{background:var(--code);color:var(--code-ink);padding:8px;border-radius:6px;overflow:auto;white-space:pre-wrap}
footer{border-top:1px solid var(--line);padding:12px 18px;background:var(--panel);display:flex;gap:8px}
footer input{flex:1}
.pills{display:flex;gap:6px;flex-wrap:wrap;padding:8px 18px;border-bottom:1px solid var(--line);background:var(--bg)}
.pills b{color:var(--muted);font-size:12px;align-self:center;margin-right:2px}
.pill{border:1px solid var(--line);border-radius:20px;padding:4px 11px;font-size:12.5px;cursor:pointer;background:var(--panel);color:var(--muted)}
.pill:hover{border-color:var(--rust);color:var(--ink)}
.chat.w .hd{position:relative}
</style></head><body>
<header>
  <h1>RAD Studio Help <span class="badge">grounded</span></h1>
  <span class="st">model <b id="model">…</b><span id="spec"></span></span>
  <span class="sp"></span>
  <span class="st">memory <b id="cfiles">0</b> files · <b id="cbytes">0</b> MB · <b id="ctok">0</b> tokens · <span style="opacity:.6">off-GPU</span></span>
  <input type="file" id="up" multiple accept=".pas,.dpr,.dpk,.inc,.fmx,.dfm,.md,.txt,.html,.zip" style="max-width:170px">
  <button class="ghost" id="reidx">Reindex</button>
</header>
<div class="pills" id="pills"><b>Try:</b></div>
<div class="cols">
  <div class="chat wo"><div class="hd">A · Without groundwire <span style="opacity:.55;text-transform:none;font-weight:400">· dumps full history</span></div><div class="msgs" id="mwo"></div></div>
  <div class="chat w"><div class="hd">B · With groundwire <span style="opacity:.55;text-transform:none;font-weight:400">· retrieves relevant turns + source</span></div><div class="msgs" id="mw"></div></div>
</div>
<footer>
  <button class="ghost" id="newc">New chat</button>
  <input type="text" id="q" placeholder="Ask a follow-up — watch A drift as history grows, B stay sharp…" autocomplete="off">
  <span class="st" style="align-self:center;white-space:nowrap">turn <b id="turnc">0</b></span>
  <button id="go">Send to both</button>
</footer>
<script>
const $=id=>document.getElementById(id);
let turns=[];
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function md(s){s=esc(s);
  // fenced code — close at ``` OR end-of-string so a truncated block still renders
  s=s.replace(/```[a-zA-Z0-9]*\n?([\s\S]*?)(?:```|$)/g,(_,c)=>"<pre>"+c.replace(/\n+$/,"")+"</pre>");
  return s.replace(/`([^`\n]+)`/g,"<code>$1</code>")}
function bubble(pane,cls,html){const d=document.createElement("div");d.className="b "+cls;d.innerHTML=html;
  $(pane).appendChild(d);$(pane).scrollTop=$(pane).scrollHeight;return d}
async function meta(){const m=await fetch("/api/meta").then(r=>r.json());
  $("model").textContent=m.model;
  const sp=m.spec||{};
  $("spec").textContent=(sp.root?" · "+sp.root:"")+(sp.ctx?" · "+Math.round(sp.ctx/1024)+"K ctx":"");
  $("cfiles").textContent=m.files;$("cbytes").textContent=(m.bytes/1e6).toFixed(0);
  const t=m.tokens||0;$("ctok").textContent=t>=1e6?(t/1e6).toFixed(1)+"M":(t/1e3).toFixed(0)+"K"}
const PILLS=["What does the IScopes interface ToArray function return?",
  "What is the TFDPhysMongoDBConnection class used for?",
  "How does Data.Cloud.AzureAPI sign a request?",
  "Which class represents a FireDAC database connection?"];
PILLS.forEach(q=>{const c=document.createElement("span");c.className="pill";c.textContent=q;
  c.onclick=()=>{$("q").value=q;send()};$("pills").appendChild(c)});
async function send(){const q=$("q").value.trim();if(!q)return;$("q").value="";$("go").disabled=true;
  bubble("mwo","user",esc(q));bubble("mw","user",esc(q));        // thread grows — that's the point
  const bwo=bubble("mwo","bot","…"),bw=bubble("mw","bot","…");
  try{const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({question:q,turns:turns})}).then(r=>r.json());
    bwo.innerHTML=md(r.without)+'<div class="m">full history · turn '+r.turns+' · '+r.t_without+'s</div>';
    bw.innerHTML=md(r.with)+'<div class="m">'+r.k+' source + '+r.kh+' history chunks · '+r.t_with+'s</div>';
    const bits=[];
    if(r.history_used&&r.history_used.length)bits.push('<b>retrieved history</b>'+r.history_used.map(h=>'<pre>'+esc(h.text)+'</pre>').join(''));
    if(r.sources&&r.sources.length)bits.push('<b>retrieved source</b>'+r.sources.map(s=>'<div class="t">'+esc(s.title)+'</div><pre>'+esc(s.text)+'</pre>').join(''));
    if(bits.length){const d=document.createElement("details");d.className="src";
      d.innerHTML='<summary>grounded on '+r.kh+' history + '+r.k+' source chunks</summary>'+bits.join('');
      $("mw").appendChild(d);$("mw").scrollTop=$("mw").scrollHeight}
    turns.push({q:q,wo:r.without,w:r.with});$("turnc").textContent=turns.length;
  }catch(e){bwo.textContent=bw.textContent="[error: "+e.message+"]"}
  $("go").disabled=false;$("q").focus()}
function newChat(){turns=[];$("mwo").innerHTML="";$("mw").innerHTML="";$("turnc").textContent=0;$("q").focus()}
$("go").onclick=send;$("newc").onclick=newChat;$("q").onkeydown=e=>{if(e.key==="Enter")send()};
$("reidx").onclick=async()=>{$("reidx").disabled=true;$("reidx").textContent="Reindexing…";
  await fetch("/api/reindex",{method:"POST"});await meta();$("reidx").textContent="Reindex";$("reidx").disabled=false};
$("up").onchange=async e=>{const fs=[...e.target.files];if(!fs.length)return;
  $("reidx").disabled=true;$("reidx").textContent="Uploading…";
  for(const f of fs){await fetch("/api/upload?name="+encodeURIComponent(f.name),{method:"POST",body:f})}
  $("reidx").textContent="Reindexing…";await fetch("/api/reindex",{method:"POST"});await meta();
  $("up").value="";$("reidx").textContent="Reindex";$("reidx").disabled=false};
meta();$("q").focus();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(PAGE, "text/html")
        if self.path == "/api/meta":
            with _LOCK:
                s = dict(STATS)
            return self._send(json.dumps({**s, "model": LLM_MODEL,
                                          "spec": model_spec(),
                                          "tokens": int(s["bytes"] / 4)}))
        self._send(json.dumps({"error": "not found"}), code=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/ask":
            b = json.loads(self._read() or b"{}")
            q = (b.get("question") or "").strip()
            if not q:
                return self._send(json.dumps({"error": "empty"}))
            return self._send(json.dumps(answer(q, b.get("turns") or [])))
        if path == "/api/reindex":
            return self._send(json.dumps(reindex()))
        if path == "/api/upload":
            from urllib.parse import parse_qs, urlparse
            name = (parse_qs(urlparse(self.path).query).get("name") or ["upload.bin"])[0]
            return self._send(json.dumps(save_upload(name, self._read())))
        self._send(json.dumps({"error": "not found"}), code=404)


def main():
    global CORPUS_DIR
    ap = argparse.ArgumentParser(description="RAD Studio grounding demo (two chats)")
    ap.add_argument("--corpus", default="~/radstudio-corpus")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    CORPUS_DIR = os.path.expanduser(args.corpus)
    os.makedirs(CORPUS_DIR, exist_ok=True)
    print(f"reader: {LLM_MODEL} @ {LLM_URL}\ncorpus drop folder: {CORPUS_DIR}")
    reindex()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"RAD Studio demo: http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
