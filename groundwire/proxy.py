#!/usr/bin/env python3
"""
groundwire proxy -- a TRANSPARENT drop-in memory layer for Ollama.

The trick: groundwire takes Ollama's port. Point the real Ollama at another port
and let groundwire own 11434, so every existing client -- the Ollama desktop app,
Open WebUI, curl, LangChain -- flows through groundwire with ZERO client config:

    Ollama app ──▶ :11434 (groundwire) ──▶ :11435 (real ollama)
                      │
                      └─ retrieve top-k chunks from the index, inject as context

groundwire speaks the NATIVE Ollama API. Everything is proxied byte-for-byte
(so /api/tags keeps the model dropdown working, streaming passes straight
through); only /api/chat and /v1/chat/completions get a retrieved-context
system message spliced in before forwarding upstream.

Files to answer from just get dropped in a folder (--watch); see watcher.py.

    # move the real ollama off 11434, then:
    OLLAMA_HOST=127.0.0.1:11435 ollama serve
    python -m groundwire.proxy --listen 11434 --upstream 127.0.0.1:11435 \
        --watch ~/groundwire/watch

Stdlib only (retrieval is groundwire; the file watcher is optional, see watcher.py).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pipeline import Groundwire
from .spans import SpanRegistry, QUOTE_INTENT, READ_INTENT

READ_MAX_CTX = 8192      # cap when we grow the window to fit a span to summarize

# Paths we splice retrieved context into. Everything else is a blind passthrough.
INJECT_PATHS = ("/api/chat", "/v1/chat/completions")

CONTEXT_HEADER = (
    "The CONTEXT below is retrieved from the user's own indexed files. Ground "
    "your answer in it and cite the file paths you used. Do NOT invent quotes, "
    "names, or facts. If asked to quote or reproduce exact text and that exact "
    "passage is not present in the CONTEXT, say you could not find it in the "
    "indexed files -- never fabricate a passage from memory. If the CONTEXT does "
    "not contain the answer, say so plainly rather than guessing.\n\n"
    "=== CONTEXT ===\n"
)


class ProxyState:
    """Shared, hot-swappable retrieval state. The watcher rebuilds `mem` on file
    changes and assigns it here; the proxy reads it per request. `enabled`
    toggles injection (the tray 'pause' switch) without dropping the proxy."""

    def __init__(self, mem: Groundwire, k: int = 6, enabled: bool = True,
                 tag_sources: bool = True, spans: SpanRegistry = None):
        self.mem = mem
        self.k = k
        self.enabled = enabled
        self.tag_sources = tag_sources     # append an audit footer to answers
        self.spans = spans or SpanRegistry()  # [[SPAN:...]] post-fill registry
        self.nbytes = 0                        # indexed text bytes (for the tray)
        self.watch_dir = None                  # set by main(); lets /groundwire/reindex rebuild
        self.backend = "sqlite_fts"
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "injected": 0, "filled": 0}

    def retrieve_block(self, query: str):
        """Return (context_string, sources) for a query, or (None, []) to inject
        nothing. `sources` is [(title, cid, score), ...] -- the audit trail of
        exactly which chunks were fed to the model for this answer."""
        with self.lock:
            mem, k, on = self.mem, self.k, self.enabled
        if not on or mem is None or mem._next == 0 or not query.strip():
            return None, []
        hits = mem.retrieve(query, k=k)
        if not hits:
            return None, []
        parts, sources = [], []
        for cid, text, score in hits:
            src = mem.source_of(cid)
            parts.append(f"[{src}]\n{text}" if src else text)
            sources.append((src or str(cid), cid, score))
        block = CONTEXT_HEADER + "\n\n---\n\n".join(parts) + "\n=== END CONTEXT ==="
        return block, sources


def sources_footer(sources) -> str:
    """A compact, human-auditable footer listing the chunks retrieved for an
    answer: 'groundwire ▸ drop/war_and_peace.txt #24 #5 · drop/notes.md #3'."""
    from collections import OrderedDict
    by = OrderedDict()
    for title, cid, _ in sources:
        by.setdefault(title, []).append(cid)
    parts = [f"{t} " + " ".join(f"#{c}" for c in cids) for t, cids in by.items()]
    return "\n\n— groundwire ▸ retrieved: " + "  ·  ".join(parts)


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # OpenAI content-parts form
                return " ".join(p.get("text", "") for p in c
                                if isinstance(p, dict))
    return ""


def _inject(body: dict, state: ProxyState, native: bool = True):
    """Route a chat request three ways and splice the right context in as system
    messages. Returns (body, sources, cands): `cands` non-empty means a span was
    OFFERED (a verbatim quote) and the response must be post-filled.

    - READ a located span (summarize/explain/analyze chapter N): inject the
      passage TEXT so the model can read it, and grow the context to fit it.
    - QUOTE a located span (quote / how does chapter N start): offer a handle;
      the model references it and groundwire post-fills verbatim.
    - otherwise: normal top-k retrieval."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body, [], []
    question = _last_user_text(messages)
    cands = state.spans.resolve(question)          # structural refs (may be [])

    if cands and READ_INTENT.search(question):
        h, label = cands[0]
        full = state.spans.text_of(h) or ""
        num_ctx = min(READ_MAX_CTX, len(full) // 4 + 1200)
        text = state.spans.text_of(h, cap_chars=(num_ctx - 1000) * 4)
        ctx = CONTEXT_HEADER + f"[{label}]\n{text}\n=== END CONTEXT ==="
        body["messages"] = [{"role": "system", "content": ctx}] + messages
        if native:                                 # grow the window to fit it
            body.setdefault("options", {})["num_ctx"] = num_ctx
        with state.lock:
            state.stats["injected"] += 1
        return body, [(state.spans.source_of(h), label.split(" of ")[0], 0.0)], []

    if QUOTE_INTENT.search(question):
        footer_src = None
        if not cands and state.mem is not None and state.mem._next:
            # content-anchored quote ("quote the passage about X"): locate the
            # passage by retrieval, register it as a span, offer it for verbatim
            # post-fill. Degrades to whatever retrieval found (audit footer shows
            # it); the fallback patches it in if the model fumbles the handle.
            hits = state.mem.retrieve(question, k=1)
            if hits:
                cid, text, _ = hits[0]
                footer_src = state.mem.source_of(cid) or "?"
                lbl = f"the passage matching your request (from {footer_src})"
                cands = [(state.spans.register_text_span(lbl, text), lbl)]
        if cands:
            # ONLY the span offer -- deliberately NOT the retrieval block, whose
            # "say you couldn't find it" header makes the model decline instead
            # of using the handle. The span (or the fallback) is the answer.
            h, label = cands[0]
            body["messages"] = [{"role": "system",
                                 "content": state.spans.offer(cands)}] + messages
            with state.lock:
                state.stats["injected"] += 1
            if footer_src:                              # content span
                src, tag = footer_src, "passage"
            else:                                       # structural span
                src, tag = state.spans.source_of(h) or "?", label.split(" of ")[0]
            return body, [(src, tag, 0.0)], cands

    block, sources = state.retrieve_block(question)
    if block:
        body["messages"] = [{"role": "system", "content": block}] + messages
        with state.lock:
            state.stats["injected"] += 1
        return body, sources, []
    return body, [], []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # shared per-process
    STATE: ProxyState = None
    UPSTREAM: str = "127.0.0.1:11435"

    def log_message(self, *a):  # quiet; the tray shows stats
        pass

    # ------------------------------------------------------------------ #
    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def _write_chunk(self, data: bytes):
        self.wfile.write(f"{len(data):X}\r\n".encode())
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _forward(self, method: str, body: bytes, footer: str = "",
                 stream: bool = True, fill: bool = False, fallback: str = ""):
        """Proxy to the upstream Ollama. Normally byte-for-byte (streaming
        preserved). When `footer` is set (audit tag), splice it into the answer.
        When `fill` is set (a span was offered), the upstream was forced to
        buffer so we can replace [[SPAN:...]] handles with verbatim stored text;
        `stream` still reflects what the CLIENT asked for, so the filled answer
        is re-emitted as stream frames or one JSON accordingly."""
        url = f"http://{self.UPSTREAM}{self.path}"
        req = urllib.request.Request(url, data=body if body else None,
                                     method=method)
        # copy client headers except hop-by-hop / length (we set our own)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection",
                                 "accept-encoding"):
                req.add_header(k, v)
        native = self.path.startswith("/api")
        try:
            with urllib.request.urlopen(req, timeout=600) as up:
                # span post-fill, or buffered footer: read the whole answer,
                # rewrite message.content, re-emit per the client's stream pref.
                if fill or (footer and not stream):
                    return self._forward_buffered(up, footer, native, fill=fill,
                                                  client_stream=stream,
                                                  fallback=fallback)
                self.send_response(up.status)
                ct = up.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ct)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if footer and stream and native:
                    # native NDJSON stream: emit an extra content frame carrying
                    # the audit footer just before the upstream's done:true frame
                    tagged = json.dumps({"message": {"role": "assistant",
                             "content": footer}, "done": False}).encode() + b"\n"
                    for line in up:
                        try:
                            done = json.loads(line).get("done")
                        except ValueError:
                            done = False
                        if done:
                            self._write_chunk(tagged)
                        self._write_chunk(line)
                else:
                    while True:
                        chunk = up.read(2048)
                        if not chunk:
                            break
                        self._write_chunk(chunk)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type",
                             e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:  # upstream down, timeout, etc.
            msg = json.dumps({"error": f"groundwire proxy: upstream error: {e}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def _forward_buffered(self, up, footer: str, native: bool,
                          fill: bool = False, client_stream: bool = False,
                          fallback: str = ""):
        """Read the whole upstream JSON answer, post-fill [[SPAN:...]] handles
        with verbatim stored text (if `fill`), append the audit footer, then
        re-emit -- as native NDJSON stream frames when the client asked to
        stream, else as one buffered JSON with a Content-Length. `fallback` is a
        handle whose span is appended if the model emitted no valid handle on a
        quote request (so a fumbled protocol still returns the passage)."""
        raw = up.read()
        obj, content = None, None
        try:
            obj = json.loads(raw)
            msg = obj["message"] if native else obj["choices"][0]["message"]
            content = msg.get("content", "")
            if fill and self.STATE is not None:
                new = self.STATE.spans.fill(content)
                if new == content and fallback:
                    # model didn't emit the handle -> patch the resolved span in
                    span = self.STATE.spans.text_of(fallback)
                    if span:
                        new = (content.rstrip() + "\n\n" + span) if content.strip() \
                            else span
                if new != content:
                    with self.STATE.lock:
                        self.STATE.stats["filled"] += 1
                content = new
            content += footer
            msg["content"] = content
        except (ValueError, KeyError, IndexError, TypeError):
            obj = None                              # unexpected shape -> as-is

        if client_stream and native and obj is not None:
            # client wanted a stream but we buffered: hand it back as two frames
            self.send_response(up.status)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._write_chunk(json.dumps({"model": obj.get("model", ""),
                "message": {"role": "assistant", "content": content},
                "done": False}).encode() + b"\n")
            done = dict(obj)
            done["message"] = {"role": "assistant", "content": ""}
            done["done"] = True
            self._write_chunk(json.dumps(done).encode() + b"\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        raw = json.dumps(obj).encode() if obj is not None else raw
        self.send_response(up.status)
        self.send_header("Content-Type",
                         up.headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _respond_direct(self, content: str, native: bool, client_stream: bool,
                        model: str):
        """Send a synthesized answer WITHOUT calling upstream (a map-reduce
        result). Native NDJSON frames if the client streamed, else one JSON."""
        if native and client_stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._write_chunk(json.dumps({"model": model, "message":
                {"role": "assistant", "content": content},
                "done": False}).encode() + b"\n")
            self._write_chunk(json.dumps({"model": model, "message":
                {"role": "assistant", "content": ""},
                "done": True}).encode() + b"\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        if native:
            raw = json.dumps({"model": model, "done": True, "message":
                {"role": "assistant", "content": content}}).encode()
        else:
            raw = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _upstream_complete(self, model: str, prompt: str, num_ctx: int) -> str:
        """One non-streaming chat completion from the upstream (used as the
        `call_llm` for map-reduce). Returns just the answer text."""
        payload = {"model": model, "stream": False,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"num_ctx": num_ctx}}
        req = urllib.request.Request(f"http://{self.UPSTREAM}/api/chat",
                                     data=json.dumps(payload).encode(),
                                     method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=600) as up:
            return json.loads(up.read())["message"]["content"]

    def _map_reduce_summary(self, model: str, question: str, source: str,
                            on_progress=None) -> str:
        """Summarize a whole file that's too big for the window: segment by the
        coarsest structural unit and map-reduce over the segments."""
        from .mapreduce import summarize
        segments = self.STATE.spans.coarse_segments(source)
        budget = (READ_MAX_CTX - 1200) * 4
        print(f"groundwire: map-reduce summary of {source} over {len(segments)} segments")
        return summarize(
            lambda p: self._upstream_complete(model, p, READ_MAX_CTX),
            segments, budget, question, on_progress=on_progress)

    def _map_reduce_stream(self, model: str, question: str, source: str):
        """Stream a map-reduce summary: a chunked NDJSON response that emits a
        progress line as each segment completes, then the final summary. Keeps
        the Ollama app alive (and informative) through a multi-minute run."""
        segments = self.STATE.spans.coarse_segments(source)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def frame(text, done=False):
            self._write_chunk(json.dumps({"model": model, "done": done,
                "message": {"role": "assistant", "content": text}}).encode() + b"\n")

        frame(f"Summarizing {source} in {len(segments)} parts — this runs one "
              f"pass per part.\n\n")
        try:
            final = self._map_reduce_summary(
                model, question, source,
                on_progress=lambda m: frame(f"▸ {m}\n"))
            frame(f"\n{final}\n\n— groundwire ▸ map-reduce summary of {source}")
        except Exception as e:
            frame(f"\n[groundwire: map-reduce failed: {e}]")
        frame("", done=True)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _large_read_source(self, question: str):
        """If this is a 'summarize the whole book' request whose target is too
        big for one context, return the source; else None."""
        st = self.STATE
        if not (self.path.startswith("/api") and st and question
                and READ_INTENT.search(question) and not st.spans.resolve(question)):
            return None
        src = st.spans.resolve_whole(question)
        if src and len(st.spans.texts.get(src, "")) > (READ_MAX_CTX - 1200) * 4:
            return src
        return None

    def _handle(self, method: str):
        if self.STATE is not None:
            with self.STATE.lock:
                self.STATE.stats["requests"] += 1
        body = self._read_body()
        footer, stream, fill, fallback = "", True, False, ""
        if method == "POST" and self.path in INJECT_PATHS and body:
            try:
                parsed = json.loads(body)
                # LARGE READ: "summarize the whole book" -> map-reduce, answered
                # directly (many upstream calls, then one synthesized response).
                q = _last_user_text(parsed.get("messages"))
                src = self._large_read_source(q)
                if src is not None:
                    model = parsed.get("model", "")
                    if parsed.get("stream", True):
                        # long run -> stream progress frames as each book is done
                        return self._map_reduce_stream(model, q, src)
                    ans = (self._map_reduce_summary(model, q, src)
                           + f"\n\n— groundwire ▸ map-reduce summary of {src}")
                    return self._respond_direct(ans, True, False, model)
                # /api/chat streams by default; /v1 does not. Keep the CLIENT's
                # preference for the reply shape even if we buffer upstream.
                stream = parsed.get("stream", self.path.startswith("/api"))
                parsed, sources, cands = _inject(
                    parsed, self.STATE, native=self.path.startswith("/api"))
                if cands:
                    # a span was offered -> we must post-fill the answer, which
                    # needs the whole thing, so force the upstream to buffer.
                    # Keep the top handle as a fallback if the model fumbles it.
                    fill = True
                    fallback = cands[0][0]
                    parsed["stream"] = False
                body = json.dumps(parsed).encode()
                if sources and self.STATE and self.STATE.tag_sources:
                    footer = sources_footer(sources)
            except (ValueError, TypeError):
                pass  # not JSON we understand -> pass through unchanged
        self._forward(method, body, footer=footer, stream=stream, fill=fill,
                      fallback=fallback)

    def do_GET(self):
        if self.path == "/groundwire/ping":
            return self._ping()
        self._handle("GET")

    def _ping(self):
        """Health marker so a second launch can tell three cases apart: nothing
        here, a HEALTHY groundwire (just open the app), or a DEGRADED groundwire whose
        upstream ollama died (revive it). Reports upstream reachability so the
        singleton check isn't fooled by a zombie proxy over a dead backend."""
        st = self.STATE
        chunks = st.mem._next if (st and st.mem) else 0
        nbytes = getattr(st, "nbytes", 0) if st else 0
        toks = est_tokens(nbytes)
        body = json.dumps({"groundwire": True, "chunks": chunks,
                           "bytes": nbytes, "tokens": toks,
                           "gpu": gpu_rig(full_attention_gb(toks)),
                           "enabled": st.enabled if st else True,
                           "injected": st.stats["injected"] if st else 0,
                           "requests": st.stats["requests"] if st else 0,
                           "upstream": self.UPSTREAM,
                           "upstream_ok": self._upstream_alive()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _upstream_alive(self) -> bool:
        import socket
        host, _, port = self.UPSTREAM.partition(":")
        try:
            with socket.create_connection((host, int(port or 11434)), timeout=0.4):
                return True
        except (OSError, ValueError):
            return False

    def do_POST(self):
        if self.path == "/groundwire/toggle":
            return self._toggle()
        if self.path == "/groundwire/reindex":
            return self._reindex()
        self._handle("POST")

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _toggle(self):
        """Flip injection on/off -- the tray's Injection checkbox."""
        st = self.STATE
        if not st:
            return self._json({"error": "no state"}, 503)
        with st.lock:
            st.enabled = not st.enabled
            en = st.enabled
        self._json({"enabled": en})

    def _reindex(self):
        """Force a rebuild from the watch folder -- the tray's Reindex now."""
        st = self.STATE
        if not st or not st.watch_dir:
            return self._json({"error": "no watch dir configured"}, 400)
        mem = build_index(st.watch_dir, st.backend, st.k)
        spans = build_spans(st.watch_dir)
        nbytes = index_text_bytes(mem)
        with st.lock:
            st.mem = mem
            st.spans = spans
            st.nbytes = nbytes
        self._json({"reindexed": True, "chunks": mem._next})

    def do_DELETE(self):
        self._handle("DELETE")


def build_index(watch_dir: str | None, backend: str = "sqlite_fts",
                k: int = 6) -> Groundwire:
    """Build the retrieval index from the watch folder (empty index if none)."""
    mem = Groundwire(memory=backend, k=k)
    if watch_dir and os.path.isdir(watch_dir):
        mem.ingest_repo(watch_dir, prefix="drop")
    return mem


def build_spans(watch_dir: str | None) -> SpanRegistry:
    """Build the span registry (addressable verbatim passages) from the folder."""
    reg = SpanRegistry()
    if watch_dir and os.path.isdir(watch_dir):
        reg.build_from_folder(watch_dir)
    return reg


def index_text_bytes(mem: Groundwire) -> int:
    """Total bytes of indexed text -- the real 'how much memory is held' number
    for the tray, not an estimate. Uses the backend's get() over every chunk;
    cheap at build time, not per request."""
    get = getattr(mem.memory, "get", None)
    if not get:
        return 0
    total = 0
    for i in range(mem._next):
        t = get(i)
        if t:
            total += len(t.encode("utf-8"))
    return total


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def est_tokens(nbytes: int) -> int:
    """Rough token count from indexed text bytes (~4 chars/token, English)."""
    return nbytes // 4


def human_tokens(n: int) -> str:
    if n < 1000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n / 1000:.0f}K"
    return f"{n / 1_000_000:.2f}M"


def full_attention_gb(tokens: int) -> float:
    """VRAM a 7B model would need to hold this many tokens in a full-attention
    KV cache. Fits the README table exactly: ~16 GB weights + ~53 GB per 1M
    tokens of fp16 KV (256K->29, 1M->69, 2M->122, 3M->175 GB)."""
    return 16.0 + 53.0 * (tokens / 1_000_000)


def gpu_rig(gb: float) -> str:
    """The GPU/rig you'd have to buy to hold that KV cache on-device."""
    if gb <= 24:
        return "a 24 GB GPU"
    if gb <= 48:
        return "a 48 GB GPU"
    if gb <= 80:
        return "an 80 GB GPU"
    if gb <= 160:
        return "a 2x80 GB rig (160 GB)"
    if gb <= 256:
        return "a 4x80 GB rig (~256 GB)"
    return f"a ~{int(round(gb / 80))}x80 GB rig ({int(gb)} GB)"


def _signature(watch_dir: str) -> tuple:
    """A cheap fingerprint of the folder: (path, mtime, size) for every file.
    Any drop / edit / delete / rename changes it, triggering a rebuild."""
    sig = []
    for dirpath, dirnames, filenames in os.walk(watch_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                sig.append((p, int(st.st_mtime), st.st_size))
            except OSError:
                pass
    return tuple(sorted(sig))


def watch_loop(watch_dir: str, state: ProxyState, backend: str, k: int,
               interval: float = 1.0, on_change=None):
    """Poll the folder; on any change, rebuild the index and hot-swap it into
    the shared state. Full rebuild (not incremental) -- lexical ingest is
    ~instant and rebuild handles deletes/renames for free. Dependency-free.
    `on_change(mem, n_files)` fires after each rebuild (the tray uses it to
    refresh the context-held indicator)."""
    import time
    last = _signature(watch_dir)
    while True:
        try:
            time.sleep(interval)
            cur = _signature(watch_dir)
            if cur == last:
                continue
            last = cur
            mem = build_index(watch_dir, backend, k)
            spans = build_spans(watch_dir)
            nbytes = index_text_bytes(mem)
            with state.lock:
                state.mem = mem
                state.spans = spans
                state.nbytes = nbytes
            print(f"groundwire proxy: reindexed {watch_dir} "
                  f"-> {mem._next} chunks ({len(cur)} files)")
            if on_change:
                try:
                    on_change(mem, len(cur))
                except Exception:
                    pass
        except Exception as e:            # never let the watcher kill the proxy
            print(f"groundwire proxy: watch error: {e}")


def serve(listen_port: int, upstream: str, state: ProxyState):
    Handler.STATE = state
    Handler.UPSTREAM = upstream
    srv = ThreadingHTTPServer(("127.0.0.1", listen_port), Handler)
    print(f"groundwire proxy: listening on 127.0.0.1:{listen_port} "
          f"-> upstream {upstream}  (index: {state.mem._next} chunks)")
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Transparent groundwire proxy for Ollama")
    ap.add_argument("--listen", type=int, default=11434,
                    help="port groundwire listens on (take over Ollama's 11434)")
    ap.add_argument("--upstream", default=os.environ.get("GROUNDWIRE_UPSTREAM",
                    "127.0.0.1:11435"), help="the REAL ollama host:port")
    ap.add_argument("--watch", default=None,
                    help="folder whose files are ingested (drop files here)")
    ap.add_argument("--backend", default="sqlite_fts")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    mem = build_index(args.watch, args.backend, args.k)
    state = ProxyState(mem, k=args.k, spans=build_spans(args.watch))
    state.nbytes = index_text_bytes(mem)
    state.watch_dir = args.watch
    state.backend = args.backend
    if args.watch:
        t = threading.Thread(target=watch_loop,
                             args=(args.watch, state, args.backend, args.k),
                             daemon=True)
        t.start()
        print(f"groundwire proxy: watching {args.watch} for dropped files")
    serve(args.listen, args.upstream, state)


if __name__ == "__main__":
    main()
