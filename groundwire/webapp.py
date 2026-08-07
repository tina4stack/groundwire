"""
The groundwire app server: a stdlib HTTP server exposing the chat Session as a
JSON + SSE API and serving the web UI. This is the seam the desktop shell (a
native webview) and any browser plug into -- groundwire owns the whole turn, so
there is no Ollama port takeover.

    python -m groundwire.webapp            # serves http://127.0.0.1:8770

API:
    GET  /api/models                       -> {models:[...], default}
    GET  /api/sources                      -> [{id,path,scope,enabled,local_only}]
    POST /api/sources    {path,scope,local_only}   -> add + reindex
    POST /api/sources/toggle   {id,enabled}        -> enable/disable + reindex
    POST /api/sources/remove   {id}                -> delete + reindex
    GET  /api/conversations                -> [{id,title,model,updated}]
    GET  /api/conversations/<id>           -> {..., messages:[...]}
    POST /api/conversations/delete {id}
    POST /api/chat  {conv_id?, message, backend}   -> Server-Sent Events stream
"""
from __future__ import annotations

import json
import os
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import make_session, save_config

WEBUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
_CTYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
           ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class App:
    """Holds the session + store; the handler calls into this."""
    def __init__(self, session, store, cfg):
        self.session, self.store, self.cfg = session, store, cfg


class Handler(BaseHTTPRequestHandler):
    APP: App = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- io helpers ---------------------------------------------------------- #
    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return {}

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _static(self, rel):
        path = os.path.normpath(os.path.join(WEBUI, rel.lstrip("/")))
        if not path.startswith(WEBUI) or not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type",
                         _CTYPES.get(os.path.splitext(path)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routing ------------------------------------------------------------- #
    def do_GET(self):
        p, app = self.path.split("?", 1)[0], self.APP
        if p in ("/", "/index.html"):
            return self._static("index.html")
        if p.startswith("/static/"):
            return self._static(p[len("/static/"):])
        if p == "/api/models":
            return self._json({"models": sorted(app.session.backends),
                               "default": app.session.default_backend})
        if p == "/api/sources":
            return self._json(app.store.list_paths())
        if p == "/api/conversations":
            return self._json(app.store.list_conversations())
        if p.startswith("/api/conversations/"):
            cid = p.rsplit("/", 1)[-1]
            conv = app.store.get_conversation(int(cid)) if cid.isdigit() else None
            return self._json(conv or {"error": "not found"},
                              200 if conv else 404)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        p, app, b = self.path.split("?", 1)[0], self.APP, self._body()
        if p == "/api/chat":
            return self._chat(b)
        if p == "/api/inspect":
            # DEBUG PANEL: the exact context that WOULD be injected, no model call
            return self._json(app.session.inspect(b.get("message", ""),
                                                  b.get("backend")))
        if p == "/api/sources":
            scope = b.get("scope") or os.path.basename(b["path"].rstrip("/\\")) or "docs"
            app.store.add_path(b["path"], scope, bool(b.get("local_only")))
            app.session.reindex()
            return self._json(app.store.list_paths())
        if p == "/api/sources/toggle":
            app.store.set_path_enabled(int(b["id"]), bool(b["enabled"]))
            app.session.reindex()
            return self._json(app.store.list_paths())
        if p == "/api/sources/remove":
            app.store.remove_path(int(b["id"]))
            app.session.reindex()
            return self._json(app.store.list_paths())
        if p == "/api/conversations/delete":
            app.store.delete_conversation(int(b["id"]))
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)

    # -- the streaming chat turn (SSE) -------------------------------------- #
    def _chat(self, b):
        app = self.APP
        msg, backend = b.get("message", ""), b.get("backend")
        conv_id = b.get("conv_id")
        if not app.session.backends:
            return self._json({"error": "no backends configured"}, 400)
        if conv_id is None:
            conv_id = app.store.new_conversation(msg[:60], backend
                                                 or app.session.default_backend)
        app.store.add_message(conv_id, "user", msg)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def sse(event, data):
            frame = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
            self.wfile.write(f"{len(frame):X}\r\n".encode() + frame + b"\r\n")
            self.wfile.flush()

        sse("meta", {"conv_id": conv_id})
        parts = []
        try:
            for delta in app.session.turn(conv_id, msg, backend):
                parts.append(delta)
                sse("delta", {"text": delta})
        except Exception as e:
            sse("error", {"message": str(e)})
        answer = "".join(parts)
        app.store.add_message(conv_id, "assistant", answer,
                              model=backend or app.session.default_backend)
        sse("done", {"conv_id": conv_id})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_DELETE(self):
        p, app = self.path.split("?", 1)[0], self.APP
        if p.startswith("/api/conversations/"):
            cid = p.rsplit("/", 1)[-1]
            if cid.isdigit():
                app.store.delete_conversation(int(cid))
                return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8770):
    session, store, cfg = make_session()
    Handler.APP = App(session, store, cfg)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"groundwire: http://{host}:{port}  "
          f"(backends: {', '.join(session.backends) or 'none — configure one'})")
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="groundwire app server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
