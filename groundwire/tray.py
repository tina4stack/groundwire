#!/usr/bin/env python3
"""
groundwire tray -- the desktop app. One icon that runs the whole drop-in memory
layer: it starts the real Ollama on an upstream port, puts the groundwire proxy on
Ollama's normal port (11434) so every client gets memory, watches your drop
folder, and shows how much CONTEXT is currently held.

    python -m groundwire.tray                       # full takeover (11434 <- 11435)
    python -m groundwire.tray --listen 8899 --upstream 127.0.0.1:11434 --no-ollama
                                                # sit beside a running Ollama

Tray menu:
    ● Context held: N chunks · ~T tokens        (the live indicator)
    Injection: On/Off                           (pause without dropping proxy)
    Open drop folder
    Reindex now
    Quit

Deps: pystray + Pillow (pip install "tina4-groundwire[app]"). Everything else is
groundwire.proxy, which is stdlib.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request

from PIL import Image
import pystray

from . import proxy as P


def groundwire_status(listen: int):
    """Ping the port and classify: returns the ping dict if a groundwire proxy is
    there (with `upstream_ok`), or None if the port is free / not groundwire. Lets
    a second launch tell a HEALTHY singleton from a DEGRADED one (dead upstream)
    from a free port."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{listen}/groundwire/ping", timeout=1) as r:
            d = json.loads(r.read())
            return d if d.get("groundwire") is True else None
    except Exception:
        return None

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WATCH_DEFAULT = os.path.join(os.path.expanduser("~"), "groundwire", "watch")
# empirical average from our corpora (NIAH, books, code): ~400 tokens/chunk.
TOKENS_PER_CHUNK = 400


def _port_open(host: str, port: int, timeout=0.4) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_ollama(upstream: str):
    """Start the real ollama on the upstream host:port if nothing answers there
    yet. Returns True if a backend is reachable."""
    host, _, port = upstream.partition(":")
    port = int(port or 11434)
    if _port_open(host, port):
        return True
    env = dict(os.environ, OLLAMA_HOST=f"{host}:{port}", OLLAMA_KEEP_ALIVE="-1")
    try:
        subprocess.Popen(["ollama", "serve"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return False
    for _ in range(30):
        if _port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def kill_existing_ollama():
    """Kill any running Ollama so groundwire owns exactly one runtime. The desktop
    app auto-starts its own server and collides with groundwire's port-takeover of
    11434. On macOS the app is an Electron app (process name 'Ollama', capital)
    that auto-RESURRECTS via the launchd item 'com.ollama.ollama' -- so we boot
    that out first, then kill the app + the CLI serve by exact process name."""
    if os.name == "nt":
        cmds = [["taskkill", "/F", "/IM", "ollama app.exe"],
                ["taskkill", "/F", "/IM", "ollama.exe"]]
    else:
        uid = os.getuid()
        cmds = [["launchctl", "bootout", f"gui/{uid}/com.ollama.ollama"],
                ["pkill", "-x", "Ollama"],      # macOS desktop app (Electron)
                ["pkill", "-x", "ollama"]]      # CLI serve
    for c in cmds:
        try:
            subprocess.run(c, capture_output=True, timeout=10)
        except Exception:
            pass


def find_ollama_app() -> str | None:
    """Locate the Ollama DESKTOP app (the chat GUI), distinct from ollama.exe
    (the server). Windows install paths first, then a couple of fallbacks."""
    cands = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "Ollama", "ollama app.exe"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Ollama",
                     "ollama app.exe"),
        "/Applications/Ollama.app",                      # macOS
    ]
    return next((p for p in cands if p and os.path.exists(p)), None)


def open_ollama_app(listen_addr: str):
    """Launch the Ollama chat app pointed at groundwire's port, so the app talks
    THROUGH groundwire (which injects) instead of straight to the raw model. The
    app finds a healthy server already on that port (us) and reuses it rather
    than starting its own."""
    exe = find_ollama_app()
    if not exe:
        print("groundwire: Ollama desktop app not found; open it yourself.")
        return
    # force the app's OLLAMA_HOST to groundwire so its client connects to us
    env = dict(os.environ, OLLAMA_HOST=listen_addr)
    try:
        if exe.endswith(".app"):
            subprocess.Popen(["open", exe], env=env)
        else:
            subprocess.Popen([exe], env=env)
        print(f"groundwire: opened Ollama app (pointed at {listen_addr})")
    except OSError as e:
        print(f"groundwire: could not open Ollama app: {e}")


def load_icon(paused: bool = False) -> Image.Image:
    name = "groundwire-paused-64.png" if paused else "groundwire-64.png"
    path = os.path.join(ASSETS, name)
    if os.path.exists(path):
        return Image.open(path)
    # fallback: a plain square so the app still runs without committed assets
    return Image.new("RGBA", (64, 64), (37, 40, 61, 255))


class TrayApp:
    def __init__(self, watch, listen, upstream, backend="sqlite_fts", k=6,
                 start_ollama=True, open_app=True):
        self.watch = watch
        self.listen = listen
        self.upstream = upstream
        self.backend = backend
        self.k = k
        self.start_ollama = start_ollama
        self.open_app = open_app
        os.makedirs(watch, exist_ok=True)
        mem = P.build_index(watch, backend, k)
        self.state = P.ProxyState(mem, k=k, spans=P.build_spans(watch))
        self.nbytes = P.index_text_bytes(mem)
        self.icon = pystray.Icon("groundwire", load_icon(), "groundwire", self._menu())

    # -- live indicator ------------------------------------------------------ #
    def _context_label(self, _=None) -> str:
        toks = P.est_tokens(self.nbytes)
        return (f"Context held: {P.human_bytes(self.nbytes)} · "
                f"~{P.human_tokens(toks)} tokens")

    def _gpu_label(self, _=None) -> str:
        gb = P.full_attention_gb(P.est_tokens(self.nbytes))
        return f"Full attention would need {P.gpu_rig(gb)} — groundwire runs it off-GPU"

    def _status_label(self, _=None) -> str:
        s = self.state.stats
        on = "on" if self.state.enabled else "PAUSED"
        return f"groundwire {on}  ·  {s['injected']}/{s['requests']} injected"

    def _menu(self):
        return pystray.Menu(
            pystray.MenuItem(self._status_label, None, enabled=False),
            pystray.MenuItem(self._context_label, None, enabled=False),
            pystray.MenuItem(self._gpu_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Ollama chat",
                             lambda i, item: open_ollama_app(
                                 f"127.0.0.1:{self.listen}")),
            pystray.MenuItem("Injection",
                             lambda i, item: self._toggle(),
                             checked=lambda item: self.state.enabled),
            pystray.MenuItem("Open drop folder", lambda i, item: self._open()),
            pystray.MenuItem("Reindex now", lambda i, item: self._reindex()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda i, item: self._quit()),
        )

    # -- actions ------------------------------------------------------------- #
    def _toggle(self):
        with self.state.lock:
            self.state.enabled = not self.state.enabled
        self.icon.icon = load_icon(paused=not self.state.enabled)
        self.icon.update_menu()

    def _open(self):
        try:
            os.startfile(self.watch)          # Windows
        except AttributeError:
            subprocess.Popen(["xdg-open", self.watch])

    def _reindex(self):
        mem = P.build_index(self.watch, self.backend, self.k)
        spans = P.build_spans(self.watch)
        with self.state.lock:
            self.state.mem = mem
            self.state.spans = spans
        self._refresh(mem)

    def _refresh(self, mem=None, *a):
        if mem is not None:
            self.nbytes = P.index_text_bytes(mem)
        self.icon.title = self._context_label()
        self.icon.update_menu()

    def _quit(self):
        self.icon.stop()

    # -- lifecycle ----------------------------------------------------------- #
    def run(self):
        # singleton: decide from a full-stack health check, not just "port answers".
        status = groundwire_status(self.listen)
        if status is not None:
            if status.get("upstream_ok"):
                # healthy groundwire already running -> a click just opens the chat
                print("groundwire: already running -> opening Ollama app")
            else:
                # zombie: proxy alive but its upstream ollama died. Revive the
                # backend (don't start a second proxy -- the port is taken and
                # the existing proxy will forward again once ollama is back).
                up = status.get("upstream") or self.upstream
                print(f"groundwire: proxy up but upstream {up} is down -> reviving ollama")
                ensure_ollama(up)
            if self.open_app:
                open_ollama_app(f"127.0.0.1:{self.listen}")
            return
        if self.start_ollama:
            kill_existing_ollama()      # groundwire owns exactly one ollama
        if self.start_ollama and not ensure_ollama(self.upstream):
            print(f"groundwire: WARNING could not reach/start ollama at {self.upstream}")
        # proxy + watcher in daemon threads; pystray owns the main loop
        P.Handler.STATE = self.state
        P.Handler.UPSTREAM = self.upstream
        threading.Thread(target=self._serve, daemon=True).start()
        threading.Thread(
            target=P.watch_loop,
            args=(self.watch, self.state, self.backend, self.k),
            kwargs={"on_change": lambda mem, n: self._refresh(mem)},
            daemon=True).start()
        # give the proxy a beat to bind, then open the chat app through it
        if self.open_app:
            def _launch():
                time.sleep(1.5)
                if _port_open("127.0.0.1", self.listen):
                    open_ollama_app(f"127.0.0.1:{self.listen}")
            threading.Thread(target=_launch, daemon=True).start()
        self.icon.title = self._context_label()
        self.icon.run()

    def _serve(self):
        from http.server import ThreadingHTTPServer
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", self.listen), P.Handler)
            print(f"groundwire tray: proxy on 127.0.0.1:{self.listen} -> {self.upstream}")
            srv.serve_forever()
        except OSError as e:
            print(f"groundwire tray: cannot bind :{self.listen} ({e}). "
                  f"Is the Ollama app still holding it? Quit it and relaunch.")


def main():
    ap = argparse.ArgumentParser(description="groundwire desktop tray")
    ap.add_argument("--watch", default=WATCH_DEFAULT)
    ap.add_argument("--listen", type=int, default=11434)
    ap.add_argument("--upstream", default="127.0.0.1:11435")
    ap.add_argument("--backend", default="sqlite_fts")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--no-ollama", action="store_true",
                    help="don't try to start the upstream ollama (it's already "
                         "running, e.g. the desktop app on the upstream port)")
    ap.add_argument("--no-open-app", action="store_true",
                    help="don't launch the Ollama chat app on startup")
    args = ap.parse_args()
    TrayApp(args.watch, args.listen, args.upstream, args.backend, args.k,
            start_ollama=not args.no_ollama,
            open_app=not args.no_open_app).run()


if __name__ == "__main__":
    main()
