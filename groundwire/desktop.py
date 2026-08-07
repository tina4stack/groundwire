"""
The groundwire desktop app: a native window around the web UI.

Starts the groundwire server on a private localhost port in a background thread,
waits for it, then opens a native webview window pointed at it -- WKWebView on
macOS, WebView2 on Windows, GTK/WebKit on Linux. That's the whole "app": one
process, the OS's own webview, groundwire owning every turn. No Ollama port
takeover, so it behaves the same on macOS as on Windows.

    python -m groundwire.desktop          # opens the app window
"""
from __future__ import annotations

import socket
import threading
import time
import urllib.request

from .webapp import serve


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until_up(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/models"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(0.4)
    return False


def main():
    import webview                                   # imported here so the core
    # stays import-light and headless-testable without a GUI toolkit present.

    port = _free_port()
    threading.Thread(target=serve, args=("127.0.0.1", port), daemon=True).start()
    if not _wait_until_up(port):
        print("groundwire: server did not start in time")
        return
    webview.create_window("groundwire", f"http://127.0.0.1:{port}",
                          width=1200, height=820, min_size=(760, 560))
    webview.start()                                  # blocks until the window closes


if __name__ == "__main__":
    main()
