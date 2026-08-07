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

import os
import socket
import sys
import threading
import time
import urllib.request

from .webapp import serve

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _icon_path():
    """The app/window icon: a multi-size .ico on Windows (taskbar quality), a
    PNG elsewhere (GTK/Cocoa expect that). None if the asset is missing."""
    ico = os.path.join(_ASSETS, "groundwire.ico")
    png = os.path.join(_ASSETS, "groundwire.png")
    if sys.platform == "win32" and os.path.exists(ico):
        return ico
    return png if os.path.exists(png) else None


def _set_app_id():
    """Give the process an explicit AppUserModelID so Windows shows OUR window
    icon in the taskbar instead of grouping it under the pythonw host icon."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "tina4stack.groundwire.app")
        except Exception:
            pass


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

    _set_app_id()
    port = _free_port()
    threading.Thread(target=serve, args=("127.0.0.1", port), daemon=True).start()
    if not _wait_until_up(port):
        print("groundwire: server did not start in time")
        return
    webview.create_window("groundwire", f"http://127.0.0.1:{port}",
                          width=1200, height=820, min_size=(760, 560))
    icon = _icon_path()                              # taskbar / window icon
    webview.start(**({"icon": icon} if icon else {}))  # blocks until window closes


if __name__ == "__main__":
    main()
