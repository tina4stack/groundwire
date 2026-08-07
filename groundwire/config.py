"""
App configuration: where the store lives, which backends exist, and how to
assemble a Session for the desktop app / web server.

Local Ollama models are auto-discovered (each installed model becomes a
selectable backend). Cloud backends come from ~/.groundwire/config.json, with
API keys resolved from the ENVIRONMENT (or an injected getter) -- never stored
in the config file.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .backends import make_backend, OllamaBackend
from .store import Store
from .app import Session

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".groundwire")
DB_PATH = os.path.join(CONFIG_DIR, "groundwire.db")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_CONFIG = {
    # the LOCAL ollama this app talks to as a client (its normal port, since the
    # app no longer takes it over). Override with GROUNDWIRE_OLLAMA_HOST.
    "ollama_host": "127.0.0.1:11434",
    # cloud backends, e.g. {"name": "Gemini Flash", "type": "gemini",
    #   "model": "gemini-2.0-flash"}  (key from GEMINI_API_KEY)
    "backends": [],
    "default_backend": None,
}


def load_config(path: str = CONFIG_PATH) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            cfg.update(json.load(open(path, encoding="utf-8")))
        except (ValueError, OSError):
            pass
    return cfg


def save_config(cfg: dict, path: str = CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def ollama_host(cfg: dict) -> str:
    """The Ollama host the app talks to (env override wins)."""
    return os.environ.get("GROUNDWIRE_OLLAMA_HOST", cfg.get("ollama_host")
                          or DEFAULT_CONFIG["ollama_host"])


def ollama_status(host: str) -> dict:
    """Probe a local Ollama: is it reachable, and which CHAT models does it have?
    Distinguishes 'not running' (connection refused) from 'running, no models'."""
    probe = OllamaBackend(host)                        # normalises host -> http://…
    try:
        with urllib.request.urlopen(probe.host + "/api/tags", timeout=3) as r:
            data = json.loads(r.read())
        names = [m.get("name", "") for m in data.get("models", [])]
        return {"host": host, "running": True,
                "models": [m for m in names if m and "embed" not in m.lower()]}
    except Exception:
        return {"host": host, "running": False, "models": []}


def apply_config(cfg: dict, change: dict) -> dict:
    """Apply a Setup-screen change to `cfg` in place (pure dict mutation, no I/O).
    Recognised keys: `ollama_host`, `add_backend` {name,type,model}, `remove_backend`
    (name), `default`. Adding a backend de-dupes by name. Returns cfg."""
    if "ollama_host" in change:
        cfg["ollama_host"] = ((change["ollama_host"] or "").strip()
                              or DEFAULT_CONFIG["ollama_host"])
    ab = change.get("add_backend")
    if ab and ab.get("type") and ab.get("model"):
        name = (ab.get("name") or ab.get("model") or "").strip()
        if name:
            cfg["backends"] = [x for x in cfg.get("backends", [])
                               if x.get("name") != name]
            cfg["backends"].append({"name": name, "type": ab["type"],
                                    "model": ab["model"].strip()})
    if change.get("remove_backend"):
        cfg["backends"] = [x for x in cfg.get("backends", [])
                           if x.get("name") != change["remove_backend"]]
    if "default" in change:
        cfg["default_backend"] = change["default"] or None
    return cfg


def build_backends(cfg: dict, get_key=None):
    """Return {name: Backend}. Local Ollama models are discovered live; cloud
    backends come from config (keys via get_key / env)."""
    backends = {}
    host = os.environ.get("GROUNDWIRE_OLLAMA_HOST", cfg.get("ollama_host"))
    if host:
        probe = OllamaBackend(host)
        for m in probe.models():                       # e.g. "qwen2.5:7b"
            if "embed" in m.lower():                   # embedders aren't chat models
                continue
            backends[m] = OllamaBackend(host, m)
    for entry in cfg.get("backends", []):
        try:
            backends[entry["name"]] = make_backend(entry, get_key)
        except Exception as e:                         # bad entry / missing key
            print(f"groundwire: skipping backend {entry.get('name')!r}: {e}")
    return backends


def make_session(cfg: dict = None, db_path: str = None, get_key=None):
    """Assemble (session, store, cfg) ready for the server."""
    cfg = cfg or load_config()
    store = Store(db_path or DB_PATH)      # read DB_PATH at call time (overridable)
    backends = build_backends(cfg, get_key)
    default = cfg.get("default_backend")
    if default not in backends:
        default = next(iter(backends), None)
    session = Session(store, backends, default)
    return session, store, cfg
