"""
The app store: one SQLite file holding chat history and the sanctioned-path
allowlist (alongside the FTS index / spans, which live in their own tables).

Two responsibilities:
  * conversations + messages -- persistent chat history, WITH provenance (which
    chunks each answer used) so history stays auditable after the fact.
  * sanctioned_paths -- the explicit allowlist of folders groundwire may index.
    Each carries `local_only`; `paths_for(cloud=True)` withholds those, so
    selecting a cloud model never ships a folder you marked local-only.

Stdlib sqlite3 only.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    title    TEXT,
    model    TEXT,
    created  TEXT,
    updated  TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role     TEXT NOT NULL,
    content  TEXT NOT NULL,
    model    TEXT,
    sources  TEXT,                       -- JSON list of {file, cids}
    created  TEXT
);
CREATE INDEX IF NOT EXISTS ix_messages_conv ON messages(conv_id);
CREATE TABLE IF NOT EXISTS sanctioned_paths (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       TEXT UNIQUE NOT NULL,
    scope      TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1,
    local_only INTEGER NOT NULL DEFAULT 0,
    added      TEXT
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # -- conversations ------------------------------------------------------- #
    def new_conversation(self, title: str = None, model: str = None) -> int:
        now = _now()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO conversations(title, model, created, updated) "
                "VALUES (?,?,?,?)", (title, model, now, now))
            self.conn.commit()
            return cur.lastrowid

    def add_message(self, conv_id: int, role: str, content: str,
                    model: str = None, sources=None) -> int:
        now = _now()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages(conv_id, role, content, model, sources, "
                "created) VALUES (?,?,?,?,?,?)",
                (conv_id, role, content, model,
                 json.dumps(sources) if sources is not None else None, now))
            self.conn.execute("UPDATE conversations SET updated=? WHERE id=?",
                              (now, conv_id))
            self.conn.commit()
            return cur.lastrowid

    def get_conversation(self, conv_id: int):
        with self._lock:
            c = self.conn.execute("SELECT * FROM conversations WHERE id=?",
                                  (conv_id,)).fetchone()
            if not c:
                return None
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE conv_id=? ORDER BY id", (conv_id,)
            ).fetchall()
        conv = dict(c)
        conv["messages"] = [self._msg(r) for r in rows]
        return conv

    @staticmethod
    def _msg(r):
        m = dict(r)
        m["sources"] = json.loads(m["sources"]) if m["sources"] else None
        return m

    def list_conversations(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, title, model, updated FROM conversations "
                "ORDER BY updated DESC").fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conv_id: int, title: str):
        with self._lock:
            self.conn.execute("UPDATE conversations SET title=?, updated=? "
                              "WHERE id=?", (title, _now(), conv_id))
            self.conn.commit()

    def delete_conversation(self, conv_id: int):
        with self._lock:
            self.conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
            self.conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            self.conn.commit()

    # -- sanctioned paths ---------------------------------------------------- #
    def add_path(self, path: str, scope: str, local_only: bool = False) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR REPLACE INTO sanctioned_paths(path, scope, enabled, "
                "local_only, added) VALUES (?,?,1,?,?)",
                (path, scope, 1 if local_only else 0, _now()))
            self.conn.commit()
            return cur.lastrowid

    def list_paths(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, path, scope, enabled, local_only FROM "
                "sanctioned_paths ORDER BY id").fetchall()
        return [{**dict(r), "enabled": bool(r["enabled"]),
                 "local_only": bool(r["local_only"])} for r in rows]

    def set_path_enabled(self, path_id: int, enabled: bool):
        with self._lock:
            self.conn.execute("UPDATE sanctioned_paths SET enabled=? WHERE id=?",
                              (1 if enabled else 0, path_id))
            self.conn.commit()

    def remove_path(self, path_id: int):
        with self._lock:
            self.conn.execute("DELETE FROM sanctioned_paths WHERE id=?", (path_id,))
            self.conn.commit()

    def paths_for(self, cloud: bool):
        """Enabled sanctioned paths to index for this request. When the selected
        backend is a cloud model, local-only paths are withheld -- the privacy
        guard that makes mixing local + cloud safe."""
        return [p for p in self.list_paths()
                if p["enabled"] and not (cloud and p["local_only"])]

    def close(self):
        with self._lock:
            self.conn.close()
