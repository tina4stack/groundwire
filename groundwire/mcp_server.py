#!/usr/bin/env python3
"""
groundwire as a STANDALONE MCP server, built on the Tina4 stack.

groundwire is its own MCP server (not part of tina4-coder) but rides Tina4's MCP
implementation for the protocol/transport: `McpServer` + `@mcp_tool` +
`register_routes`, served by Tina4's HTTP server, which speaks both the
Streamable-HTTP (Claude Code `--transport http`) and legacy HTTP+SSE
(Claude Desktop) transports for free.

The corpus lives in groundwire's off-model index (RAM/disk). The client's context
window only ever holds the chunks it chose to pull — the window becomes a
working buffer over unbounded external memory. groundwire's internal steps are
exposed as tools the model drives, so it runs the agentic loop that our
benchmarks showed matters:

    search -> read_source (more) -> search again (multi-hop 14%->100%)
           -> write code -> verify_code -> fix (the 63%->74% self-correction)

Run (Tina4 serves it):
    groundwire-mcp --repo ~/proj/tina4-python --scope tina4-python \
               --package tina4_python --port 7180

Register with Claude Code / Desktop (Streamable HTTP):
    { "groundwire": { "transport": "http", "url": "http://localhost:7180/groundwire" } }
"""

from __future__ import annotations
import argparse
import os

# Tina4 refuses to boot outside its own CLI unless this is set. We ARE running
# it directly (as a standalone MCP server), so opt in before importing.
os.environ.setdefault("TINA4_OVERRIDE_CLIENT", "true")

import tina4_python                                   # noqa: F401,E402 (initializes)
from tina4_python.core import router as tina4_router  # module = decorator-style
from tina4_python.core.server import run
from tina4_python.mcp import McpServer, mcp_tool

from .pipeline import Groundwire
from .server import build_symbol_map, symbol_map_text, check_imports, SYMBOLS

# one MCP server mounted at /groundwire; Tina4 wires /groundwire, /groundwire/message,
# /groundwire/sse when we register_routes below
mcp = McpServer("/groundwire", name="groundwire memory", version="0.1.0")

MEM: Groundwire | None = None
SCOPE: str | None = None
PACKAGE: str | None = None
CONFIG: dict = {}


# --------------------------------------------------------------------------- #
# Tools. Type hints become the JSON Schema Tina4 advertises to the client, so
# the parameter signatures below are the tool contract the model sees.
# --------------------------------------------------------------------------- #
@mcp_tool("search", description="Search the indexed corpus for passages "
          "relevant to a query. Returns ranked results with source paths so "
          "you can read more or refine and search again. The corpus never "
          "enters your context; only what this returns does.", server=mcp)
def search(query: str, k: int = 6, scope: str = "") -> dict:
    hits = MEM.retrieve(query, k=k, scope=scope or SCOPE)
    return {
        "query": query,
        "probes": getattr(MEM.memory, "last_queries", [query])[1:],
        "results": [
            {"source": MEM.source_of(cid) or str(cid),
             "score": round(score, 4), "text": text}
            for cid, text, score in hits
        ],
    }


@mcp_tool("read_source", description="Read a source file (or the region around "
          "a substring) from the index. Use after search when a result is "
          "truncated or you need the whole definition.", server=mcp)
def read_source(path: str, around: str = "", window: int = 120) -> dict:
    text = _read_source_path(path)
    if text is None:
        return {"path": path, "error": "not found"}
    if around:
        lines = text.splitlines()
        idx = next((i for i, l in enumerate(lines) if around in l), None)
        if idx is not None:
            lo, hi = max(0, idx - window), idx + window
            text = "\n".join(lines[lo:hi])
    return {"path": path, "text": text}


@mcp_tool("list_symbols", description="List the real public names in a package "
          "module (classes, functions, factories). Pass a module like "
          "'tina4_python.queue', or '' for the whole map. Check this before "
          "importing so you reference names that exist.", server=mcp)
def list_symbols(module: str = "") -> dict:
    if not SYMBOLS:
        return {"error": "no symbol map (started without --package)"}
    if module and module in SYMBOLS:
        return {"module": module,
                "names": sorted(n for n in SYMBOLS[module]
                                if not n.startswith("_"))}
    return {"map": symbol_map_text(PACKAGE, max_lines=200)}


@mcp_tool("verify_code", description="Check generated code's imports and "
          "attribute paths against the installed package. Returns [] if all "
          "resolve, or a list of problems naming each bad symbol and its real "
          "home. Call this on code you wrote BEFORE returning it.", server=mcp)
def verify_code(code: str) -> dict:
    if not PACKAGE or not SYMBOLS:
        return {"problems": [], "note": "no package verifier configured"}
    return {"problems": check_imports(code, PACKAGE)}


@mcp_tool("reindex", description="Rebuild the index from the configured "
          "sources and hot-swap it, so search reflects the current state after "
          "the code/docs change.", server=mcp)
def reindex() -> dict:
    global MEM
    MEM = _build()
    return {"status": "reindexed", "chunks": MEM._next, "modules": len(SYMBOLS)}


# --------------------------------------------------------------------------- #
def _read_source_path(path):
    for root in CONFIG["repos"]:
        label = os.path.basename(root)
        if path.startswith(label + "/"):
            full = os.path.join(root, path[len(label) + 1:])
            if os.path.exists(full):
                return open(full, encoding="utf-8", errors="ignore").read()
    return None


def _build() -> Groundwire:
    # expand=0: the CLIENT (Claude) rewrites/re-queries (agentic multi-hop), so
    # groundwire doesn't need its own LLM rewriter here. But the dense RERANKER is a
    # retrieval-internal step the client can't do -- reordering groundwire's own
    # candidate pool by semantic similarity -- so enable it when an embeddings
    # endpoint is configured (EMBED_URL). Without it, search under-ranks
    # definitions (audit: core/router.py fell out of top-k for a route query).
    rerank = "dense" if os.environ.get("EMBED_URL") else None
    mem = Groundwire(memory=CONFIG["backend"], k=CONFIG["k"], expand=0,
                 rerank=rerank, encoder="openai:nomic-embed-text" if rerank else None)
    for r in CONFIG["repos"]:
        mem.ingest_repo(r)
    if SCOPE and PACKAGE:
        root = next((r for r in CONFIG["repos"]
                     if os.path.basename(r) == SCOPE), None)
        if root:
            SYMBOLS.clear()
            build_symbol_map(root, PACKAGE)
    return mem


def main():
    global MEM, SCOPE, PACKAGE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", default=[])
    ap.add_argument("--scope", default=None)
    ap.add_argument("--package", default=None)
    ap.add_argument("--backend", default="bm25")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--port", type=int, default=7180)
    args = ap.parse_args()

    SCOPE, PACKAGE = args.scope, args.package
    CONFIG.update(repos=[os.path.abspath(r) for r in args.repo],
                  backend=args.backend, k=args.k)
    MEM = _build()

    mcp.register_routes(tina4_router)    # module's decorator get/post/noauth
    run(host="0.0.0.0", port=args.port)  # Tina4's HTTP server


if __name__ == "__main__":
    main()
