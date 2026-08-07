# groundwire desktop app — developer handoff

The **app** turns groundwire from "a proxy behind Ollama's UI" into a native
chat client that **owns every turn**: it retrieves over your sanctioned folders
*before* the model answers and post-processes *after*, while talking to either a
local Ollama model or a cloud model. One Python process, the OS's own webview.

Pull this branch (`feat/app-core`) and read this file first, then
[`PACKAGING.md`](PACKAGING.md) when you're ready to build installers.

---

## Run from source (macOS)

```bash
git checkout feat/app-core && git pull
pip install -e ".[desktop]"        # pywebview (WKWebView), keyring, pypdf
python -m groundwire.desktop       # native window
# browser-only alternative (no webview): python -m groundwire.webapp  -> :8770
```

- **Ollama**: auto-discovered at `127.0.0.1:11434`. Point elsewhere with
  `GROUNDWIRE_OLLAMA_HOST=host:port`. Each installed chat model becomes a
  selectable backend (embedding models are filtered out).
- **Cloud models**: add entries to `~/.groundwire/config.json` (see below); API
  keys are read from the **environment only**, never stored in the config.
- **State**: `~/.groundwire/groundwire.db` (SQLite) holds conversations,
  messages with provenance, and the sanctioned-path allowlist. Add your corpus
  folders live via the **Memory sources** panel — nothing is indexed until you do.

### Tests
```bash
python -m pytest tests/ -q          # 121 tests
python -m unittest discover tests   # 73 tests, stdlib-only, network-free
```
Everything is deterministic and offline (a `FakeBackend` stands in for the model).

---

## Architecture (the before/after pipeline)

```
user turn
   │  BEFORE  retrieve over SANCTIONED paths + route the request
   ▼          (normal / quote / read / whole-book map-reduce)
[ Session._plan ]  builds EXACTLY what gets injected — no model call yet
   │  DISPATCH  stream through the selected backend (local Ollama or cloud)
   ▼
[ Session._execute ]
   │  AFTER   post-fill [[SPAN:…]] handles with verbatim bytes,
   ▼          append the audit footer, persist the turn with provenance
answer
```

Two guarantees:
- **BEFORE** (retrieval) is probabilistic → made auditable by `Session.inspect()`,
  which returns the exact `{mode, context, sources}` that *would* be injected,
  **without calling the model**. This powers the Debug panel: *you look, you
  don't trust*.
- **AFTER** (span post-fill) is **byte-exact by construction** — the answer's
  quoted passage is a slice of the stored file, not model output.

### Cloud safety
Two indexes are cached: **FULL** and **CLOUD-SAFE** (folders marked `local_only`
withheld). A turn picks by backend type, so selecting a cloud model never ships a
local-only folder, and nothing is re-indexed per request.

---

## File map

```
groundwire/
  desktop.py       native window (pywebview): free port → serve() in a thread →
                   webview.start(icon=…). Sets a Windows AppUserModelID.
  webapp.py        stdlib HTTP server: static UI + JSON/SSE API
                   (/api/models, /api/sources, /api/conversations, /api/chat [SSE],
                    /api/inspect). This is the seam the webview and any browser use.
  webui/index.html the tina4-js single-page UI (sidebar history, model dropdown,
                   chat, composer, Debug panel, Memory sources). Includes an
                   offline markdown renderer (renderRich) — no external deps/CDN.
  app.py           Session: the before/after core. _plan() routes + builds the
                   injected context; _execute() dispatches + post-fills; inspect().
  store.py         SQLite: conversations, messages(+sources JSON), sanctioned_paths.
                   paths_for(cloud=True) drops local_only folders.
  config.py        ~/.groundwire paths; build_backends() (discover Ollama + cloud);
                   make_session().
  backends.py      Backend.chat(messages, stream) → text deltas.
                   OllamaBackend (/api/chat NDJSON) + OpenAICompatBackend
                   (/v1 SSE: OpenAI, Gemini, OpenRouter, Groq, Ollama Cloud, vLLM).
  spans.py         SpanRegistry: addressable verbatim passages + resolution
                   (see "Retrieval & quote routing" below).
  pipeline.py      Groundwire retrieval engine (ingest/chunk/retrieve). LIBRARY —
  memory_systems.py  the retrieval layer; see the repo-root CLAUDE.md for its
  rerank.py / …       invariants (fold() symmetry, source-over-tests, etc.).
docs/APP.md        this file            docs/PACKAGING.md  build + signing
```

`server.py` and `mcp_server.py` are **out of scope** (de-scoped proxy/agentic
paths) — don't touch them.

---

## Retrieval & quote routing (how a turn is decided)

`Session._plan()` classifies each turn; `SpanRegistry.resolve()` does the
addressing. The order that matters:

1. **READ** — a question *about* a located passage ("explain psalm 23", "who dies
   in chapter 5") → inject the passage, let the model answer.
2. **Whole-file map-reduce** — "summarize the whole book" over a large file.
3. **QUOTE (verbatim)** — an explicit "quote …", **or a bare structural/positional
   reference** ("psalm 23", "section 8") → offer a `[[SPAN:handle]]`, post-fill the
   exact bytes. The model never sees the passage, only a handle.
4. **NORMAL** — retrieve top-k, inject as grounding.

### Addressing (all in `spans.py`)
- **Colon-numbered** divisions (`resolve_numbered`): named books whose bodies
  number `chapter:verse` and reset at `1:1` — scripture, legal codes, specs.
  `psalm 23`, `psalm23`, `psalm-23`, `john 3:16` all resolve (case-insensitive,
  separator optional). Guarded by a **monotonicity** check so sports scores /
  ratios / clock-time logs (`1:1 2:1 3:0…`) are **not** mistaken for a book.
- **Label-aware structural** (`resolve_structural`): division words
  (chapter/section/article/amendment/act/scene/sonnet/canto/part/…) each with an
  **outline level**. A unit's body runs to the next same-or-higher marker, so:
  - `section 8` = the **labelled** Section 8, not the 8th marker;
  - `article 2 section 8` / `section 8 of article 2` = Section 8 **inside**
    Article II (nesting; innermost ref is the target, outer refs scope it);
  - `act 2 scene 3` nests; `amendment xiv` parses roman → 14; inline-body legal
    headers ("Section 8. The Congress…") quote in full, bare headers
    ("CHAPTER III") exclude the label line.
- **Accuracy guard**: a positional reference we *can't* resolve is **never**
  faked from a lexical hit — the turn falls through to normal grounding, whose
  system header tells the model to admit it couldn't find the passage.

Adversarial coverage lives in `tests/test_spans.py`; a throwaway multi-corpus
fuzz harness pattern is described in the git log for
"Don't mistake non-monotonic N:M data…".

---

## Adding a cloud backend

`~/.groundwire/config.json`:
```json
{
  "ollama_host": "127.0.0.1:11434",
  "backends": [
    { "name": "Gemini Flash", "type": "gemini", "model": "gemini-2.0-flash" }
  ]
}
```
Set the key in the environment (`GEMINI_API_KEY`, `OPENAI_API_KEY`, …) — see
`PROVIDERS` in `backends.py` for each type's env var. Keys are never written to
disk.

---

## Status & next steps

**Working now:** local + cloud backends, streaming chat, conversation history,
sanctioned paths (+ local-only cloud guard), the Debug/inspect panel, markdown
rendering, and the full quote/positional/label-aware addressing above.
Cross-platform window + app icon.

**Next:**
1. **Package** signed `.dmg` (macOS) / `.msi` (Windows) — Briefcase + simplesign.
   Full steps in [`PACKAGING.md`](PACKAGING.md). On macOS the Dock/bundle icon
   comes from the packaged app (uses `assets/groundwire.png`); the `.ico` path is
   Windows-only.
2. **First-run onboarding** — pick model, add first sanctioned folder, optional
   key entry (via `keyring`, not the config file).
3. **Auto-update** channel.
4. **Gemini live validation** once a key is provided via env.

**Constraints:** stdlib-first core (only the `[desktop]` extra adds
pywebview/keyring/pypdf); keep every retrieval change validated on the weak
reader (Qwen-7B) per the repo-root `CLAUDE.md`.
