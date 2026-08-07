# Packaging groundwire as a downloadable macOS + Windows app

The app is one Python process: `groundwire.desktop` starts the server on a
private localhost port and opens a native webview window (WKWebView on macOS,
WebView2 on Windows) at the tina4-js UI. Packaging wraps that process into a
signed installer.

## Run from source (dev)

```bash
pip install -e ".[desktop]"     # pywebview + keyring + pypdf
python -m groundwire.desktop    # native window
# or, browser only:
python -m groundwire.webapp     # http://127.0.0.1:8770
```

Local Ollama is auto-discovered at `127.0.0.1:11434` (override
`GROUNDWIRE_OLLAMA_HOST`). Cloud backends come from `~/.groundwire/config.json`;
keys are read from the environment (users bring their own).

## Build the installers — Briefcase

[Briefcase](https://briefcase.readthedocs.io) turns the Python app into a signed
`.dmg` (macOS) and `.msi` (Windows), bundling the interpreter and deps.

```bash
pip install briefcase
briefcase create <platform>     # scaffold the native project
briefcase build   <platform>
briefcase package <platform>    # -> dist/groundwire-<ver>.dmg / .msi
```

Requirements to bundle: the `groundwire` package (incl. `webui/` assets — already
in `package-data`) and the `[desktop]` deps. Bundle no model — the app talks to
the user's local Ollama and/or their cloud keys.

## Signing

- **Windows** — sign the `.msi`/`.exe` with **simplesign** (Authenticode). An EV
  cert avoids SmartScreen warnings; OV works with an initial reputation ramp.
- **macOS** — codesign the `.app` with an **Apple Developer ID Application**
  cert (hardened runtime), then **notarize** (`notarytool`) and `staple` before
  wrapping the `.dmg`. This is mandatory for Gatekeeper; it's the biggest single
  fix for the "can't open on Mac" problem.

Signing runs *after* `briefcase package` (or via Briefcase's own signing hooks).

## CI — GitHub Actions release matrix

On a version tag, a matrix builds + signs both installers and attaches them to a
GitHub Release:

- `macos-latest`: briefcase build/package → codesign → notarize → staple → `.dmg`
- `windows-latest`: briefcase build/package → simplesign → `.msi`
- secrets: Apple Developer ID cert + App Store Connect API key; the simplesign
  credential — all as encrypted Actions secrets.

## First-run UX (in the app)

1. Detect local Ollama; if absent, link to install it or skip to cloud.
2. Add a cloud key (stored in the OS keychain via `keyring`) — optional.
3. Add a sanctioned folder (folder picker) → first index. The **local-only**
   flag hides a folder's contents from cloud models.

## Known follow-ups
- **Startup latency**: the server builds the index synchronously before the
  window loads (~seconds for large corpora). Serve immediately and index in a
  background thread, showing an "indexing…" state in the UI.
- **Auto-update**: check GitHub Releases (v1); Sparkle/Squirrel later.
- **Tauri path**: if bundle size / built-in auto-update matter, the same
  server+UI can be wrapped by a Tauri shell with the Python as a sidecar; only
  the shell/packaging layer changes.
