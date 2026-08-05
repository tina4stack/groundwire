// groundwire -- native macOS menu-bar app (the Mac equivalent of the Windows tray).
//
// One status-bar icon that runs the whole drop-in context layer: it KILLS any
// running Ollama (desktop app + CLI serve) so groundwire owns the runtime, starts a
// single Ollama on an upstream port, puts the groundwire proxy on Ollama's normal
// port (11434) so every client gets grounded, watches your drop folder, and
// shows how much CONTEXT is held (polled from the proxy's /groundwire/ping).
//
// Menu mirrors groundwire/tray.py on Windows: status · context · GPU line · Open
// Ollama chat · Injection (toggle) · Open drop folder · Reindex now · Quit.
//
// Built with swiftc (see ../build.sh) into a LSUIElement .app -- no dock icon.
import Cocoa
import Foundation

let REPO = "__GROUNDWIRE_REPO__"                               // filled in by build.sh
let WATCH = (NSHomeDirectory() as NSString).appendingPathComponent("groundwire/watch")
let LISTEN = 11434                                         // groundwire takes Ollama's port
let UPSTREAM = 11435                                       // the real ollama moves here

func firstExecutable(_ candidates: [String]) -> String {
    for c in candidates where FileManager.default.isExecutableFile(atPath: c) { return c }
    return candidates.last ?? ""
}

@discardableResult
func launch(_ path: String, _ args: [String],
            env extra: [String: String] = [:], cwd: String? = nil,
            wait: Bool = false) -> Process? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    var e = ProcessInfo.processInfo.environment
    for (k, v) in extra { e[k] = v }
    p.environment = e
    if let cwd = cwd { p.currentDirectoryURL = URL(fileURLWithPath: cwd) }
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    do { try p.run(); if wait { p.waitUntilExit() }; return p } catch { return nil }
}

func humanBytes(_ n: Int) -> String {
    if n < 1024 { return "\(n) B" }
    if n < 1024 * 1024 { return String(format: "%.1f KB", Double(n) / 1024) }
    return String(format: "%.1f MB", Double(n) / (1024 * 1024))
}

func humanTokens(_ n: Int) -> String {
    if n < 1000 { return "\(n)" }
    if n < 1_000_000 { return String(format: "%.0fK", Double(n) / 1000) }
    return String(format: "%.2fM", Double(n) / 1_000_000)
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var proxy: Process?
    var ollama: Process?
    var timer: Timer?
    let ollamaPath = firstExecutable(["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"])
    let pythonPath = firstExecutable(["/opt/homebrew/bin/python3",
                                      "/usr/local/bin/python3", "/usr/bin/python3"])

    func applicationDidFinishLaunching(_ notification: Notification) {
        try? FileManager.default.createDirectory(atPath: WATCH,
                                                 withIntermediateDirectories: true)
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let btn = statusItem.button {
            if let img = NSImage(systemSymbolName: "sparkle.magnifyingglass",
                                 accessibilityDescription: "groundwire") {
                img.isTemplate = true
                btn.image = img
            } else {
                btn.title = "◈"
            }
        }
        setMenu(status: "groundwire ▸ starting…")
        startStack()
        timer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            self?.refresh()
        }
    }

    /// Kill any running Ollama so groundwire owns exactly one. On Mac the desktop app
    /// auto-starts its own server on 11434 and collides with our port-takeover
    /// (the source of the 'connection refused' saga). We quit it + any CLI serve,
    /// then start a single upstream ourselves.
    func killExistingOllama() {
        // The macOS Ollama desktop app (Electron; process name "Ollama") auto-
        // resurrects via a launchd background item and monopolizes the single
        // Ollama runtime, killing groundwire's own serve. Boot out the resurrector,
        // then kill the app AND the CLI serve by EXACT process name (-x is
        // name-exact, so it never hits an unrelated process that merely mentions
        // "ollama" in its arguments -- and note the app is "Ollama", the CLI is
        // "ollama"). bootout is per-login-session; we redo it on every launch.
        launch("/bin/launchctl",
               ["bootout", "gui/\(getuid())/com.ollama.ollama"], wait: true)
        launch("/usr/bin/pkill", ["-x", "Ollama"], wait: true)   // desktop app
        launch("/usr/bin/pkill", ["-x", "ollama"], wait: true)   // CLI serve
    }

    /// Whole stack startup runs off the main thread so the UI never blocks.
    func startStack() {
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            self.killExistingOllama()
            Thread.sleep(forTimeInterval: 1.0)              // let ports/locks release
            self.ollama = launch(self.ollamaPath, ["serve"],
                                 env: ["OLLAMA_HOST": "127.0.0.1:\(UPSTREAM)",
                                       "OLLAMA_KEEP_ALIVE": "-1"])  // model stays resident
            Thread.sleep(forTimeInterval: 1.5)              // let ollama bind
            if self.proxy == nil {
                self.proxy = launch(self.pythonPath,
                    ["-m", "groundwire.proxy",
                     "--listen", "\(LISTEN)",
                     "--upstream", "127.0.0.1:\(UPSTREAM)",
                     "--watch", WATCH],
                    cwd: REPO)
            }
        }
    }

    /// Poll the proxy's health marker and refresh the live indicator + menu.
    func refresh() {
        guard let url = URL(string: "http://127.0.0.1:\(LISTEN)/groundwire/ping") else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.0
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            var statusLine = "groundwire ▸ starting…"
            var contextLine = ""
            var gpuLine = ""
            var enabled = true
            if let d = data,
               let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                let chunks = obj["chunks"] as? Int ?? 0
                let tokens = obj["tokens"] as? Int ?? 0
                let bytes = obj["bytes"] as? Int ?? 0
                let injected = obj["injected"] as? Int ?? 0
                let requests = obj["requests"] as? Int ?? 0
                enabled = obj["enabled"] as? Bool ?? true
                let up = (obj["upstream_ok"] as? Bool ?? false) ? "up" : "down"
                statusLine = "groundwire \(enabled ? "on" : "PAUSED") · "
                           + "\(injected)/\(requests) injected · upstream \(up)"
                contextLine = "Context held: \(chunks) chunks · \(humanBytes(bytes)) "
                            + "· ~\(humanTokens(tokens)) tokens"
                if let g = obj["gpu"] as? String, !g.isEmpty {
                    gpuLine = "Full attention would need \(g) — groundwire runs it off-GPU"
                }
            }
            DispatchQueue.main.async {
                self?.setMenu(status: statusLine, context: contextLine,
                              gpu: gpuLine, enabled: enabled)
            }
        }.resume()
    }

    func setMenu(status: String, context: String = "", gpu: String = "",
                 enabled: Bool = true) {
        let menu = NSMenu()
        menu.autoenablesItems = false
        addDisabled(menu, status)
        if !context.isEmpty { addDisabled(menu, context) }
        if !gpu.isEmpty { addDisabled(menu, gpu) }
        menu.addItem(.separator())
        add(menu, "Open Ollama chat", #selector(openOllama))
        let inj = add(menu, "Injection", #selector(toggleInjection))
        inj.state = enabled ? .on : .off
        add(menu, "Open drop folder", #selector(openFolder))
        add(menu, "Reindex now", #selector(reindexNow))
        menu.addItem(.separator())
        add(menu, "Quit groundwire", #selector(quit))
        statusItem.menu = menu
    }

    private func addDisabled(_ menu: NSMenu, _ title: String) {
        let it = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        it.isEnabled = false
        menu.addItem(it)
    }

    @discardableResult
    private func add(_ menu: NSMenu, _ title: String, _ sel: Selector) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: sel, keyEquivalent: "")
        item.target = self
        item.isEnabled = true
        menu.addItem(item)
        return item
    }

    /// POST a control command to the proxy, then refresh the menu.
    private func post(_ path: String) {
        guard let url = URL(string: "http://127.0.0.1:\(LISTEN)\(path)") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 10.0
        URLSession.shared.dataTask(with: req) { [weak self] _, _, _ in
            DispatchQueue.main.async { self?.refresh() }
        }.resume()
    }

    @objc func toggleInjection() { post("/groundwire/toggle") }
    @objc func reindexNow() { post("/groundwire/reindex") }

    @objc func openFolder() { NSWorkspace.shared.open(URL(fileURLWithPath: WATCH)) }

    @objc func openOllama() {
        // groundwire already holds 11434 (Ollama's default), so the app's client
        // targets us by default. Open it (best-effort).
        let app = URL(fileURLWithPath: "/Applications/Ollama.app")
        if FileManager.default.fileExists(atPath: app.path) {
            NSWorkspace.shared.open(app)
        }
    }

    @objc func quit() { teardown(); NSApp.terminate(nil) }

    func applicationWillTerminate(_ notification: Notification) { teardown() }

    private func teardown() {
        proxy?.terminate()
        ollama?.terminate()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)     // menu-bar agent -- no dock icon
app.run()
