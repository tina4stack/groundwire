# groundwire-ollama.ps1 -- put groundwire in front of the Ollama desktop app.
#
# The Ollama app has no "server URL" setting, so we intercept by TAKING its port:
# the real ollama backend is moved to 11435 and groundwire listens on 11434. The app
# UI is only a *client* of 127.0.0.1:11434, so it then talks to groundwire -- which
# injects your dropped files and forwards to the real ollama. Zero app config.
#
#   Usage (PowerShell):
#     1. Quit the Ollama app FIRST (system tray -> Quit) so it releases 11434.
#     2. .\scripts\groundwire-ollama.ps1                 # starts backend + proxy
#     3. Re-open the Ollama app and chat -- it now answers from your files.
#
#   Reverse it: close this window (Ctrl+C), stop the 11435 ollama, reopen the
#   app normally. Nothing is permanently changed.

param(
    [string]$Watch        = "$env:USERPROFILE\groundwire\watch",
    [int]   $ProxyPort    = 11434,   # groundwire takes Ollama's normal port
    [int]   $UpstreamPort = 11435    # the real ollama moves here
)

function Test-Port([int]$Port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', $Port); $c.Close(); return $true
    } catch { return $false }
}

# 0. drop folder
New-Item -ItemType Directory -Force -Path $Watch | Out-Null
Write-Host "groundwire: watch folder -> $Watch"

# 1. is the app still holding 11434? If so we can't take it.
if (Test-Port $ProxyPort) {
    Write-Host "`n[!] Something is already on port $ProxyPort." -ForegroundColor Yellow
    Write-Host "    Quit the Ollama app (tray icon -> Quit) and run this again." -ForegroundColor Yellow
    exit 1
}

# 2. start the REAL ollama backend on the upstream port (if not already up)
if (-not (Test-Port $UpstreamPort)) {
    Write-Host "groundwire: starting real ollama on 127.0.0.1:$UpstreamPort ..."
    $env:OLLAMA_HOST = "127.0.0.1:$UpstreamPort"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Port $UpstreamPort); $i++) {
        Start-Sleep -Milliseconds 500
    }
}
if (-not (Test-Port $UpstreamPort)) {
    Write-Host "[!] ollama did not come up on $UpstreamPort" -ForegroundColor Red
    exit 1
}
Write-Host "groundwire: real ollama ready on 127.0.0.1:$UpstreamPort"

# 3. run the groundwire proxy on 11434 -> 11435 (blocks; Ctrl+C to stop)
Write-Host "groundwire: proxy on 127.0.0.1:$ProxyPort  ->  upstream 127.0.0.1:$UpstreamPort"
Write-Host "groundwire: now re-open the Ollama app and chat.`n"
python -m groundwire.proxy --listen $ProxyPort --upstream "127.0.0.1:$UpstreamPort" --watch $Watch
