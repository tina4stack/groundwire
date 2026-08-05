# install-desktop.ps1 -- put a "groundwire" icon on the desktop that launches the
# tray app (drop-in memory for Ollama) with our own icon. Run once.
#
#   .\scripts\install-desktop.ps1
#
# Double-clicking the icon runs `pythonw -m groundwire.tray`: no console window, it
# starts the real ollama on 11435, takes 11434, watches ~\groundwire\watch, and
# lives in the system tray. Remove it by deleting the desktop shortcut.

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$icon = Join-Path $repo "groundwire\assets\groundwire.ico"
$watch = Join-Path $env:USERPROFILE "groundwire\watch"

# pythonw.exe = python with no console window (proper app feel)
$pyw = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $pyw) { $pyw = (Get-Command python).Source }   # fallback

New-Item -ItemType Directory -Force -Path $watch | Out-Null

$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "groundwire.lnk"

$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut($lnkPath)
$lnk.TargetPath       = $pyw
$lnk.Arguments        = "-m groundwire.tray"
$lnk.WorkingDirectory = $repo
$lnk.IconLocation     = $icon
$lnk.Description       = "groundwire - drop-in memory for Ollama"
$lnk.Save()

Write-Host "Created desktop icon: $lnkPath"
Write-Host "  launches: $pyw -m groundwire.tray"
Write-Host "  drop files into: $watch"
Write-Host ""
Write-Host "Tip: quit the Ollama desktop app before first launch so groundwire can"
Write-Host "     take port 11434; then re-open the app -- it will use groundwire."
