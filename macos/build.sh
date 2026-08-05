#!/bin/bash
# Build the native groundwire menu-bar app (Groundwire.app) with swiftc -- no Xcode
# project needed. Produces a LSUIElement agent app (menu bar only, no dock icon)
# that supervises the groundwire proxy. Ad-hoc signed so it launches locally.
#
#   ./macos/build.sh          # -> macos/Groundwire.app
#   open macos/Groundwire.app     # icon appears in the menu bar
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
APP="$DIR/Groundwire.app"
BIN="groundwire"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>groundwire</string>
    <key>CFBundleDisplayName</key><string>groundwire</string>
    <key>CFBundleIdentifier</key><string>com.groundwire.tray</string>
    <key>CFBundleExecutable</key><string>$BIN</string>
    <key>CFBundleVersion</key><string>0.1</string>
    <key>CFBundleShortVersionString</key><string>0.1</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

# Substitute the repo path into the source, then compile in Swift 5 mode (avoids
# Swift 6 strict-concurrency errors on the AppKit/URLSession callbacks).
SRC="$(mktemp -t groundwire).swift"
sed "s#__GROUNDWIRE_REPO__#$REPO#g" "$DIR/Groundwire/main.swift" > "$SRC"

xcrun swiftc -swift-version 5 -O "$SRC" \
    -framework Cocoa \
    -o "$APP/Contents/MacOS/$BIN"

rm -f "$SRC"
codesign --force --sign - "$APP" >/dev/null 2>&1 || true

echo "built: $APP"
echo "run:   open \"$APP\"   (icon appears in the menu bar)"
