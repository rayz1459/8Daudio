#!/bin/bash
# ─── 8D Spatial Audio — Setup & Launch ───────────────────────────────────────
# Run this script once to install dependencies, then double-click to launch.

echo "🎧 8D Spatial Audio Setup"
echo "─────────────────────────────────────"

# 1. Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install from https://python.org"
  exit 1
fi
echo "✅ Python 3 found"

# 2. Install dependencies
echo "📦 Installing audio libraries..."
pip3 install sounddevice numpy scipy --quiet
echo "✅ Dependencies ready"

# 3. Check for BlackHole
if system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole"; then
  echo "✅ BlackHole virtual audio driver detected"
else
  echo ""
  echo "⚠️  BlackHole not found — you need it to capture Spotify's audio."
  echo "   Download FREE from: https://existential.audio/blackhole/"
  echo "   Install BlackHole 2ch, then:"
  echo "   System Settings → Sound → Output → select BlackHole 2ch"
  echo ""
  echo "   (You can still launch the app now to explore)"
fi

echo ""
echo "─────────────────────────────────────"
echo "🚀 Launching 8D Spatial Audio..."
echo ""

# 4. Launch app
cd "$(dirname "$0")"
python3 spatial8d.py
