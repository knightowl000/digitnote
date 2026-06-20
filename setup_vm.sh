#!/usr/bin/env bash
# ============================================================================
# DigitNote — Ubuntu VM Build Setup (one-shot)
# ============================================================================
# Usage:
#   1. Copy digitnote/ folder into your Ubuntu VM
#   2. cd digitnote
#   3. bash setup_vm.sh
#   4. Wait ~20-40 min (first build downloads Android SDK/NDK)
#   5. APK at: bin/digitnote-*-debug.apk
# ============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  DigitNote — Ubuntu VM Build Setup${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# ---- Step 1: System packages ----
echo -e "${YELLOW}[1/5] Installing system packages...${NC}"
sudo apt update -qq
sudo apt install -y -qq \
    python3-pip python3-dev \
    openjdk-17-jdk \
    git autoconf libtool zip unzip cmake \
    libffi-dev libssl-dev zlib1g-dev \
    libltdl-dev \
    ccache

echo "  Java: $(java -version 2>&1 | head -1)"
echo "  Python: $(python3 --version)"

# ---- Step 2: Python tools ----
echo -e "${YELLOW}[2/5] Installing Buildozer + Cython...${NC}"

# Detect virtualenv — --user is incompatible with venv
if python3 -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' 2>/dev/null; then
    PIP_USER=""
    echo "  (detected virtualenv, skipping --user)"
else
    PIP_USER="--user"
fi

pip install $PIP_USER --upgrade pip buildozer cython

# Ensure ~/.local/bin is on PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "  Buildozer: $(buildozer version 2>&1 || echo 'need relogin')"

# ---- Step 3: Accept Android licenses ----
echo -e "${YELLOW}[3/5] Accepting Android SDK licenses...${NC}"
if [ ! -d "$HOME/.android" ]; then
    mkdir -p "$HOME/.android"
fi
# Pre-create the Android SDK license acceptance file
mkdir -p "$HOME/.android/sdk/licenses"
echo -e "\nd56f5187479451eabf01fb78af6dfcb131a6481e" > "$HOME/.android/sdk/licenses/android-sdk-license"
echo "  SDK license pre-accepted"

# ---- Step 4: Verify project structure ----
echo -e "${YELLOW}[4/5] Verifying project...${NC}"
REQUIRED_FILES=(
    "main.py"
    "buildozer.spec"
    "model/digit_full_weights.npz"
    "model/inference.py"
    "model/inference_numpy.py"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo -e "  ${RED}[MISSING]${NC} $f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo -e "${RED}Missing required files. Make sure you copied the full digitnote/ directory.${NC}"
    echo "From Windows, run:  tar -czf digitnote.tar.gz digitnote/"
    echo "Then copy digitnote.tar.gz to the VM and extract:  tar -xzf digitnote.tar.gz"
    exit 1
fi

# ---- Step 5: Build ----
echo -e "${YELLOW}[5/5] Building APK (this will take a while)...${NC}"
echo ""
echo "  First build downloads:"
echo "    - Android SDK (~150 MB)"
echo "    - Android NDK 25b (~450 MB)"
echo "    - Python 3 for Android (~30 MB)"
echo "    - Compiling kivy, numpy, opencv, Pillow (~15 min)"
echo ""

buildozer android debug

# ---- Done ----
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Build complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
APK=$(ls -t bin/*.apk 2>/dev/null | head -1)
if [ -n "$APK" ]; then
    SIZE=$(du -h "$APK" | cut -f1)
    echo -e "  APK: ${GREEN}$APK${NC} ($SIZE)"
    echo ""
    echo "  Transfer to your phone:"
    echo "    adb install $APK"
    echo "  Or copy back to Windows (from VM):"
    echo "    python3 -m http.server 8888"
    echo "    Then open http://<vm-ip>:8888/bin/ in Windows browser"
else
    echo -e "  ${RED}APK not found. Check output above for errors.${NC}"
    echo "  Common fixes:"
    echo "    1. Relogin and retry:  buildozer android clean && buildozer android debug"
    echo "    2. Check disk space:  df -h"
    echo "    3. Check RAM:  free -h  (needs ~4GB free)"
fi
