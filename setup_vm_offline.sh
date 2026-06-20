#!/usr/bin/env bash
# ============================================================================
# DigitNote — Offline/China-Friendly Android Build
# ============================================================================
# This script can use a LOCAL copy of python-for-android, avoiding the
# need to clone from GitHub (which is very slow in China).
#
# Usage:
#   1. On Windows, download:
#      https://github.com/kivy/python-for-android/archive/refs/heads/develop.zip
#   2. Place the zip in your shared folder
#   3. In the VM:
#      unzip /mnt/hgfs/win_code/python-for-android-develop.zip -d ~/
#      cd ~/digitnote
#      bash setup_vm_offline.sh
#
#   Or let the script try to download from gitee mirror:
#      cd ~/digitnote
#      bash setup_vm_offline.sh online
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  DigitNote — Offline Android Build${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# ---- Step 1: System packages ----
echo -e "${YELLOW}[1/5] System packages...${NC}"
sudo apt update -qq
sudo apt install -y -qq python3-pip python3-dev openjdk-17-jdk \
    git autoconf libtool zip unzip cmake libffi-dev libssl-dev \
    zlib1g-dev libltdl-dev ccache 2>&1 | tail -3
echo "  Done"

# ---- Step 2: Install Buildozer (use Aliyun pip mirror) ----
echo -e "${YELLOW}[2/5] Install Buildozer...${NC}"
pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    buildozer cython 2>&1 | tail -3

export PATH="$HOME/.local/bin:$PATH"
echo "  Done"

# ---- Step 3: Setup python-for-android ----
echo -e "${YELLOW}[3/5] Setup python-for-android...${NC}"

P4A_DIR="$HOME/python-for-android"

if [ -d "$P4A_DIR" ]; then
    echo "  Using existing copy: $P4A_DIR"
elif [ -d "/mnt/hgfs/win_code/python-for-android-develop" ]; then
    echo "  Found in shared folder, copying..."
    cp -r /mnt/hgfs/win_code/python-for-android-develop "$P4A_DIR"
elif [ -d "/mnt/hgfs/win_code/pytorch/python-for-android-develop" ]; then
    echo "  Found in pytorch shared folder, copying..."
    cp -r /mnt/hgfs/win_code/pytorch/python-for-android-develop "$P4A_DIR"
elif [ "${1:-}" = "online" ]; then
    echo "  Trying gitee mirror..."
    git clone --depth 1 https://gitee.com/kivy-org/python-for-android.git "$P4A_DIR" 2>/dev/null || \
    git clone --depth 1 https://gitcode.com/mirrors/kivy/python-for-android.git "$P4A_DIR" 2>/dev/null || \
    git clone --depth 1 https://github.com/kivy/python-for-android.git "$P4A_DIR" 2>/dev/null || {
        echo -e "${RED}  Could not download python-for-android.${NC}"
        echo "  On Windows, download this and put in shared folder:"
        echo "  https://github.com/kivy/python-for-android/archive/refs/heads/develop.zip"
        echo "  Then re-run: bash setup_vm_offline.sh"
        exit 1
    }
else
    echo -e "${RED}  python-for-android not found!${NC}"
    echo ""
    echo "  Option 1 — download on Windows and share:"
    echo "    1. Visit: https://github.com/kivy/python-for-android/archive/refs/heads/develop.zip"
    echo "    2. Save to your shared folder (same place as digitnote/)"
    echo "    3. In VM: unzip /mnt/hgfs/win_code/python-for-android-develop.zip -d ~/"
    echo "    4. Re-run: bash setup_vm_offline.sh"
    echo ""
    echo "  Option 2 — let the script try online:"
    echo "    bash setup_vm_offline.sh online"
    exit 1
fi

echo "  Done"

# ---- Step 4: Configure Chinese mirrors in pip.conf ----
echo -e "${YELLOW}[4/5] Configure PyPI mirror...${NC}"
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF
echo "  Done"

# ---- Step 5: Build ----
echo -e "${YELLOW}[5/5] Build APK (with local p4a)...${NC}"
echo ""

buildozer android debug --p4a-local="$P4A_DIR"

# ---- Done ----
echo ""
APK=$(ls -t bin/*.apk 2>/dev/null | head -1)
if [ -n "$APK" ]; then
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  BUILD SUCCESSFUL!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo "  APK: $APK ($(du -h "$APK" | cut -f1))"
else
    echo -e "${RED}  APK not found. Check output above.${NC}"
fi
