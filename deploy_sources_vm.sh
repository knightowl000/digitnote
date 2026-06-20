#!/usr/bin/env bash
# ============================================================================
# DigitNote — 离线部署源码包到 p4a 缓存
# ============================================================================
# p4a 对 git 类 recipe 会在缓存目录存 bare repo，构建时先 git fetch 更新。
# 在中国 GitHub 被墙，fetch 会卡死。本脚本用本地 zip 创建无远端的 bare repo，
# 让 git fetch 秒返回。
#
# 前置：在 Windows 浏览器下载以下 4 个 zip 放到共享文件夹：
#   numpy-2.3.0.zip    → https://github.com/numpy/numpy/archive/refs/tags/v2.3.0.zip
#   kivy-2.3.1.zip     → https://github.com/kivy/kivy/archive/refs/tags/2.3.1.zip
#   Pillow-11.3.0.zip  → https://github.com/python-pillow/Pillow/archive/refs/tags/11.3.0.zip
#   opencv-4.12.0.zip  → https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip
#
# 用法：在 VM 中 cd ~/digitnote && bash deploy_sources_vm.sh
# ============================================================================
set -euo pipefail

SHARED="/mnt/hgfs/win_code/pytorch"
# p4a 缓存 bare repo 的位置
P4A_CACHE="$HOME/.local/share/python-for-android/packages"
# p4a 构建时实际放置源码的位置
BUILD_PKGS="$HOME/digitnote/.buildozer/android/platform/build-arm64-v8a/packages"

echo "==> 创建目录..."
mkdir -p "$P4A_CACHE"

# --------------------------------------------------
# 函数：把 zip 变成 p4a 缓存目录中的 bare repo
# --------------------------------------------------
deploy_git_pkg() {
    local name="$1"        # 包名（与 recipe 目录名一致）
    local version="$2"     # 版本标签，如 v2.3.0
    local zipname="$3"     # zip 文件名
    local cache_dir="$P4A_CACHE/$name"

    echo ""
    echo "--- $name ($version) [git] ---"

    # 找 zip
    local zip_path=""
    for dir in "$SHARED" "$HOME/桌面" "$HOME/下载"; do
        if [ -f "$dir/$zipname" ]; then
            zip_path="$dir/$zipname"
            break
        fi
    done

    if [ -z "$zip_path" ]; then
        echo "  [SKIP] $zipname 未找到"
        return
    fi
    echo "  找到: $zip_path"

    # 如果缓存已有 bare repo 且无远端 → 跳过
    if [ -d "$cache_dir" ] && git -C "$cache_dir" rev-parse --is-bare-repository 2>/dev/null | grep -q true; then
        if ! git -C "$cache_dir" remote -v 2>/dev/null | grep -q .; then
            echo "  [OK] bare repo 已存在且无远端，跳过"
            return
        fi
    fi

    # 删除旧缓存和构建目录残留
    rm -rf "$cache_dir"
    rm -rf "$BUILD_PKGS/$name" 2>/dev/null || true

    # 解压到临时目录
    local tmp_dir="/tmp/p4a_${name}"
    rm -rf "$tmp_dir"
    mkdir -p "$tmp_dir"
    unzip -q "$zip_path" -d "$tmp_dir"

    # zip 里通常有一层目录，进入它
    local inner
    inner=$(ls -d "$tmp_dir"/*/ 2>/dev/null | head -1)
    if [ -z "$inner" ] || [ ! -d "$inner" ]; then
        inner="$tmp_dir"
    fi

    # 创建完整 git 仓库
    cd "$inner"
    git init -q
    git config user.email "otion@local"
    git config user.name "otion"
    git add -A
    git commit -q -m "local mirror of $name $version"
    # 打标签（p4a 靠 tag 检出正确版本）
    git tag "$version" 2>/dev/null || true
    # numpy 用 v2.3.0，kivy/Pillow 可能不带 v 前缀 — 都打上
    local v2="${version#v}"   # 去 v
    local v3="v${version#v}"  # 加 v
    [ "$v2" != "$version" ] && git tag "$v2" 2>/dev/null || true
    [ "$v3" != "$version" ] && git tag "$v3" 2>/dev/null || true

    # 克隆为 bare repo 放到 p4a 缓存
    local bare_tmp="/tmp/p4a_${name}.git"
    rm -rf "$bare_tmp"
    git clone --bare "$inner" "$bare_tmp" 2>/dev/null
    mv "$bare_tmp" "$cache_dir"

    # 清理
    rm -rf "$tmp_dir"
    echo "  [OK] bare repo 就绪 → $cache_dir"
}

# --------------------------------------------------
# 函数：opencv 用 https zip，p4a 直接下载 zip 到缓存
# --------------------------------------------------
deploy_zip_pkg() {
    local name="$1"
    local zipname="$2"
    local cache_dir="$P4A_CACHE/$name"

    echo ""
    echo "--- $name [zip] ---"

    local zip_path=""
    for dir in "$SHARED" "$HOME/桌面" "$HOME/下载"; do
        if [ -f "$dir/$zipname" ]; then
            zip_path="$dir/$zipname"
            break
        fi
    done

    if [ -z "$zip_path" ]; then
        echo "  [SKIP] $zipname 未找到"
        return
    fi
    echo "  找到: $zip_path"

    mkdir -p "$cache_dir"
    cp "$zip_path" "$cache_dir/$zipname"
    echo "  [OK] zip 就绪 → $cache_dir/$zipname"
}

# --------------------------------------------------
# 执行部署
# --------------------------------------------------
deploy_git_pkg "numpy"  "v2.3.0" "numpy-2.3.0.zip"
deploy_git_pkg "kivy"   "2.3.1"  "kivy-2.3.1.zip"
deploy_git_pkg "Pillow" "11.3.0" "Pillow-11.3.0.zip"
deploy_zip_pkg "opencv" "opencv-4.12.0.zip"

echo ""
echo "==> 部署完成 =="
echo "    运行: cd ~/digitnote && buildozer android debug"
