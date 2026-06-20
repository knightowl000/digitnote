# Buildozer configuration for DigitNote — Android APK
# ==================================================================
#
# IMPORTANT: Buildozer requires a Linux environment to build Android APKs.
# On Windows, use WSL:
#     wsl
#     sudo apt update && sudo apt install -y python3-pip openjdk-17-jdk git autoconf libtool
#     pip install buildozer
#     cd /mnt/c/Users/user/Desktop/code/pytorch/digitnote
#     buildozer android debug
#
# Inference backend: Pure NumPy CNN (no ONNX Runtime needed!)
#   - model/inference_numpy.py implements the full DigitCNN forward pass
#   - Uses only numpy (standard p4a recipe, available on Android)
#   - 99.71% accuracy, ~0.8 ms inference on ARM CPUs
#   - Weights: model/digit_full_weights.npz (6.3 MB, auto-loaded)
#
# Full Buildozer docs: https://buildozer.readthedocs.io/

[app]

# (str) Title of your application
title = DigitNote

# (str) Package name
package.name = digitnote

# (str) Package domain (needed for android/ios packaging)
package.domain = org.digitnote

# (str) Application version
version = 1.0.0

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,jpeg,bmp,gif,kv,onnx,npz,db

# (list) Source files patterns to include
source.include_patterns = model/*.onnx,model/*.npz,ui/*.py,utils/*.py

# (list) Source files to exclude
source.exclude_exts = spec,pth,pyc

# (list) List of directory to exclude
source.exclude_dirs = data,runs,__pycache__,model/__pycache__,ui/__pycache__,utils/__pycache__,.git

# (list) List of inclusions using pattern matching
# source.include_patterns = assets/*,images/*.png

# (bool) Set to False to skip clean before each build
# android.skip_clean = False

# (bool) Set to True to skip the update of the android sdk
# android.skip_update = False

# (str) The Android arch to build for (armeabi-v7a, arm64-v8a, x86_64)
android.arch = arm64-v8a

# (int) Minimum API level required (24 = Android 7.0 Nougat)
android.minapi = 24

# (int) Target API level
android.api = 34

# (str) NDK version to use
android.ndk = 25b

# (str) SDK version to use
# android.sdk = 34.0.0

# (list) Java compiler args
# android.javac_args = -source 1.8 -target 1.8

# (list) Gradle dependencies
# android.gradle_dependencies =

# (list) Java classes to add as activities to the manifest
# android.add_activities =

# (list) Android permissions
android.permissions = INTERNET

# (list) The Android SDK features
# android.features =

# (list) Android library projects to include
# android.add_libs =

# (list) services to export
# android.services =

# (list) Gradle libraries to include
# android.gradle_libs =

# (str) Android logcat filters to use
# android.logcat_filters = *:S python:D

# (str) Android entry point (default = main.py)
# android.entrypoint = main.py

# (list) Pattern to whitelist for the whole project
# android.whitelist =

# (str) Path to a custom whitelist file
# android.whitelist_src =

# (str) Path to a custom blacklist file
# android.blacklist_src =

# (list) List of Java .jar files to add to the libs
# android.add_jars =

# (list) List of Java files to add to the android project
# android.add_src =

# (list) List of Java files to add to the root of the apk
# android.add_java =

# (str) python-for-android branch to use
# p4a.branch = master

# (str) python-for-android fork to use
# p4a.fork = kivy

# (str) python-for-android specific commit to use
# p4a.commit = HEAD

# (str) python-for-android local directory
# p4a.local_recipes =

# (str) python-for-android git clone directory
# p4a.source_dir =

# (str) The directory in which python-for-android should look for your own build recipes
# p4a.recipe_dir =

# (str) The build directory
# android.build_dir = .buildozer

# (str) python-for-android additional parameters
# p4a.extra_args =

# ------------------------------------------------------------------
# Requirements — Python packages compiled for Android by p4a
# ------------------------------------------------------------------
# NumPy-only stack (no onnxruntime needed!):
#   - The app uses a pure NumPy CNN forward pass (model/inference_numpy.py)
#   - 99.71% accuracy, ~0.8ms inference on mobile ARM CPUs
#   - Only 6.3 MB for weights (digit_full_weights.npz)
#   - All packages below have standard p4a recipes
#
# Package notes:
#   - 'opencv' → p4a's built-in OpenCV recipe (provides cv2)
#   - 'numpy'  → p4a's built-in NumPy recipe
#   - 'sqlite3' → Part of Python stdlib, no extra requirement needed
requirements = python3,hostpython3,kivy==2.3.1,Pillow,numpy,opencv

# (str) Custom source folders for requirements -- comma-separated
# requirements.source.kivy = ../../kivy

# (str) Presplash of the application
# presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
# icon.filename = %(source.dir)s/data/icon.png

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (int) Presplash colors
# android.presplash_color = #FFFFFF

# (bool) Indicate if the application is fullscreen or not
fullscreen = 0


# ------------------------------------------------------------------
# Android specific: copy library files instead of making lib links
# ------------------------------------------------------------------
# android.copy_libs = 1

# (str) The Android arch to build for (armeabi-v7a, arm64-v8a, x86_64)
# android.arch = armeabi-v7a

# (str) Android logcat filters to use
# android.logcat_filters = *:S python:D

# (bool) Activate Java 8 support
# android.enable_androidx = True

# (str) XML file for custom backup rules
# android.backup_rules =

# (str) Android TV banner
# android.tv_banner =

# (str) Path to a custom AndroidManifest.xml template
# android.manifest.template =

# (list) Permissions
# android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE

# (list) The Android SDK features
# android.features =

# (int) Target SDK version
# android.api = 34

# (int) Minimum SDK version
# android.minapi = 24

# (int) NDK API version
# android.ndk_api = 24

# (bool) Use --private data storage (True) or --dir public (False)
# android.private_storage = True

# (str) NDK directory (if empty, auto-download)
# android.ndk_path = ~/android-ndk-r25b   # VM only: use --android-ndk flag

# (str) SDK directory (if empty, auto-download)
# android.sdk_path = ~/android-sdk   # VM only: use --android-sdk flag

# (str) ANT directory (if empty, auto-download)
# android.ant_path =

# (str) Extra Java compile-time arguments
# android.extra_java_args =

# (str) Gradle build tool
# android.gradle = gradle

# (str) Which Gradle version to use
# android.gradle_version = 8.7

# (bool) Enable AndroidX
android.accept_sdk_license = True
android.enable_androidx = True

# (bool) Enable Java 8
android.enable_java8 = True

# (str) Gradle build directory
# android.gradle_build_dir =

# (str) Which platform to build for (ios, android)
# build_platform = android


[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug)
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build artifact storage
# build_dir = ./.buildozer

# (str) Path to build output (i.e. .apk, .ipa) storage
# bin_dir = ./bin

# ------------------------------------------------------------------
# iOS specific (not used — DigitNote targets Android only)
# ------------------------------------------------------------------
# [app:ios]
# ...
