# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DigitNote — Handwritten Digit Recognition
====================================================================
Produces a standalone Windows .exe (--onedir mode).

Usage:
    cd digitnote
    pyinstaller digitnote.spec
    # Output: dist/DigitNote/DigitNote.exe

Cross-platform notes:
    - macOS: replace kivy_deps paths with kivy_deps.sdl2/glew paths from pip
    - Linux: kivy uses system SDL2; remove the kivy_deps Tree() calls
"""

import sys
import os
from pathlib import Path

# ------------------------------
# Project paths
# ------------------------------
PROJECT_ROOT = os.path.abspath(SPECPATH)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

# ------------------------------
# Collect Kivy native DLLs from kivy_deps
# ------------------------------
kivy_binaries = []

# SDL2 (windowing, input, audio)
try:
    import kivy_deps.sdl2 as sdl2
    sdl2_bin = sdl2.dep_bins[0]
    if os.path.isdir(sdl2_bin):
        for f in os.listdir(sdl2_bin):
            src = os.path.join(sdl2_bin, f)
            if os.path.isfile(src):
                kivy_binaries.append((src, '.'))
except Exception:
    pass

# GLEW (OpenGL extension manager)
try:
    import kivy_deps.glew as glew
    glew_bin = glew.dep_bins[0]
    if os.path.isdir(glew_bin):
        for f in os.listdir(glew_bin):
            src = os.path.join(glew_bin, f)
            if os.path.isfile(src):
                kivy_binaries.append((src, '.'))
except Exception:
    pass

# ANGLE (OpenGL ES → Direct3D 11 translation layer)
try:
    import kivy_deps.angle as angle
    angle_bin = angle.dep_bins[0]
    if os.path.isdir(angle_bin):
        for f in os.listdir(angle_bin):
            src = os.path.join(angle_bin, f)
            if os.path.isfile(src):
                kivy_binaries.append((src, '.'))
except Exception:
    pass

# ------------------------------
# Collect onnxruntime native DLLs
# ------------------------------
ort_binaries = []
try:
    import onnxruntime
    ort_dir = os.path.dirname(onnxruntime.__file__)
    capi_dir = os.path.join(ort_dir, 'capi')
    if os.path.isdir(capi_dir):
        for f in os.listdir(capi_dir):
            if f.endswith('.dll') or f.endswith('.pyd'):
                ort_binaries.append((os.path.join(capi_dir, f), 'onnxruntime'))
    # Also check ort_dir top-level
    for f in os.listdir(ort_dir):
        fpath = os.path.join(ort_dir, f)
        if os.path.isfile(fpath) and (f.endswith('.dll') or f.endswith('.pyd')):
            ort_binaries.append((fpath, 'onnxruntime'))
except Exception:
    pass

# ------------------------------
# Collect OpenCV extra DLLs
# ------------------------------
cv2_binaries = []
try:
    import cv2
    cv2_dir = os.path.dirname(cv2.__file__)
    for f in os.listdir(cv2_dir):
        if f.endswith('.dll') and f != 'cv2.pyd':
            cv2_binaries.append((os.path.join(cv2_dir, f), 'cv2'))
except Exception:
    pass

# ------------------------------
# Model data files (include in bundle)
# ------------------------------
model_datas = [
    (os.path.join(MODEL_DIR, 'digit_full.onnx'), 'model'),
    (os.path.join(MODEL_DIR, 'digit_full_weights.npz'), 'model'),
]

# Also include MVP model if it exists
mvp_path = os.path.join(MODEL_DIR, 'digit_mvp.onnx')
if os.path.exists(mvp_path):
    model_datas.append((mvp_path, 'model'))

# ------------------------------
# Hidden imports
# ------------------------------
hiddenimports = [
    # Kivy core
    'kivy.core.window',
    'kivy.core.image',
    'kivy.core.text',
    'kivy.core.audio',
    'kivy.core.gl',
    'kivy.core.spelling',
    'kivy.core.clipboard',
    # Kivy built-in widgets
    'kivy.uix.label',
    'kivy.uix.button',
    'kivy.uix.textinput',
    'kivy.uix.togglebutton',
    'kivy.uix.boxlayout',
    'kivy.uix.gridlayout',
    'kivy.uix.scrollview',
    'kivy.uix.popup',
    'kivy.uix.filechooser',
    'kivy.uix.image',
    'kivy.uix.widget',
    # Kivy graphics
    'kivy.graphics',
    'kivy.graphics.instructions',
    'kivy.graphics.context_instructions',
    'kivy.graphics.vertex_instructions',
    # Kivy properties & events
    'kivy.properties',
    'kivy.event',
    'kivy.factory',
    'kivy.clock',
    'kivy.lang',
    'kivy.logger',
    # Kivy input providers
    'kivy.input.providers.mouse',
    'kivy.input.providers.tuio',
    'kivy.input.providers.wm_touch',
    'kivy.input.providers.wm_pen',
    # Kivy window providers
    'kivy.core.window.window_sdl2',
    # Kivy image providers
    'kivy.core.image.img_pygame',
    'kivy.core.image.img_pil',
    'kivy.core.image.img_sdl2',
    # Kivy text providers
    'kivy.core.text.text_sdl2',
    'kivy.core.text.text_pil',
    # Kivy dependencies
    'kivy_deps.sdl2',
    'kivy_deps.glew',
    'kivy_deps.angle',
    'kivymd',
    # Our app subpackages
    'ui',
    'ui.canvas_widget',
    'ui.main_screen',
    'ui.history_popup',
    'model',
    'model.inference',
    'utils',
    'utils.database',
    'utils.preprocessing',
    # ONNX Runtime
    'onnxruntime',
    'onnxruntime.capi',
    # OpenCV
    'cv2',
    # PIL
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    # Standard library (sometimes missed)
    'sqlite3',
    'io',
    'json',
    'datetime',
    'tempfile',
]

# ------------------------------
# PyInstaller Analysis
# ------------------------------
a = Analysis(
    [os.path.join(PROJECT_ROOT, 'main.py')],
    pathex=[PROJECT_ROOT],
    binaries=kivy_binaries + ort_binaries + cv2_binaries,
    datas=model_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary large packages
        'torch',
        'torchvision',
        'matplotlib',
        'tensorboard',
        'tkinter',
        'unittest',
        'test',
        'pydoc',
        'distutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ------------------------------
# Filter out common false-positives from Tree() collection
# ------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ------------------------------
# EXE — single-directory bundle
# ------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DigitNote',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # Set to True for smaller size (requires UPX installed)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,      # GUI mode (no console window on Windows)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,          # Add icon path here (e.g., 'digitnote.ico')
)
