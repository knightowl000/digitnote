"""
DigitNote — Cross-platform handwritten digit recognition
========================================================
Entry point: launches the Kivy app, loads the ONNX model, displays the main screen.

Usage:
    python main.py

Before first run, train the model:
    python model/train_mvp.py
"""

import os
import sys

# Ensure project root is on the Python path
# Handle PyInstaller frozen bundle (sys._MEIPASS is the temp extraction dir)
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = sys._MEIPASS

    # Fix kivy_deps crash: site.USER_BASE is None in frozen bundles,
    # causing `join(None, 'share', ...)` TypeError in kivy_deps.angle
    import site as _site
    if _site.USER_BASE is None:
        _site.USER_BASE = sys._MEIPASS

    # Add bundle directory to DLL search path so Windows can find
    # SDL2, GLEW, ANGLE, onnxruntime DLLs at runtime
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(sys._MEIPASS)
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import kivy
from kivy.app import App
from kivy.config import Config
from kivy.core.window import Window

# Kivy configuration
Config.set('graphics', 'resizable', True)
Config.set('graphics', 'width', '420')
Config.set('graphics', 'height', '700')
Config.set('graphics', 'minimum_width', '320')
Config.set('graphics', 'minimum_height', '520')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')


class DigitNoteApp(App):
    """DigitNote main application"""

    title = "DigitNote — Handwritten Digit Recognition"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recognizer = None
        self.db = None

    def build(self):
        """Build the application UI"""
        from kivy.core.window import Window
        Window.clearcolor = (0.98, 0.98, 0.98, 1)

        self._load_model()
        self._init_db()

        from ui.main_screen import MainScreen
        screen = MainScreen(recognizer=self.recognizer, db=self.db)

        return screen

    def _load_model(self):
        """Load the recognition model (ONNX preferred, NumPy fallback)."""
        from model.inference import DigitRecognizer

        # Priority: full ONNX → full NumPy weights → MVP ONNX
        candidates = [
            os.path.join(PROJECT_ROOT, "model", "digit_full.onnx"),
            os.path.join(PROJECT_ROOT, "model", "digit_full_weights.npz"),
            os.path.join(PROJECT_ROOT, "model", "digit_mvp.onnx"),
        ]

        model_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if model_path is None:
            print("[DigitNote] No model found. Attempting auto-train...")
            self._auto_train()
            # Check again after auto-train
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if model_path is None:
            print("[DigitNote] Model unavailable — recognition disabled.")
            self.recognizer = None
            return

        try:
            self.recognizer = DigitRecognizer(model_path)
            print(f"[DigitNote] Model loaded: {model_path}")
            print(f"[DigitNote] Backend: {self.recognizer.backend}")
        except Exception as e:
            print(f"[DigitNote] Model load failed: {e}")
            self.recognizer = None

    def _init_db(self):
        """Initialize the history database."""
        try:
            from utils.database import HistoryDB
            self.db = HistoryDB()
            print(f"[DigitNote] History database ready ({self.db.count()} records)")
        except Exception as e:
            print(f"[DigitNote] Database init failed: {e}")
            self.db = None

    def _auto_train(self):
        """Auto-train the MVP model"""
        try:
            import subprocess
            train_script = os.path.join(PROJECT_ROOT, "model", "train_mvp.py")
            result = subprocess.run(
                [sys.executable, train_script],
                cwd=PROJECT_ROOT,
                capture_output=False
            )
            if result.returncode != 0:
                print("[DigitNote] Auto-train failed. Please run manually:")
                print(f"  python {train_script}")
        except Exception as e:
            print(f"[DigitNote] Auto-train error: {e}")
            print(f"[DigitNote] Please run manually: python model/train_mvp.py")

    def on_stop(self):
        """Cleanup on app exit"""
        self.recognizer = None
        print("[DigitNote] App exited")


def main():
    """Application entry point"""
    app = DigitNoteApp()
    app.run()


if __name__ == "__main__":
    main()
