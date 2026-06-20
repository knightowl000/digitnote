"""
DigitNote main screen
Contains the handwriting canvas, action buttons, auto/manual toggle,
and recognition result display.
"""

import os

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
from kivy.clock import Clock

from ui.canvas_widget import HandwritingCanvas


AUTO_DELAY = 0.8  # seconds to wait before auto-recognize after stroke ends


class MainScreen(BoxLayout):
    """Main screen layout."""

    canvas_widget = ObjectProperty(None)
    result_text = StringProperty("Draw a digit, then tap Recognize")
    auto_mode = BooleanProperty(False)

    def __init__(self, recognizer=None, db=None, **kwargs):
        super().__init__(**kwargs)

        self.recognizer = recognizer
        self.db = db  # HistoryDB instance (optional)
        self._auto_timer = None  # Clock event for auto-recognition debounce
        self._pending = False    # True while a recognition is in progress

        self.orientation = 'vertical'
        self.padding = [12, 12, 12, 12]
        self.spacing = 8

        self._build_ui()

    def _build_ui(self):
        """Build UI components"""
        # === Row 0: Title + Mode toggle ===
        top_row = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height=42,
            spacing=10
        )

        title = Label(
            text="DigitNote",
            font_size='18sp',
            size_hint=(None, None),
            width=120,
            height=42,
            bold=True,
            halign='left',
            valign='middle',
            color=(0.2, 0.2, 0.2, 1)
        )
        title.bind(size=title.setter('text_size'))
        top_row.add_widget(title)

        # Spacer
        top_row.add_widget(Label(size_hint=(1, None), height=42))

        # Auto / Manual toggle
        self._mode_toggle = ToggleButton(
            text="Auto: OFF",
            font_size='14sp',
            size_hint=(None, None),
            width=110,
            height=38,
            background_normal='',
            background_color=(0.7, 0.7, 0.7, 1),
            color=(1, 1, 1, 1)
        )
        self._mode_toggle.bind(on_release=self._on_toggle_mode)
        top_row.add_widget(self._mode_toggle)

        self.add_widget(top_row)

        # === Row 1: Handwriting canvas ===
        self.canvas_widget = HandwritingCanvas(
            size_hint=(1, 1),
            stroke_end_callback=self._on_stroke_end,
        )
        self.add_widget(self.canvas_widget)

        # === Row 2: Result display ===
        self.result_label = Label(
            text=self.result_text,
            font_size='22sp',
            size_hint=(1, None),
            height=64,
            halign='center',
            valign='middle',
            color=(0.1, 0.1, 0.1, 1),
            markup=True
        )
        with self.result_label.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.95, 0.95, 0.95, 1)
            self._result_bg = Rectangle(
                pos=self.result_label.pos,
                size=self.result_label.size
            )
        self.result_label.bind(
            pos=lambda inst, val: setattr(self._result_bg, 'pos', val),
            size=lambda inst, val: setattr(self._result_bg, 'size', val)
        )
        self.add_widget(self.result_label)

        # === Row 3: Button bar ===
        buttons_outer = BoxLayout(
            orientation='vertical',
            size_hint=(1, None),
            height=56,
            spacing=6
        )

        btn_row = BoxLayout(
            orientation='horizontal',
            size_hint=(1, 1),
            spacing=10
        )

        clear_btn = Button(
            text="Clear",
            font_size='15sp',
            size_hint=(1, 1),
            background_normal='',
            background_color=(0.85, 0.3, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        clear_btn.bind(on_release=self._on_clear)
        btn_row.add_widget(clear_btn)

        self._recognize_btn = Button(
            text="Recognize",
            font_size='15sp',
            size_hint=(1, 1),
            background_normal='',
            background_color=(0.2, 0.6, 0.9, 1),
            color=(1, 1, 1, 1)
        )
        self._recognize_btn.bind(on_release=self._on_recognize)
        btn_row.add_widget(self._recognize_btn)

        history_btn = Button(
            text="History",
            font_size='15sp',
            size_hint=(1, 1),
            background_normal='',
            background_color=(0.6, 0.5, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        history_btn.bind(on_release=self._on_history)
        btn_row.add_widget(history_btn)

        buttons_outer.add_widget(btn_row)
        self.add_widget(buttons_outer)

    # --------------------------------------------------------
    # Auto / Manual mode
    # --------------------------------------------------------

    def _on_toggle_mode(self, instance):
        """Toggle between auto and manual recognition."""
        self.auto_mode = (instance.state == 'down')
        if self.auto_mode:
            instance.text = "Auto: ON"
            instance.background_color = (0.2, 0.7, 0.3, 1)  # green
            self._recognize_btn.text = "Recognize"
            self._recognize_btn.disabled = True
            self._recognize_btn.background_color = (0.5, 0.5, 0.5, 1)
            self.result_label.text = "Auto mode ON — draw a digit"
        else:
            instance.text = "Auto: OFF"
            instance.background_color = (0.7, 0.7, 0.7, 1)  # grey
            self._recognize_btn.text = "Recognize"
            self._recognize_btn.disabled = False
            self._recognize_btn.background_color = (0.2, 0.6, 0.9, 1)
            self.result_label.text = "Manual mode — tap Recognize"

    def _on_stroke_end(self):
        """Called by canvas when a stroke finishes (touch up)."""
        if not self.auto_mode:
            return
        # Reset debounce timer: wait AUTO_DELAY seconds, then recognize
        if self._auto_timer:
            self._auto_timer.cancel()
        self._auto_timer = Clock.schedule_once(self._auto_recognize, AUTO_DELAY)

    def _auto_recognize(self, dt):
        """Auto-recognition triggered after debounce."""
        self._auto_timer = None
        if self._pending:
            return
        if self.canvas_widget and self.canvas_widget.is_empty():
            return
        self._do_recognize_impl()

    # --------------------------------------------------------
    # Button callbacks
    # --------------------------------------------------------

    def _on_clear(self, instance):
        if self.canvas_widget:
            self.canvas_widget.clear_canvas()
        if self._auto_timer:
            self._auto_timer.cancel()
            self._auto_timer = None
        self.result_label.text = (
            "Auto mode ON — draw a digit" if self.auto_mode
            else "Canvas cleared. Draw a digit."
        )

    def _on_recognize(self, instance):
        """Manual recognize button."""
        if self._pending:
            return
        if self.canvas_widget and self.canvas_widget.is_empty():
            self.result_label.text = "Draw a digit on the canvas first"
            return
        self._do_recognize_impl()

    def _do_recognize_impl(self):
        """Shared recognition logic (used by both manual and auto)."""
        if self.recognizer is None:
            self.result_label.text = "Error: Recognizer not initialized"
            return
        if self._pending:
            return

        self._pending = True
        if not self.auto_mode:
            self._recognize_btn.disabled = True
            self._recognize_btn.text = "Working..."
        Clock.schedule_once(self._run_recognition, 0.05)

    def _run_recognition(self, dt):
        """Execute recognition (scheduled to let UI update)."""
        try:
            png_path = self.canvas_widget.export_to_image()

            from utils.preprocessing import preprocess_canvas_image
            input_array = preprocess_canvas_image(png_path, invert=True)

            digit, confidence = self.recognizer.predict(input_array)

            if confidence >= 0.80:
                color_hex = "22aa22"
            elif confidence >= 0.50:
                color_hex = "dd8800"
            else:
                color_hex = "cc3333"

            self.result_label.text = (
                f"Result: [b]{digit}[/b]  "
                f"Confidence: [color={color_hex}]{confidence:.1%}[/color]"
            )

            # Save to history database
            if self.db:
                try:
                    blob = self.db.image_to_blob(png_path)
                    self.db.add_record(digit, confidence, blob, input_type="canvas")
                except Exception:
                    pass  # DB save failure shouldn't break recognition

            try:
                os.remove(png_path)
            except OSError:
                pass

        except Exception as e:
            self.result_label.text = f"Recognition failed: {str(e)}"

        finally:
            self._pending = False
            if not self.auto_mode:
                self._recognize_btn.disabled = False
                self._recognize_btn.text = "Recognize"

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def _on_history(self, instance):
        """Open history popup."""
        if self.db is None:
            self.result_label.text = "History unavailable (no database)"
            return
        from ui.history_popup import HistoryPopup
        HistoryPopup(db=self.db).open()

    def set_recognizer(self, recognizer):
        """Set recognizer instance (injected after app startup)."""
        self.recognizer = recognizer

    def set_db(self, db):
        """Set database instance (injected after app startup)."""
        self.db = db
