"""
History popup — shows past recognition records with view/delete actions.
"""

from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.core.image import Image as CoreImage
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
import io


class HistoryPopup(Popup):
    """Popup window displaying recognition history."""

    def __init__(self, db, **kwargs):
        super().__init__(**kwargs)
        self.db = db
        self.title = "History"
        self.size_hint = (0.92, 0.88)
        self.separator_height = 0

        self._build_ui()
        self._load_records()

    def _build_ui(self):
        """Build popup layout."""
        root = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))

        # Header
        header = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height=dp(44),
            spacing=dp(8)
        )
        header.add_widget(Label(
            text="Recognition History",
            font_size='18sp',
            size_hint=(1, None),
            height=dp(44),
            halign='left',
            valign='middle',
            bold=True,
            color=(0.2, 0.2, 0.2, 1)
        ))

        # Delete all button
        del_all_btn = Button(
            text="Delete All",
            font_size='13sp',
            size_hint=(None, None),
            width=dp(90),
            height=dp(36),
            background_normal='',
            background_color=(0.9, 0.3, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        del_all_btn.bind(on_release=self._on_delete_all)
        header.add_widget(del_all_btn)

        root.add_widget(header)

        # Scrollable record list
        self._scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self._list_layout = GridLayout(
            cols=1,
            size_hint_y=None,
            spacing=dp(6),
            padding=[0, 0, 0, dp(6)]
        )
        self._list_layout.bind(minimum_height=self._list_layout.setter('height'))
        self._scroll.add_widget(self._list_layout)
        root.add_widget(self._scroll)

        # Bottom: count + close
        bottom = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height=dp(44),
            spacing=dp(8)
        )
        self._count_label = Label(
            text="0 records",
            font_size='14sp',
            size_hint=(1, None),
            height=dp(44),
            halign='left',
            valign='middle',
            color=(0.5, 0.5, 0.5, 1)
        )
        bottom.add_widget(self._count_label)

        close_btn = Button(
            text="Close",
            font_size='15sp',
            size_hint=(None, None),
            width=dp(80),
            height=dp(40),
            background_normal='',
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1)
        )
        close_btn.bind(on_release=self.dismiss)
        bottom.add_widget(close_btn)

        root.add_widget(bottom)
        self.content = root

    def _load_records(self):
        """Load records from DB and populate the list."""
        self._list_layout.clear_widgets()
        records = self.db.get_all_records(limit=200)
        self._count_label.text = f"{len(records)} record(s)"

        if not records:
            self._list_layout.add_widget(Label(
                text="No history yet.\nDraw digits and recognize them!",
                font_size='15sp',
                size_hint_y=None,
                height=dp(80),
                halign='center',
                valign='middle',
                color=(0.5, 0.5, 0.5, 1)
            ))
            return

        for rec in records:
            row = self._build_record_row(rec)
            self._list_layout.add_widget(row)

    def _build_record_row(self, record: dict) -> BoxLayout:
        """Build a single history row."""
        row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(64),
            spacing=dp(8),
            padding=[dp(6), dp(4), dp(6), dp(4)]
        )
        # Background
        with row.canvas.before:
            Color(0.97, 0.97, 0.97, 1)
            rect = Rectangle(pos=row.pos, size=row.size)
            row.bind(pos=lambda inst, val: setattr(rect, 'pos', val))
            row.bind(size=lambda inst, val: setattr(rect, 'size', val))

        # Info section
        info = BoxLayout(orientation='vertical', size_hint=(1, 1), spacing=0)
        info.add_widget(Label(
            text=f"Digit: [b]{record['predicted_digit']}[/b]  "
                 f"Confidence: {record['confidence']:.1%}",
            font_size='15sp',
            size_hint=(1, 0.55),
            halign='left',
            valign='bottom',
            markup=True,
            color=(0.1, 0.1, 0.1, 1)
        ))
        info.add_widget(Label(
            text=f"{record['timestamp']}  [{record['input_type']}]",
            font_size='11sp',
            size_hint=(1, 0.45),
            halign='left',
            valign='top',
            color=(0.5, 0.5, 0.5, 1)
        ))
        row.add_widget(info)

        # View button
        view_btn = Button(
            text="View",
            font_size='12sp',
            size_hint=(None, None),
            width=dp(50),
            height=dp(32),
            pos_hint={'center_y': 0.5},
            background_normal='',
            background_color=(0.2, 0.6, 0.9, 1),
            color=(1, 1, 1, 1)
        )
        view_btn.record_id = record['id']
        view_btn.bind(on_release=self._on_view_record)
        row.add_widget(view_btn)

        # Delete button
        del_btn = Button(
            text="Del",
            font_size='12sp',
            size_hint=(None, None),
            width=dp(44),
            height=dp(32),
            pos_hint={'center_y': 0.5},
            background_normal='',
            background_color=(0.9, 0.3, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        del_btn.record_id = record['id']
        del_btn.bind(on_release=self._on_delete_record)
        row.add_widget(del_btn)

        return row

    def _on_view_record(self, instance):
        """View a record's image in a detail popup."""
        record = self.db.get_record_detail(instance.record_id)
        if not record:
            return

        # Build detail popup
        detail = Popup(
            title=f"Digit {record['predicted_digit']}",
            size_hint=(0.75, 0.6),
            separator_height=0
        )
        layout = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))

        # Reconstructed image
        try:
            pil_img = self.db.blob_to_image(record['image_png'])
            # Scale up for display
            display_size = (dp(200), dp(200))
            pil_img_resized = pil_img.resize(
                (int(display_size[0]), int(display_size[1])),
            )

            buf = io.BytesIO()
            pil_img_resized.save(buf, format='PNG')
            buf.seek(0)

            img_widget = KivyImage(
                source='',
                size_hint=(1, 1),
            )
            img_widget.texture = CoreImage(buf, ext='png').texture
            layout.add_widget(img_widget)
        except Exception:
            layout.add_widget(Label(text="(Image unavailable)"))

        # Info
        layout.add_widget(Label(
            text=(
                f"Predicted: [b]{record['predicted_digit']}[/b]\n"
                f"Confidence: {record['confidence']:.1%}\n"
                f"Time: {record['timestamp']}\n"
                f"Source: {record['input_type']}"
            ),
            font_size='14sp',
            size_hint=(1, None),
            height=dp(80),
            markup=True,
            halign='center',
            valign='middle',
            color=(0.1, 0.1, 0.1, 1)
        ))

        close_btn = Button(
            text="Close",
            font_size='14sp',
            size_hint=(1, None),
            height=dp(40),
            background_normal='',
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1)
        )
        close_btn.bind(on_release=detail.dismiss)
        layout.add_widget(close_btn)

        detail.content = layout
        detail.open()

    def _on_delete_record(self, instance):
        """Delete a single record."""
        self.db.delete_record(instance.record_id)
        # Remove the parent row from layout
        row = instance.parent
        if row and row.parent == self._list_layout:
            self._list_layout.remove_widget(row)
        self._count_label.text = f"{self.db.count()} record(s)"

    def _on_delete_all(self, instance):
        """Delete all records after confirmation."""
        self.db.delete_all()
        self._load_records()
