"""
SQLite database manager for storing recognition history.
"""

import os
import sqlite3
import io
from datetime import datetime
from PIL import Image


DB_FILENAME = "digitnote_history.db"


def get_db_path():
    """Get the database file path (in the project root, or app data dir)."""
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        # PyInstaller bundle: use the directory containing the exe (or temp dir)
        # Prefer a writable location for the database
        import sysconfig
        project_root = os.path.dirname(_sys.executable)
        # Fallback: use user's app data directory
        if not os.access(project_root, os.W_OK):
            project_root = os.path.join(os.path.expanduser('~'), '.digitnote')
            os.makedirs(project_root, exist_ok=True)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, DB_FILENAME)


class HistoryDB:
    """Manages the recognition history SQLite database."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_db_path()
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predicted_digit INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    image_png BLOB NOT NULL,
                    input_type TEXT DEFAULT 'canvas'
                )
            """)
            conn.commit()

    def add_record(self, predicted_digit: int, confidence: float,
                   image_png: bytes, input_type: str = "canvas") -> int:
        """Insert a recognition record. Returns the new row id."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO history (timestamp, predicted_digit, confidence, image_png, input_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, predicted_digit, confidence, image_png, input_type)
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_records(self, limit: int = 200) -> list:
        """Retrieve all records, newest first. Returns list of dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, timestamp, predicted_digit, confidence, input_type "
                "FROM history ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_image(self, record_id: int) -> bytes:
        """Retrieve the PNG image blob for a record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT image_png FROM history WHERE id = ?", (record_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_record_detail(self, record_id: int) -> dict:
        """Get full record detail including image."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM history WHERE id = ?", (record_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_record(self, record_id: int):
        """Delete a single record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
            conn.commit()

    def delete_all(self):
        """Delete all history records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()

    def count(self) -> int:
        """Return total number of records."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM history")
            return cursor.fetchone()[0]

    @staticmethod
    def image_to_blob(image_path: str) -> bytes:
        """Read a PNG file and return its bytes."""
        with open(image_path, 'rb') as f:
            return f.read()

    @staticmethod
    def blob_to_image(blob: bytes) -> Image.Image:
        """Convert a PNG blob back to a PIL Image."""
        return Image.open(io.BytesIO(blob))
