import sqlite3
import os

class ImageCache:
    def __init__(self, db_path="image_cache.sqlite3"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        original_url TEXT PRIMARY KEY,
                        s3_url TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
        except Exception as e:
            print(f"Error initializing image cache DB: {e}")

    def get_s3_url(self, original_url):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT s3_url FROM images WHERE original_url = ?", (original_url,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def save_s3_url(self, original_url, s3_url):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO images (original_url, s3_url) VALUES (?, ?)",
                    (original_url, s3_url)
                )
        except Exception as e:
            print(f"Error saving to image cache DB: {e}")

image_cache = ImageCache()
