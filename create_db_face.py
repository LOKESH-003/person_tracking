# file: create_embeddings_db.py
import sqlite3
import numpy as np

DB_PATH = "embeddings_v3.db"

def init_database():
    """Create database and table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS people_embeddings (
            name TEXT PRIMARY KEY,
            body_embedding BLOB,
            face_embedding BLOB,
            photo BLOB
        )
    """)
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}.")

def np_to_blob(arr: np.ndarray) -> bytes:
    """Convert numpy array to BLOB for storage."""
    return arr.astype(np.float32).tobytes()

def blob_to_np(blob: bytes) -> np.ndarray:
    """Convert BLOB back to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)

def insert_person(name, body_embedding=None, face_embedding=None, photo_bytes=None):
    """Insert a person into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO people_embeddings
        (name, body_embedding, face_embedding, photo)
        VALUES (?, ?, ?, ?)
    """, (
        name,
        np_to_blob(body_embedding) if body_embedding is not None else None,
        np_to_blob(face_embedding) if face_embedding is not None else None,
        photo_bytes
    ))
    conn.commit()
    conn.close()
    print(f"💾 Inserted/Updated: {name}")

if __name__ == "__main__":
    init_database()

    # Optional: predefine known people names (without embeddings yet)
    people_names = [
        "Lokesh", "Shriyans", "Arun", "Sanjay", "Naveen",
        "Ramesh", "Prasanth", "Vivek", "Balaji", "Santhos",
        "Saravanan", "Praveen", "Mukesh", "Rajesh"
    ]

    for name in people_names:
        insert_person(name)
