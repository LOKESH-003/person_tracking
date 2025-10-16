# file: create_people_db.py
import sqlite3

DB_PATH = "people.db"

def init_database():
    """Create database and table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized.")


def insert_names(names):
    """Insert a list of names into the database in order."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for name in names:
        try:
            c.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name,))
            print(f"💾 Inserted: {name}")
        except Exception as e:
            print(f"⚠️ Skipped {name}: {e}")
    conn.commit()
    conn.close()
    print("✅ All names inserted successfully.")


if __name__ == "__main__":
    init_database()

    # 🧠 Define your people list (order matters)
    people_names = [
        "Lokesh",
        "Shriyans",
        "Arun",
        "Sanjay",
        "Naveen",
        "Ramesh",
        "prasanth",
        "Vivek",
        "Balaji",
        "Santhos",
        "Saravanan",
        "Praveen",
        "Mukesh",
        "Rajesh"
    ]

    insert_names(people_names)
