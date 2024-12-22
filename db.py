"""
db.py - Handles database connection & queries.
"""

import sqlite3
from passlib.hash import bcrypt

DATABASE_NAME = "vignettes.db"

def create_connection():
    """Create and return a connection to the SQLite database."""
    return sqlite3.connect(DATABASE_NAME, check_same_thread=False)

def init_db():
    """Initialize the database (creates tables if they don't already exist)."""
    conn = create_connection()
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    """)

    # Create vignettes table with columns for topic, initial_vignette, final_vignette, conversation
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vignettes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT,
        initial_vignette TEXT,
        final_vignette TEXT,
        conversation TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

# ---------------- USER FUNCTIONS ----------------

def create_user(username: str, password: str) -> bool:
    """
    Create a new user with hashed password.
    Returns True if successful, False if the username already exists.
    """
    conn = create_connection()
    cursor = conn.cursor()
    try:
        password_hash = bcrypt.hash(password)
        cursor.execute("""
            INSERT INTO users (username, password_hash)
            VALUES (?, ?)
        """, (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    finally:
        conn.close()

def get_user(username: str):
    """
    Retrieve user record by username.
    Returns tuple (id, username, password_hash) or None if not found.
    """
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, password_hash
        FROM users
        WHERE username = ?
    """, (username,))
    user_data = cursor.fetchone()
    conn.close()
    return user_data

# ---------------- VIGNETTE FUNCTIONS ----------------

def save_vignette(user_id: int, topic: str, init_vig: str, final_vig: str, conv_json: str):
    """
    Save a newly generated vignette to the database, including topic, 
    initial version, final version, and entire conversation JSON.
    """
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vignettes (user_id, topic, initial_vignette, final_vignette, conversation)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, topic, init_vig, final_vig, conv_json))
    conn.commit()
    conn.close()

def get_user_vignettes(user_id: int):
    """
    Retrieve all vignettes created by a given user.
    Returns a list of (id, topic, initial_vignette, final_vignette, conversation).
    """
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, topic, initial_vignette, final_vignette, conversation
        FROM vignettes
        WHERE user_id=?
        ORDER BY id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows
