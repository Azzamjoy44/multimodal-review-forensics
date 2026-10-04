"""
auth.py
-------
Local authentication using SQLite + SHA-256 password hashing + in-memory sessions.
Sessions reset on server restart (fine for a thesis demo).
"""

import datetime
import hashlib
import json
import os
import secrets
import sqlite3

DB_FILE  = os.path.join(os.path.dirname(__file__), "data", "users.db")
_sessions: dict[str, str] = {}   # token → username


def init_db():
    """Create the users table if it does not exist yet."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT    UNIQUE NOT NULL,
            password_hash TEXT   NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # One row per Analyze upload that contained a CSV file (groups its review rows).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS csv_uploads (
            batch_id    TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            filename    TEXT,
            meta_json   TEXT,
            uploaded_at TEXT NOT NULL
        )
    """)
    # One row per analyzed item (a text review, an image review, or a CSV review row).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT NOT NULL,
            review_id   TEXT NOT NULL,
            kind        TEXT NOT NULL,          -- 'text' | 'image' | 'csv'
            source      TEXT,
            csv_batch   TEXT,                   -- csv_uploads.batch_id for kind='csv'
            text        TEXT,
            is_review   INTEGER,
            result_json TEXT NOT NULL,          -- the full /analyze item dict
            uploaded_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(username)")
    conn.commit()
    conn.close()


def _new_review_id() -> str:
    """Short, human-typeable id shown on each saved result (searchable)."""
    return secrets.token_hex(3).upper()   # e.g. "3F9A2C"


def save_upload(username: str, text_item, image_items, csv_obj) -> str:
    """
    Persist the results of one /analyze request for `username`.

    text_item   : the out["text"] dict (or None)
    image_items : the out["images"] list
    csv_obj     : the out["csv"] dict (or None) — {filename, status, warning, rows,
                  review_rows, review_pct, items:[...]}

    Each saved item gets a `review_id` and `uploaded_at` injected in place (so the
    /analyze response carries them too). Returns the shared upload timestamp.
    """
    uploaded_at = datetime.datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(DB_FILE)
    try:
        def _insert(kind, source, csv_batch, item):
            rid = _new_review_id()
            item["review_id"]   = rid
            item["uploaded_at"] = uploaded_at
            conn.execute(
                "INSERT INTO analyses (username, review_id, kind, source, csv_batch, "
                "text, is_review, result_json, uploaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (username, rid, kind, source, csv_batch,
                 item.get("text", ""), 1 if item.get("is_review") else 0,
                 json.dumps(item), uploaded_at))

        if text_item:
            _insert("text", text_item.get("source", "text"), None, text_item)
        for it in (image_items or []):
            _insert("image", it.get("filename") or it.get("source", ""), None, it)
        if csv_obj and csv_obj.get("items"):
            batch_id = secrets.token_hex(8)
            meta = {k: csv_obj.get(k) for k in
                    ("filename", "status", "warning", "message",
                     "rows", "review_rows", "review_pct")}
            conn.execute(
                "INSERT INTO csv_uploads (batch_id, username, filename, meta_json, uploaded_at) "
                "VALUES (?,?,?,?,?)",
                (batch_id, username, csv_obj.get("filename", ""), json.dumps(meta), uploaded_at))
            for it in csv_obj["items"]:
                _insert("csv", it.get("source", ""), batch_id, it)
        conn.commit()
    finally:
        conn.close()
    return uploaded_at


def get_uploads(username: str) -> dict:
    """Return the user's saved analyses as {text:[...], images:[...], csv_groups:[...]}."""
    conn = sqlite3.connect(DB_FILE)
    item_rows = conn.execute(
        "SELECT kind, csv_batch, result_json FROM analyses WHERE username=? ORDER BY id ASC",
        (username,)).fetchall()
    csv_rows = conn.execute(
        "SELECT batch_id, filename, meta_json, uploaded_at FROM csv_uploads "
        "WHERE username=? ORDER BY rowid ASC", (username,)).fetchall()
    conn.close()

    csv_groups = {}
    for batch_id, filename, meta_json, uploaded_at in csv_rows:
        csv_groups[batch_id] = {
            "batch_id": batch_id, "filename": filename,
            "meta": json.loads(meta_json) if meta_json else {},
            "uploaded_at": uploaded_at, "items": []}

    text_items, image_items = [], []
    for kind, csv_batch, result_json in item_rows:
        item = json.loads(result_json)
        if kind == "text":
            text_items.append(item)
        elif kind == "image":
            image_items.append(item)
        elif kind == "csv" and csv_batch in csv_groups:
            csv_groups[csv_batch]["items"].append(item)

    return {"text": text_items, "images": image_items,
            "csv_groups": list(csv_groups.values())}


def delete_item(username: str, review_id: str) -> int:
    """Delete one saved text/image review by its review_id (scoped to the user)."""
    conn = sqlite3.connect(DB_FILE)
    try:
        n = conn.execute(
            "DELETE FROM analyses WHERE username=? AND review_id=?",
            (username, review_id)).rowcount
        conn.commit()
    finally:
        conn.close()
    return n


def delete_csv_group(username: str, batch_id: str) -> int:
    """Delete a whole uploaded CSV: all its review rows + its group record (scoped to the user)."""
    conn = sqlite3.connect(DB_FILE)
    try:
        n = conn.execute(
            "DELETE FROM analyses WHERE username=? AND csv_batch=?",
            (username, batch_id)).rowcount
        conn.execute(
            "DELETE FROM csv_uploads WHERE username=? AND batch_id=?",
            (username, batch_id))
        conn.commit()
    finally:
        conn.close()
    return n


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (True, "") on success, (False, reason) on failure.
    """
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, _hash(password)),
        )
        conn.commit()
        conn.close()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "Username already taken."


def login_user(username: str, password: str) -> str | None:
    """
    Validate credentials and create a session.
    Returns a session token on success, None on failure.
    """
    username = username.strip()
    conn = sqlite3.connect(DB_FILE)
    row  = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row and row[0] == _hash(password):
        token = secrets.token_hex(32)
        _sessions[token] = username
        return token
    return None


def get_session_user(token: str) -> str | None:
    """Return the username for a valid session token, or None."""
    return _sessions.get(token)


def logout_user(token: str):
    _sessions.pop(token, None)
