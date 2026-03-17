import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from .session import Session


class Storage:
    def __init__(self, db_path: str = "data/app.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                last_subject TEXT,
                history_json TEXT,
                last_retrieved_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def get_user(self, username: str) -> Optional[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return dict(row) if row else None

    def create_user(self, username: str, password_hash: str, salt: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if not row:
            return None
        return _row_to_session(row)

    def create_session(self, user_id: str) -> Session:
        session_id = str(uuid.uuid4())
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO sessions (id, user_id, last_subject, history_json, last_retrieved_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, None, "[]", "[]"),
        )
        self._conn.commit()
        return Session(id=session_id, user_id=user_id)

    def save_session(self, session: Session) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE sessions
            SET last_subject = ?, history_json = ?, last_retrieved_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                session.last_subject,
                json.dumps(session.history, ensure_ascii=False),
                json.dumps(session.last_retrieved, ensure_ascii=False),
                session.id,
            ),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0


def _row_to_session(row: sqlite3.Row) -> Session:
    history = json.loads(row["history_json"] or "[]")
    last_retrieved = json.loads(row["last_retrieved_json"] or "[]")
    return Session(
        id=row["id"],
        history=history,
        last_subject=row["last_subject"],
        last_retrieved=last_retrieved,
        user_id=row["user_id"],
    )
