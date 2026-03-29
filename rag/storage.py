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
                last_subject_known INTEGER,
                history_json TEXT,
                last_retrieved_json TEXT,
                subject_memory_json TEXT,
                pending_intent TEXT,
                pending_options_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._ensure_session_columns(cur)
        # Clear context on server start but keep session ids.
        cur.execute(
            """
            UPDATE sessions
            SET last_subject = NULL,
                last_subject_known = NULL,
                history_json = "[]",
                last_retrieved_json = "[]",
                subject_memory_json = "[]",
                pending_intent = NULL,
                pending_options_json = "[]",
                updated_at = CURRENT_TIMESTAMP
            """
        )
        self._conn.commit()

    def _ensure_session_columns(self, cur: sqlite3.Cursor) -> None:
        cur.execute("PRAGMA table_info(sessions)")
        cols = {row["name"] for row in cur.fetchall()}
        if "last_subject_known" not in cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN last_subject_known INTEGER")
        if "subject_memory_json" not in cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN subject_memory_json TEXT")
        if "pending_intent" not in cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN pending_intent TEXT")
        if "pending_options_json" not in cols:
            cur.execute("ALTER TABLE sessions ADD COLUMN pending_options_json TEXT")

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
            INSERT INTO sessions (
                id,
                user_id,
                last_subject,
                last_subject_known,
                history_json,
                last_retrieved_json,
                subject_memory_json,
                pending_intent,
                pending_options_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, None, None, "[]", "[]", "[]", None, "[]"),
        )
        self._conn.commit()
        return Session(id=session_id, user_id=user_id)

    def save_session(self, session: Session) -> None:
        last_subject = session.last_subject
        last_subject_known = session.last_subject_known
        if last_subject and len(last_subject.strip()) < 2:
            last_subject = None
            last_subject_known = None
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE sessions
            SET last_subject = ?,
                last_subject_known = ?,
                history_json = ?,
                last_retrieved_json = ?,
                subject_memory_json = ?,
                pending_intent = ?,
                pending_options_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                last_subject,
                None if last_subject_known is None else int(last_subject_known),
                json.dumps(session.history, ensure_ascii=False),
                json.dumps(session.last_retrieved, ensure_ascii=False),
                json.dumps(session.subject_memory, ensure_ascii=False),
                session.pending_intent,
                json.dumps(session.pending_options, ensure_ascii=False),
                session.id,
            ),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def clear_session_context(self, session_id: str) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE sessions
            SET last_subject = NULL,
                last_subject_known = NULL,
                history_json = "[]",
                last_retrieved_json = "[]",
                subject_memory_json = "[]",
                pending_intent = NULL,
                pending_options_json = "[]",
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0


def _row_to_session(row: sqlite3.Row) -> Session:
    history = json.loads(row["history_json"] or "[]")
    last_retrieved = json.loads(row["last_retrieved_json"] or "[]")
    subject_memory = json.loads(row["subject_memory_json"] or "[]")
    last_subject_known = row["last_subject_known"]
    last_subject = row["last_subject"]
    pending_options = json.loads(row["pending_options_json"] or "[]")
    if last_subject and len(last_subject.strip()) < 2:
        last_subject = None
        last_subject_known = None
    subject_memory = [
        item for item in subject_memory if len(str(item.get("name", "")).strip()) >= 2
    ]
    return Session(
        id=row["id"],
        history=history,
        last_subject=last_subject,
        last_subject_known=None if last_subject_known is None else bool(last_subject_known),
        last_retrieved=last_retrieved,
        user_id=row["user_id"],
        subject_memory=subject_memory,
        pending_intent=row["pending_intent"],
        pending_options=pending_options,
    )
