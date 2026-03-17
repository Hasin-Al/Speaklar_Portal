from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import Storage


MAX_HISTORY_MESSAGES = 6


@dataclass
class Session:
    id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    last_subject: Optional[str] = None
    last_retrieved: List[dict] = field(default_factory=list)
    user_id: Optional[str] = None

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def recent_history(self) -> List[Dict[str, str]]:
        return list(self.history)


class SessionStore:
    def __init__(self, storage: "Storage") -> None:
        self._storage = storage

    def get(self, session_id: Optional[str]) -> Session:
        if session_id:
            existing = self._storage.get_session(session_id)
            if existing:
                return existing
        return self.create_for_user("anonymous")

    def get_existing(self, session_id: str) -> Optional[Session]:
        return self._storage.get_session(session_id)

    def create_for_user(self, user_id: str) -> Session:
        return self._storage.create_session(user_id)

    def delete(self, session_id: str) -> bool:
        return self._storage.delete_session(session_id)

    def save(self, session: Session) -> None:
        self._storage.save_session(session)
