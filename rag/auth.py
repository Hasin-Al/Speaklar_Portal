import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Optional

from .storage import Storage


@dataclass
class User:
    username: str
    password_hash: str
    salt: str


class UserStore:
    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    def register(self, username: str, password: str) -> User:
        if self._storage.get_user(username):
            raise ValueError("username already exists")
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        user = User(username=username, password_hash=password_hash, salt=salt)
        self._storage.create_user(username, password_hash, salt)
        return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        row = self._storage.get_user(username)
        if not row:
            return None
        user = User(
            username=row["username"],
            password_hash=row["password_hash"],
            salt=row["salt"],
        )
        expected = _hash_password(password, user.salt)
        if hmac.compare_digest(expected, user.password_hash):
            return user
        return None


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
