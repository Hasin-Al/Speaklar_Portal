from typing import Optional, Tuple

from .indexer import Indexer
from .session import Session


class Resolver:
    def __init__(self, indexer: Indexer):
        self._indexer = indexer
        self._anaphor_markers = [
            "দাম",
            "কত",
            "এটা",
            "ওটা",
            "এর",
            "এটার",
            "ওটার",
            "কেজি",
            "প্যাকেট",
            "সাইজ",
        ]

    def resolve(self, query: str, session: Session) -> Tuple[str, Optional[str], bool]:
        entity = self._indexer.find_explicit_entity(query)
        is_anaphor = entity is None and self._looks_anaphoric(query)

        if entity:
            session.last_subject = entity

        if is_anaphor and session.last_subject:
            expanded = f"{session.last_subject} {query}".strip()
            return expanded, session.last_subject, True

        return query, entity, False

    def _looks_anaphoric(self, query: str) -> bool:
        lowered = query.lower()
        for marker in self._anaphor_markers:
            if marker in lowered:
                return True
        return False
