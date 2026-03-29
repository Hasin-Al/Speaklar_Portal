import re
from typing import Optional, Tuple

from .indexer import Indexer
from .session import Session

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


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
            "ব্র্যান্ড",
            "ক্যাটাগরি",
            "ইউনিট",
            "ফিচার",
            "সুবিধা",
            "ডেলিভারি",
            "ওয়ারেন্টি",
            "ওয়ারেন্টি",
            "রিটার্ন",
            "ছাড়",
            "ডিসকাউন্ট",
        ]
        self._stopwords = {
            "কি",
            "কী",
            "কত",
            "আমাদের",
            "আপনাদের",
            "তোমাদের",
            "তাদের",
            "এই",
            "ওই",
            "এটা",
            "ওটা",
            "এর",
            "এটার",
            "ওটার",
            "আমরা",
            "আপনারা",
            "কোম্পানি",
            "পণ্য",
            "প্রোডাক্ট",
            "বিক্রি",
            "পাওয়া",
            "পাওয়া",
            "আছে",
            "উপলব্ধ",
            "দাম",
            "টাকা",
            "সুবিধা",
            "ফিচার",
            "কেমন",
            "করে",
            "করেন",
            "ব্র্যান্ড",
            "ক্যাটাগরি",
            "ইউনিট",
            "সাইজ",
            "ডেলিভারি",
            "ওয়ারেন্টি",
            "ওয়ারেন্টি",
            "রিটার্ন",
            "ছাড়",
            "ডিসকাউন্ট",
        }
        self._subject_markers = [
            "দাম",
            "বিক্রি",
            "পাওয়া",
            "পাওয়া",
            "আছে",
            "উপলব্ধ",
            "সুবিধা",
            "ফিচার",
            "ব্র্যান্ড",
            "ক্যাটাগরি",
            "ইউনিট",
            "ডেলিভারি",
            "ওয়ারেন্টি",
            "ওয়ারেন্টি",
        ]

    def resolve(self, query: str, session: Session) -> Tuple[str, Optional[str], bool, Optional[bool]]:
        if session.last_subject and len(session.last_subject.strip()) < 2:
            session.last_subject = None
            session.last_subject_known = None
        entity = self._indexer.find_explicit_entity(query)
        subject = entity
        subject_known = True if entity else None
        if not subject:
            candidate = self._extract_candidate_subject(query)
            if candidate:
                matched = self._indexer.find_explicit_entity(candidate)
                subject = matched or candidate
                subject_known = True if matched else False

        if subject is None and session.subject_memory:
            memory_match = self._match_from_memory(query, session)
            if memory_match:
                subject = str(memory_match.get("name", "")).strip()
                subject_known = bool(memory_match.get("known", False))

        is_anaphor = subject is None and self._looks_anaphoric(query)

        if subject:
            session.last_subject = subject
            session.last_subject_known = subject_known
            session.remember_subject(subject, bool(subject_known))

        if is_anaphor and session.last_subject:
            expanded = f"{session.last_subject} {query}".strip()
            return expanded, session.last_subject, True, session.last_subject_known

        if is_anaphor and session.subject_memory:
            best = session.best_subject()
            if best:
                name = str(best.get("name", "")).strip()
                if name:
                    session.last_subject = name
                    session.last_subject_known = bool(best.get("known", False))
                    expanded = f"{name} {query}".strip()
                    return expanded, name, True, bool(best.get("known", False))

        if is_anaphor and session.last_retrieved:
            name = str(session.last_retrieved[0].get("name", "")).strip()
            if name:
                session.last_subject = name
                session.last_subject_known = True
                expanded = f"{name} {query}".strip()
                session.remember_subject(name, True)
                return expanded, name, True, True

        return query, subject, False, subject_known

    def _looks_anaphoric(self, query: str) -> bool:
        lowered = query.lower()
        for marker in self._anaphor_markers:
            if marker in lowered:
                return True
        return False

    def _extract_candidate_subject(self, query: str) -> Optional[str]:
        tokens = [t for t in _TOKEN_RE.findall(query) if len(t) >= 2]
        if not tokens:
            return None

        lowered = [t.lower() for t in tokens]
        marker_index = None
        for i, tok in enumerate(lowered):
            if tok in self._subject_markers:
                marker_index = i
                break

        if marker_index is not None:
            parts = []
            idx = marker_index - 1
            while idx >= 0 and len(parts) < 3:
                tok = tokens[idx]
                if tok.lower() in self._stopwords:
                    break
                parts.append(tok)
                idx -= 1
            if parts:
                return " ".join(reversed(parts)).strip()

        # Fallback: pick the longest non-stopword token
        candidates = [t for t in tokens if t.lower() not in self._stopwords]
        if not candidates:
            return None
        return max(candidates, key=len).strip()

    def _match_from_memory(self, query: str, session: Session) -> Optional[dict]:
        q_lower = query.lower()
        q_tokens = {t for t in _TOKEN_RE.findall(q_lower) if len(t) >= 2}
        best = None
        best_score = 0
        for item in session.subject_memory:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            name_lower = name.lower()
            if name_lower and name_lower in q_lower:
                return item
            name_tokens = {t for t in _TOKEN_RE.findall(name_lower) if len(t) >= 2}
            overlap = len(q_tokens & name_tokens)
            if overlap > best_score:
                best_score = overlap
                best = item
        if best_score > 0:
            return best
        return None

    def _refresh_memory_from_history(self, session: Session) -> None:
        for msg in session.history:
            if msg.get("role") != "user":
                continue
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            entity = self._indexer.find_explicit_entity(text)
            if entity:
                session.remember_subject(entity, True)
                continue
            candidate = self._extract_candidate_subject(text)
            if candidate:
                matched = self._indexer.find_explicit_entity(candidate)
                session.remember_subject(matched or candidate, bool(matched))
