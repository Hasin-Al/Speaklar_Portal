import json
import math
import re
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _parse_products_text(f) -> List[dict]:
    key_map = {
        "id": "id",
        "আইডি": "id",
        "name": "name",
        "নাম": "name",
        "price": "price",
        "দাম": "price",
        "unit": "unit",
        "ইউনিট": "unit",
        "brand": "brand",
        "ব্র্যান্ড": "brand",
        "category": "category",
        "ক্যাটাগরি": "category",
        "sku": "sku",
        "স্কু": "sku",
    }

    products: List[dict] = []
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        item: Dict[str, object] = {}

        # Key-value format support: "নাম=নুডুলস | দাম=20 টাকা"
        if "=" in line:
            parts = [p.strip() for p in line.split("|")]
            for part in parts:
                if "=" not in part:
                    continue
                key, val = part.split("=", 1)
                key_norm = key.strip().lower()
                mapped = key_map.get(key.strip(), key_map.get(key_norm))
                if not mapped:
                    continue
                val = val.strip()
                if mapped == "price":
                    m = re.search(r"[0-9]+(?:\\.[0-9]+)?", val)
                    if m:
                        num = m.group(0)
                        item[mapped] = int(num) if num.isdigit() else float(num)
                    else:
                        item[mapped] = val
                elif mapped == "id":
                    item[mapped] = int(val) if val.isdigit() else val
                else:
                    item[mapped] = val
        else:
            # Simple format support: "নুডুলস 20"
            # Try CSV-ish first.
            if "," in line:
                parts = [p.strip() for p in line.split(",", 1)]
                if len(parts) == 2:
                    item["name"] = parts[0]
                    price_part = parts[1]
                else:
                    parts = line.split()
                    price_part = parts[-1] if parts else ""
                    item["name"] = " ".join(parts[:-1]).strip()
            else:
                parts = line.split()
                price_part = parts[-1] if parts else ""
                item["name"] = " ".join(parts[:-1]).strip()

            m = re.search(r"[0-9]+(?:\\.[0-9]+)?", price_part)
            if m:
                num = m.group(0)
                item["price"] = int(num) if num.isdigit() else float(num)
            else:
                # Treat as knowledge paragraph (no explicit price)
                item["text"] = line
                if "।" in line:
                    item["name"] = line.split("।", 1)[0].strip()
                else:
                    item["name"] = line[:80].strip()

        if item.get("name"):
            products.append(item)
    return products

class Indexer:
    def __init__(self, products: List[dict]):
        self.products = products
        self.product_ratio = self._compute_product_ratio(products)
        self.is_product_kb = self.product_ratio >= 0.6
        self._doc_tfs: List[Dict[str, int]] = []
        self._doc_norms: List[float] = []
        self._inverted: Dict[str, List[tuple]] = defaultdict(list)
        self._idf: Dict[str, float] = {}
        self._name_list: List[str] = [p.get("name", "") for p in products]
        self._name_tokens: List[set] = [set(_tokenize(name)) for name in self._name_list]
        self._search_cache: "OrderedDict[tuple, List[dict]]" = OrderedDict()
        self._search_cache_size = 128
        self._build()

    @classmethod
    def from_file(cls, path: str) -> "Indexer":
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(2048)
            f.seek(0)
            stripped = head.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                products = json.load(f)
                return cls(products)
            products = _parse_products_text(f)
            return cls(products)

    @staticmethod
    def _compute_product_ratio(products: List[dict]) -> float:
        if not products:
            return 0.0
        with_price = sum(1 for p in products if p.get("price") is not None)
        return with_price / max(1, len(products))

    def _build(self) -> None:
        df: Dict[str, int] = defaultdict(int)

        for i, p in enumerate(self.products):
            text = " ".join(
                [
                    str(p.get("name", "")),
                    str(p.get("text", "")),
                    str(p.get("description", "")),
                    str(p.get("category", "")),
                    str(p.get("brand", "")),
                ]
            )
            tokens = _tokenize(text)
            tf = Counter(tokens)
            self._doc_tfs.append(dict(tf))

            for tok in tf.keys():
                df[tok] += 1

            for tok, count in tf.items():
                self._inverted[tok].append((i, count))

        n_docs = max(1, len(self.products))
        for tok, d in df.items():
            # Smooth IDF to avoid division by zero.
            self._idf[tok] = math.log((n_docs + 1) / (d + 1)) + 1.0

        self._doc_norms = [0.0] * len(self.products)
        for i, tf in enumerate(self._doc_tfs):
            acc = 0.0
            for tok, count in tf.items():
                idf = self._idf.get(tok, 0.0)
                if idf == 0.0:
                    continue
                weight = (1.0 + math.log(count)) * idf
                acc += weight * weight
            self._doc_norms[i] = math.sqrt(acc) if acc > 0.0 else 1.0

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        cache_key = (query, top_k)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_weights = {}
        q_norm_acc = 0.0
        for tok, count in q_tf.items():
            idf = self._idf.get(tok)
            if not idf:
                continue
            weight = (1.0 + math.log(count)) * idf
            q_weights[tok] = weight
            q_norm_acc += weight * weight
        q_norm = math.sqrt(q_norm_acc) if q_norm_acc > 0.0 else 1.0

        scores: Dict[int, float] = defaultdict(float)
        for tok, q_weight in q_weights.items():
            for doc_id, tf in self._inverted.get(tok, []):
                d_weight = (1.0 + math.log(tf)) * self._idf[tok]
                scores[doc_id] += q_weight * d_weight

        # Substring bonus for simple Bangla morphology handling.
        q_lower = query.lower()
        for doc_id in list(scores.keys()):
            name = self._name_list[doc_id]
            if not name:
                continue
            name_lower = name.lower()
            bonus = 0.0
            if name_lower in q_lower or q_lower in name_lower:
                bonus += 0.25
            else:
                for tok in q_tokens:
                    if tok and tok in name_lower:
                        bonus += 0.05
                        break
            scores[doc_id] += bonus

        ranked = []
        for doc_id, score in scores.items():
            norm_score = score / (self._doc_norms[doc_id] * q_norm)
            ranked.append((norm_score, doc_id))
        ranked.sort(reverse=True)

        results = []
        for _, doc_id in ranked[: max(1, top_k)]:
            results.append(self.products[doc_id])
        self._search_cache[cache_key] = results
        if len(self._search_cache) > self._search_cache_size:
            self._search_cache.popitem(last=False)
        return results

    def find_explicit_entity(self, query: str) -> Optional[str]:
        q = query.lower()
        matches = [name for name in self._name_list if name and name.lower() in q]
        if matches:
            return max(matches, key=len)

        q_tokens = set(_tokenize(query))
        best_name = None
        best_score = 0.0
        for name, tokens in zip(self._name_list, self._name_tokens):
            if not tokens:
                continue
            overlap = len(tokens & q_tokens)
            if overlap == 0:
                continue
            score = overlap / len(tokens)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= 0.6:
            return best_name
        return None

    def find_by_name(self, name: str) -> List[dict]:
        name = name.strip()
        if not name:
            return []
        return [p for p in self.products if str(p.get("name", "")).strip() == name]
