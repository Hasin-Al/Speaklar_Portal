from typing import List, Optional, Tuple


def try_fast_answer(query: str, results: List[dict]) -> Optional[str]:
    if not results:
        return None

    q = query.lower()
    top = results[0]
    name = str(top.get("name", "")).strip()
    price = top.get("price")
    unit = str(top.get("unit", "")).strip()
    brand = str(top.get("brand", "")).strip()
    category = str(top.get("category", "")).strip()

    if _has_any(q, ["দাম", "কত টাকা", "price"]):
        if price is not None:
            unit_part = f" প্রতি {unit}" if unit else ""
            return f"{name} এর দাম {price} টাকা{unit_part}।"
        return "দুঃখিত, এই পণ্যের দাম তথ্য নেই।"

    if _has_any(q, ["বিক্রি", "আছে", "পাওয়া", "উপলব্ধ"]):
        return f"হ্যাঁ, আমাদের ক্যাটালগে {name} আছে।"

    if _has_any(q, ["ব্র্যান্ড", "brand"]):
        if brand:
            return f"{name} এর ব্র্যান্ড {brand}।"
        return "দুঃখিত, এই পণ্যের ব্র্যান্ড তথ্য নেই।"

    if _has_any(q, ["ক্যাটাগরি", "category"]):
        if category:
            return f"{name} এর ক্যাটাগরি {category}।"
        return "দুঃখিত, এই পণ্যের ক্যাটাগরি তথ্য নেই।"

    if _has_any(q, ["কেজি", "গ্রাম", "প্যাকেট", "সাইজ", "unit"]):
        if unit:
            return f"{name} এর ইউনিট {unit}।"
        return "দুঃখিত, এই পণ্যের ইউনিট তথ্য নেই।"

    # Default: short list of matches
    items = []
    for item in results[:3]:
        n = str(item.get("name", "")).strip()
        p = item.get("price")
        u = str(item.get("unit", "")).strip()
        if not n:
            continue
        if p is not None:
            unit_part = f"/{u}" if u else ""
            items.append(f"{n} - {p} টাকা{unit_part}")
        else:
            items.append(n)

    if items:
        return "পাওয়া গেছে: " + ", ".join(items) + "।"
    return "দুঃখিত, এই তথ্য আমার কাছে নেই।"


def _has_any(q: str, needles: List[str]) -> bool:
    return any(n in q for n in needles)
