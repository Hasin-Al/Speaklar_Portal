import os
from typing import List

from groq import Groq

from .session import Session


class Generator:
    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it in .env")
        timeout_s = float(os.getenv("GROQ_TIMEOUT", "20"))
        self._client = Groq(api_key=api_key, timeout=timeout_s)
        self._model = model

    def generate(
        self,
        query: str,
        expanded_query: str,
        results: List[dict],
        session: Session,
        availability: str | None = None,
    ) -> str:
        context_lines = []
        for i, item in enumerate(results, start=1):
            name = str(item.get("name", ""))
            price = item.get("price")
            unit = item.get("unit", "")
            brand = item.get("brand", "")
            category = item.get("category", "")
            text = str(item.get("text", "")).strip()
            description = str(item.get("description", "")).strip()

            price_str = ""
            if price is not None:
                price_str = f"{price} টাকা"

            parts = [
                f"নাম: {name}",
                f"দাম: {price_str}" if price_str else "দাম: উল্লেখ নেই",
            ]
            if unit:
                parts.append(f"ইউনিট: {unit}")
            if brand:
                parts.append(f"ব্র্যান্ড: {brand}")
            if category:
                parts.append(f"ক্যাটাগরি: {category}")
            if text:
                parts.append(f"তথ্য: {text}")
            elif description:
                parts.append(f"তথ্য: {description}")

            context_lines.append(f"{i}. " + "; ".join(parts))

        context_block = "\n".join(context_lines) if context_lines else "(কোনো প্রাসঙ্গিক পণ্য পাওয়া যায়নি)"

        system_prompt = (
            "তুমি একটি বাংলাভাষী ই-কমার্স সহকারী। "
            "শুধুমাত্র দেওয়া কন্টেক্সট থেকে উত্তর দেবে। "
            "যদি তথ্য না থাকে, বলবে: 'দুঃখিত, এই তথ্য আমার কাছে নেই।'"
        )
        if availability is not None:
            system_prompt += (
                " যদি প্রশ্নটি পণ্য বিক্রি/উপলব্ধতা সম্পর্কে হয়, "
                "তবে availability ফিল্ড অনুযায়ী উত্তর দেবে: "
                "availability=হ্যাঁ হলে 'হ্যাঁ, আমরা বিক্রি করি' বলবে, "
                "availability=না হলে 'না, এই পণ্য নেই' বলবে।"
            )

        user_prompt = (
            f"প্রশ্ন: {query}\n"
            f"সমৃদ্ধ প্রশ্ন: {expanded_query}\n\n"
        )
        if availability is not None:
            user_prompt += f"availability: {availability}\n"
        user_prompt += (
            f"কন্টেক্সট:\n{context_block}\n\n"
            "সংক্ষিপ্ত ও নির্ভুল উত্তর দাও।"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session.recent_history())
        messages.append({"role": "user", "content": user_prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )

        return resp.choices[0].message.content.strip()
