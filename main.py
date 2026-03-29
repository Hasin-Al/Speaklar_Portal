import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from rag.auth import UserStore
from rag.fast_answer import try_fast_answer
from rag.generator import Generator
from rag.indexer import Indexer
from rag.resolver import Resolver
from rag.session import SessionStore, Session
from rag.storage import Storage

load_dotenv()

app = FastAPI()

kb_path = os.getenv("KB_PATH")
if not kb_path:
    kb_path = "data/Knowledge_Bank.txt" if os.path.exists("data/Knowledge_Bank.txt") else "data/products.txt"
indexer = Indexer.from_file(kb_path)
resolver = Resolver(indexer)
storage = Storage("data/app.db")
store = SessionStore(storage)
generator = Generator()
users = UserStore(storage)
FAST_ONLY = os.getenv("FAST_ONLY", "false").lower() in {"1", "true", "yes"}
FORCE_LLM = os.getenv("FORCE_LLM", "true").lower() in {"1", "true", "yes"}


def _is_price_question(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in ["দাম", "কত টাকা", "price", "মূল্য"])


def _is_availability_question(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in ["বিক্রি", "আছে", "পাওয়া", "পাওয়া", "উপলব্ধ"])


def _question_intent(text: str) -> str | None:
    lowered = text.lower()
    if _is_price_question(lowered):
        return "price"
    if _is_availability_question(lowered):
        return "availability"
    if any(k in lowered for k in ["ব্র্যান্ড", "brand"]):
        return "brand"
    if any(k in lowered for k in ["ক্যাটাগরি", "category"]):
        return "category"
    if any(k in lowered for k in ["ইউনিট", "সাইজ", "unit"]):
        return "unit"
    if any(k in lowered for k in ["ডেলিভারি", "delivery"]):
        return "delivery"
    if any(k in lowered for k in ["ওয়ারেন্টি", "ওয়ারেন্টি", "warranty"]):
        return "warranty"
    if any(k in lowered for k in ["রিটার্ন", "return"]):
        return "return"
    if any(k in lowered for k in ["ছাড়", "ডিসকাউন্ট", "discount"]):
        return "discount"
    if any(k in lowered for k in ["সুবিধা", "ফিচার", "feature"]):
        return "feature"
    return None


def _intent_phrase(intent: str) -> str:
    return {
        "price": "দাম কত টাকা",
        "availability": "বিক্রি হয় কি",
        "brand": "ব্র্যান্ড কী",
        "category": "ক্যাটাগরি কী",
        "unit": "ইউনিট কী",
        "delivery": "ডেলিভারি সুবিধা আছে কি",
        "warranty": "ওয়ারেন্টি আছে কি",
        "return": "রিটার্ন পলিসি আছে কি",
        "discount": "ছাড় আছে কি",
        "feature": "সুবিধা কী",
    }.get(intent, "বিস্তারিত কী")


def _options_from_memory(session: Session, limit: int = 3) -> list[str]:
    options = []
    for item in session.subject_memory:
        name = str(item.get("name", "")).strip()
        if name and name not in options:
            options.append(name)
        if len(options) >= limit:
            break
    return options


def _resolve_clarification(query: str, options: list[str]) -> str | None:
    q = query.lower().strip()
    if not q:
        return None
    for opt in options:
        if opt.lower() in q:
            return opt
    # fallback: if user replies with just one token, accept it as subject
    tokens = re.findall(r"[^\W_]+", q)
    tokens = [t for t in tokens if len(t) >= 2]
    if tokens and len(tokens) <= 3:
        return " ".join(tokens)
    return None


def _extract_price_from_results(results: list[dict]) -> str | None:
    if not results:
        return None
    for item in results:
        price = item.get("price")
        if price is not None:
            return f"{price}"
        text = str(item.get("text", "") or item.get("description", "")).strip()
        if not text:
            continue
        m = re.search(r"(?:৳\s*|)\s*([০-৯0-9]+(?:[.,][০-৯0-9]+)?)\s*টাকা", text)
        if m:
            return m.group(1)
        m = re.search(r"৳\s*([০-৯0-9]+(?:[.,][০-৯0-9]+)?)", text)
        if m:
            return m.group(1)
    return None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    top_k: int = 5


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LogoutRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class ClearSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return RedirectResponse(url="/login")


@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.get("/register")
async def register_page():
    return FileResponse("static/register.html")


@app.get("/chat")
async def chat_page():
    return FileResponse("static/chat.html")


@app.post("/register")
async def register(req: AuthRequest):
    try:
        users.register(req.username, req.password)
    except ValueError:
        raise HTTPException(status_code=409, detail="username already exists")
    return {"status": "ok"}


@app.post("/login")
async def login(req: AuthRequest):
    user = users.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    session = store.create_for_user(user.username)
    return {"session_id": session.id}


@app.post("/logout")
async def logout(req: LogoutRequest):
    deleted = store.delete(req.session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"status": "ok"}


@app.post("/session/clear")
async def clear_session(req: ClearSessionRequest):
    cleared = store.clear_context(req.session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="session not found")
    return {"status": "ok"}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = store.get_existing(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "last_subject": session.last_subject,
        "last_retrieved_count": len(session.last_retrieved),
        "history_size": len(session.history),
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    session = store.get_existing(req.session_id)
    if not session:
        raise HTTPException(status_code=401, detail="invalid session_id")

    total_start = time.perf_counter()
    user_query = req.query
    effective_query = req.query

    if session.pending_intent:
        # Clarification disabled: clear any pending state
        session.set_pending(None, [])

    start = time.perf_counter()
    intent = _question_intent(effective_query)
    options = _options_from_memory(session, limit=3)
    explicit_candidate = resolver._extract_candidate_subject(effective_query)
    explicit_entity = indexer.find_explicit_entity(effective_query)
    if explicit_candidate and not explicit_entity:
        session.last_subject = explicit_candidate
        session.last_subject_known = False
        session.remember_subject(explicit_candidate, False)
        answer = f"না, আমরা {explicit_candidate} বিক্রি করি না।"
        session.add_turn(user_query, answer)
        store.save(session)
        return {
            "answer": answer,
            "session_id": session.id,
            "entity": explicit_candidate,
            "subject": explicit_candidate,
            "subject_known": False,
            "anaphor": False,
            "timings": {
                "resolver_ms": 0.0,
                "retrieval_ms": 0.0,
                "retrieval_total_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": 0.0,
                "retrieval_budget_ms": 100.0,
                "retrieval_under_100ms": True,
            },
            "retrieved": [],
            "llm_error": None,
        }
    if intent and not explicit_candidate and len(options) >= 1:
        # No clarification: fall back to most recent subject
        effective_query = f"{options[0]} {_intent_phrase(intent)}"

    expanded_query, subject, anaphor, subject_known = resolver.resolve(effective_query, session)
    resolver_ms = (time.perf_counter() - start) * 1000

    # Guard: if availability/price asked but resolver missed subject, try direct candidate
    if subject is None and (_is_availability_question(effective_query) or _is_price_question(effective_query)):
        candidate = resolver._extract_candidate_subject(effective_query)
        if candidate:
            matched = indexer.find_explicit_entity(candidate)
            if matched:
                subject = matched
                subject_known = True
            else:
                subject = candidate
                subject_known = False

    start = time.perf_counter()
    results = indexer.search(expanded_query, top_k=req.top_k)
    subject_in_kb = bool(subject_known) if subject is not None else False
    if subject and subject_known:
        exact = [item for item in results if str(item.get("name", "")).strip() == subject]
        if not exact:
            exact = indexer.find_by_name(subject)
        if exact:
            results = exact
            subject_in_kb = True
    elif subject and results:
        exact = [item for item in results if str(item.get("name", "")).strip() == subject]
        if exact:
            subject_in_kb = True
    retrieval_ms = (time.perf_counter() - start) * 1000
    session.last_retrieved = results

    start = time.perf_counter()
    llm_error = None
    product_mode = indexer.is_product_kb
    availability = ("হ্যাঁ" if results else "না") if product_mode else None
    answer = None

    # Price questions are answered strictly from KB (no LLM guessing)
    if _is_price_question(effective_query):
        price = None
        if product_mode:
            for item in results:
                p = item.get("price")
                if p is not None:
                    price = p
                    break
        else:
            price = _extract_price_from_results(results)

        if subject and not subject_in_kb:
            answer = f"না, আমরা {subject} বিক্রি করি না।"
        elif subject and subject_in_kb:
            if price is not None:
                answer = f"{subject} এর দাম {price} টাকা।"
            else:
                answer = f"{subject} এর দামের তথ্য আমাদের জ্ঞানভান্ডারে নেই।"
        else:
            if price is not None and results:
                name = str(results[0].get("name", "")).strip()
                if name:
                    answer = f"{name} এর দাম {price} টাকা।"
                else:
                    answer = f"{price} টাকা।"
            else:
                answer = "দামের তথ্য আমাদের জ্ঞানভান্ডারে নেই।"
        generation_ms = 0.0

    fast_answer = None
    if answer is None and indexer.is_product_kb and not FORCE_LLM:
        fast_answer = try_fast_answer(expanded_query, results)

    if subject and not subject_in_kb:
        answer = f"না, আমরা {subject} বিক্রি করি না।"

    if answer is not None:
        generation_ms = 0.0
    elif FAST_ONLY:
        if product_mode:
            answer = fast_answer or "দুঃখিত, এই তথ্য আমার কাছে নেই।"
        else:
            answer = "দুঃখিত, এই জ্ঞানভান্ডারের জন্য ফাস্ট মোড প্রযোজ্য নয়।"
        generation_ms = 0.0
    elif fast_answer:
        answer = fast_answer
        generation_ms = 0.0
    else:
        try:
            answer = generator.generate(
                effective_query, expanded_query, results, session, availability=availability
            )
            generation_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            llm_error = str(exc)
            answer = "দুঃখিত, LLM সার্ভিসে সমস্যা হয়েছে।"
            generation_ms = (time.perf_counter() - start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    session.add_turn(req.query, answer)
    store.save(session)

    retrieval_total_ms = resolver_ms + retrieval_ms
    timings = {
        "resolver_ms": round(resolver_ms, 2),
        "retrieval_ms": round(retrieval_ms, 2),
        "retrieval_total_ms": round(retrieval_total_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "total_ms": round(total_ms, 2),
        "retrieval_budget_ms": 100.0,
        "retrieval_under_100ms": retrieval_total_ms <= 100.0,
    }

    print(
        "[timing] resolver={resolver_ms:.2f}ms retrieval={retrieval_ms:.2f}ms "
        "generation={generation_ms:.2f}ms total={total_ms:.2f}ms".format(**timings)
    )

    return {
        "answer": answer,
        "session_id": session.id,
        "entity": subject,
        "subject": subject,
        "subject_known": subject_known,
        "anaphor": anaphor,
        "timings": timings,
        "retrieved": results,
        "llm_error": llm_error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
