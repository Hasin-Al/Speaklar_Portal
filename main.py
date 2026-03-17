import os
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
from rag.session import SessionStore
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


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    top_k: int = 5


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LogoutRequest(BaseModel):
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

    start = time.perf_counter()
    expanded_query, entity, anaphor = resolver.resolve(req.query, session)
    subject = entity or (session.last_subject if anaphor else None)
    resolver_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    results = indexer.search(expanded_query, top_k=req.top_k)
    if subject:
        exact = [item for item in results if str(item.get("name", "")).strip() == subject]
        if not exact:
            exact = indexer.find_by_name(subject)
        if exact:
            results = exact
    retrieval_ms = (time.perf_counter() - start) * 1000
    session.last_retrieved = results

    start = time.perf_counter()
    fast_answer = None
    if indexer.is_product_kb and not FORCE_LLM:
        fast_answer = try_fast_answer(expanded_query, results)
    llm_error = None
    product_mode = indexer.is_product_kb
    availability = ("হ্যাঁ" if results else "না") if product_mode else None

    if FAST_ONLY:
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
                req.query, expanded_query, results, session, availability=availability
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
        "entity": entity,
        "anaphor": anaphor,
        "timings": timings,
        "retrieved": results,
        "llm_error": llm_error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
