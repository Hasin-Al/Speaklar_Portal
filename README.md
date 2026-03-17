# Speaklar RAG Portal

A lightweight Bangla RAG demo with:
- **TF‑IDF + cosine similarity** retrieval (fast, in‑memory)
- **Coreference resolution** for follow‑up questions
- **Groq LLM** for grounded answers
- **Two KB formats**: product‑price lines or paragraph knowledge bank
- **SQLite** for persistent users and sessions

---

## 1) Project Structure

```
.
├── main.py                 # FastAPI server + request flow
├── rag/
│   ├── auth.py             # User register/login
│   ├── fast_answer.py       # Optional rule-based fast answers (product KB)
│   ├── generator.py         # Groq LLM prompt + response
│   ├── indexer.py           # TF‑IDF + cosine retrieval + KB parsing
│   ├── resolver.py          # Coreference resolver (last product)
│   ├── session.py           # Session state wrapper
│   └── storage.py           # SQLite persistence
├── static/
│   ├── login.html           # Bootstrap login page
│   ├── register.html        # Bootstrap registration page
│   └── chat.html            # Bootstrap chat page
├── data/
│   ├── products.txt         # Product+price KB (one per line)
│   ├── Knowledge_Bank.txt   # Paragraph KB (one paragraph per line)
│   └── app.db               # SQLite DB (auto‑created)
├── requirements.txt
└── README.md
```

---

## 2) How Retrieval Works (TF‑IDF + Cosine)

**Core idea:** Convert each KB entry into tokens, build an inverted index, and score with cosine similarity.

**Step‑by‑step retrieval:**
1. **Coreference resolution** (`resolver.resolve`) expands follow‑ups, e.g.
   - Q1: “নুডুলস বিক্রি করে?” → subject = “নুডুলস”
   - Q2: “দাম কত টাকা?” → expanded = “নুডুলস দাম কত টাকা?”
2. **TF‑IDF scoring** (`indexer.search`)
   - Tokenize text → term frequencies (TF)
   - Compute inverse document frequency (IDF)
   - Cosine similarity between query vector and document vectors
3. **Substring boost**
   - If query contains product name (Bangla morphology), add a small boost
4. **Exact match filter** (product KB only)
   - If a subject is known, keep only exact name matches to avoid multiple prices

---

## 3) Knowledge Base Formats

The system **auto‑detects KB type** based on how many entries contain `price`.

### A) Product KB (name + price)
`data/products.txt`

**Format:**
```
নুডুলস 20
চাল 75
চিনি 110
```

- Treated as **product mode**.
- Availability rules are enabled.
- Exact name match avoids multiple prices.

### B) Paragraph KB (knowledge bank)
`data/Knowledge_Bank.txt`

**Format:** one paragraph per line:
```
ফাস্ট চার্জার। দ্রুত চার্জ করে। ১০০% অরিজিনাল পণ্য...
```

- Treated as **paragraph mode**.
- The full paragraph is passed to the LLM as context.

**Select KB with `.env`:**
```
KB_PATH=data/Knowledge_Bank.txt
```

---

## 4) LLM Answering (Groq)

The LLM only answers using retrieved context.

- Product KB: includes availability logic ("হ্যাঁ, আমরা বিক্রি করি" if found)
- Paragraph KB: context only, no availability override

Model (default):
```
llama-3.3-70b-versatile
```

---

## 5) Running the Project

### 5.1 Install dependencies
```
pip install -r requirements.txt
```

### 5.2 Add Groq API key
Create `.env`:
```
GROQ_API_KEY=your_key_here
```

Optional settings:
```
KB_PATH=data/Knowledge_Bank.txt
FORCE_LLM=true
FAST_ONLY=false
GROQ_TIMEOUT=20
```

### 5.3 Start server
```
python main.py
```

Open the UI:
- Login: `http://localhost:8000/login`
- Register: `http://localhost:8000/register`
- Chat: `http://localhost:8000/chat`

---

## 6) Example Workflow

### Register + login
```
POST /register  {"username":"hasin","password":"secret123"}
POST /login     {"username":"hasin","password":"secret123"}
```

### Chat with coreference
Q1: “আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?”
Q2: “দাম কত টাকা?”

The resolver expands Q2 → “নুডুলস দাম কত টাকা?”

---

## 7) Timings

The response includes a timing breakdown:
- `resolver_ms`
- `retrieval_ms`
- `retrieval_total_ms`
- `generation_ms`
- `total_ms`
- `retrieval_under_100ms`

Retrieval is expected to stay under 100 ms on ~5,000 items.

---

## 8) Notes & Production Considerations

- Users & sessions are persisted in SQLite (`data/app.db`)
- For production:
  - Use Redis for sessions
  - Use Postgres for users
  - Add HTTPS, rate limiting, and monitoring

---

## 9) API Endpoints

- `POST /register`
- `POST /login`
- `POST /logout`
- `POST /chat`
- `GET /session/{session_id}`
- `GET /health`

---

## 10) Troubleshooting

- **No answer from LLM:** check `GROQ_API_KEY` and `GROQ_TIMEOUT`
- **Wrong KB:** set `KB_PATH` in `.env`
- **Multiple prices:** ensure product KB uses exact names or enable exact match filter

