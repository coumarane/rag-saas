# CLAUDE.md — RAG SaaS MVP (Phase 1)

## 🧠 Project Overview
This is a production-grade RAG (Retrieval-Augmented Generation) SaaS MVP.
Users upload PDF/DOCX documents and chat with them using AI.
Answers are grounded in document content with source citations.

## 🏗️ Monorepo Structure
```
rag-saas/
├── CLAUDE.md                  ← you are here
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── documents.py
│   │   │   │   ├── chat.py
│   │   │   │   └── health.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── dependencies.py
│   │   ├── ingestion/
│   │   │   ├── parser.py
│   │   │   ├── chunker.py
│   │   │   └── embedder.py
│   │   ├── retrieval/
│   │   │   ├── searcher.py
│   │   │   ├── reranker.py
│   │   │   └── fusion.py
│   │   ├── generation/
│   │   │   ├── llm.py
│   │   │   ├── prompts.py
│   │   │   └── memory.py
│   │   ├── models/
│   │   │   ├── database.py
│   │   │   ├── document.py
│   │   │   └── chunk.py
│   │   └── workers/
│   │       └── tasks.py
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── conftest.py
│   ├── alembic/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── alembic.ini
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── upload/page.tsx
│   │   ├── documents/page.tsx
│   │   └── chat/[docId]/page.tsx
│   ├── components/
│   │   ├── ui/               ← shadcn components
│   │   ├── ChatWindow.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── CitationPanel.tsx
│   │   ├── DocumentUpload.tsx
│   │   └── DocumentList.tsx
│   ├── hooks/
│   │   ├── useChat.ts
│   │   └── useDocuments.ts
│   ├── lib/
│   │   └── api.ts
│   ├── Dockerfile
│   ├── package.json
│   └── next.config.ts
└── infra/
    └── nginx/
        └── nginx.conf
```

---

## ⚙️ Tech Stack (non-negotiable)

### Backend
- **Python 3.11+**
- **FastAPI** — async API framework
- **Celery + Redis** — background workers for ingestion
- **SQLAlchemy 2.0 (async)** + **Alembic** — ORM and migrations
- **asyncpg** — async PostgreSQL driver
- **Qdrant** — vector database (dense + sparse hybrid)
- **OpenAI SDK** — text-embedding-3-large (1536 dims)
- **Anthropic SDK** — claude-sonnet-4-20250514 for generation
- **pypdf + pdfplumber** — PDF parsing
- **python-docx + mammoth** — DOCX parsing
- **unstructured** — fallback parser
- **sentence-transformers** — local CrossEncoder reranker
- **structlog** — structured logging
- **pydantic-settings** — config management
- **ruff** — linting/formatting
- **pytest + pytest-asyncio + httpx** — testing

### Frontend
- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **shadcn/ui** — component library
- **Vercel AI SDK** — streaming chat
- **TanStack Query** — server state
- **react-dropzone** — file upload
- **axios** — HTTP client

### Infrastructure (local dev)
- **Docker Compose** — orchestrates all services
- **PostgreSQL 15** — metadata DB
- **Redis 7** — Celery broker + cache
- **Qdrant** — vector store
- **MinIO** — local S3-compatible storage

---

## 🗄️ Database Schema

### PostgreSQL Tables
```sql
-- documents
id UUID PRIMARY KEY DEFAULT gen_random_uuid()
tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'
file_name TEXT NOT NULL
file_type TEXT NOT NULL  -- 'pdf' | 'docx'
s3_key TEXT NOT NULL
status TEXT NOT NULL DEFAULT 'pending'  -- pending|processing|ready|failed
error_message TEXT
chunk_count INTEGER
token_count INTEGER
created_at TIMESTAMPTZ DEFAULT now()
updated_at TIMESTAMPTZ DEFAULT now()

-- chunks
id UUID PRIMARY KEY DEFAULT gen_random_uuid()
doc_id UUID REFERENCES documents(id) ON DELETE CASCADE
tenant_id UUID NOT NULL
text TEXT NOT NULL
page_number INTEGER
section_title TEXT
chunk_index INTEGER NOT NULL
token_count INTEGER
created_at TIMESTAMPTZ DEFAULT now()

-- conversations
id UUID PRIMARY KEY DEFAULT gen_random_uuid()
tenant_id UUID NOT NULL
doc_id UUID REFERENCES documents(id) ON DELETE SET NULL
created_at TIMESTAMPTZ DEFAULT now()

-- messages
id UUID PRIMARY KEY DEFAULT gen_random_uuid()
conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE
role TEXT NOT NULL  -- 'user' | 'assistant'
content TEXT NOT NULL
citations JSONB
created_at TIMESTAMPTZ DEFAULT now()
```

---

## 🔄 Core Flows

### Ingestion Flow
```
POST /api/documents/upload
  → validate file (type, size < 50MB)
  → upload to MinIO (S3)
  → create document record (status=pending)
  → trigger Celery task: process_document(doc_id)
  → return {doc_id, status: "pending"}

Celery task:
  → update status=processing
  → parse file (PDF or DOCX)
  → clean text
  → chunk (structural → recursive → semantic boundary)
  → embed (batch 100, text-embedding-3-large, 1536 dims)
  → upsert to Qdrant (dense + sparse BM25 vectors)
  → save chunks to PostgreSQL
  → update status=ready
  → on error: status=failed, save error_message
```

### Chat / Retrieval Flow
```
POST /api/chat/stream
  → validate request {question, doc_id, conversation_id?}
  → expand query (generate 3 variants)
  → dense search Qdrant top-20
  → sparse BM25 search Qdrant top-20
  → RRF fusion → merged top-20
  → CrossEncoder rerank → top-6
  → build prompt (system + context + history + question)
  → stream claude-sonnet-4-20250514
  → parse citations from structured response
  → save message to PostgreSQL
  → return SSE stream
```

---

## 📐 Chunking Rules
- Target: 512 tokens per chunk
- Overlap: 64 tokens (~12%)
- Separators (priority order): ["\n\n", "\n", ". ", " "]
- Never split mid-sentence if avoidable
- Each chunk MUST have metadata: doc_id, page_number, section_title, chunk_index

---

## 🔍 Retrieval Config
```python
TOP_K_INITIAL = 20       # per search type (dense + sparse)
TOP_K_RERANKED = 6       # final chunks sent to LLM
RRF_K = 60               # RRF constant
SCORE_THRESHOLD = 0.35   # minimum similarity score
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 1536
```

---

## 🤖 LLM Config
```python
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048
TEMPERATURE = 0.1        # low temp for grounded answers
STREAM = True
CONTEXT_WINDOW = 6       # last N turns for memory
```

---

## 📝 Prompt Template
```
SYSTEM:
You are a precise document assistant. Answer questions ONLY using the provided context chunks.
Rules:
- Always cite your sources using [Source N] format
- If the answer is not in the context, say: "I couldn't find this information in the provided documents."
- Be concise and factual
- Never hallucinate or add information not present in the context

CONTEXT:
{for each chunk}
[Source {N} | {file_name} | Page {page_number} | Section: {section_title}]
{chunk_text}
{end for}

CONVERSATION HISTORY:
{last 6 turns}

USER QUESTION:
{question}

Respond in this JSON format:
{
  "answer": "your answer with [Source N] inline citations",
  "citations": [{"source_n": 1, "chunk_id": "...", "doc_name": "...", "page": N, "excerpt": "..."}],
  "out_of_context": false
}
```

---

## 🌍 Environment Variables
```env
# App
APP_ENV=development
SECRET_KEY=changeme

# Database
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/ragdb

# Redis
REDIS_URL=redis://localhost:6379/0

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_chunks

# MinIO / S3
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=rag-documents

# AI
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## ✅ Definition of Done (Phase 1)

Every feature is complete when:
1. Code is written and works
2. Unit test exists and passes
3. Integration test covers the happy path
4. Error cases handled gracefully (no 500s without proper error response)
5. Endpoint documented (FastAPI auto-docs sufficient)
6. Docker Compose boots it without manual steps

---

## 🚫 What NOT to build in Phase 1
- No authentication / multi-tenancy (use hardcoded tenant_id)
- No billing
- No connectors (Google Drive, Notion etc.)
- No custom model fine-tuning
- No advanced analytics dashboard
- Keep it working, tested, and clean

---

## 📏 Code Style Rules
- All Python: async/await (no sync blocking calls)
- All API responses: use Pydantic response models
- No raw SQL strings: use SQLAlchemy ORM
- All secrets: from environment variables only, never hardcoded
- Error handling: custom exception classes in core/exceptions.py
- Logging: structlog everywhere, no print() statements