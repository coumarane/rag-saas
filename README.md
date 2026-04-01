# RAG SaaS MVP

Production-grade document Q&A system. Upload PDFs/DOCX, chat with them using AI.

## Quick Start

1. Clone the repo
2. Copy environment file: `cp .env.example .env`
3. Add your API keys to `.env`:
   - OPENAI_API_KEY=sk-...
   - ANTHROPIC_API_KEY=sk-ant-...
4. Start all services: `make dev`
5. Run database migrations: `make migrate`
6. Open http://localhost:3000

## Architecture
- Frontend: Next.js 14 on :3000
- API: FastAPI on :8000
- Worker: Celery (background ingestion)
- Flower: Celery monitor on :5555
- PostgreSQL: :5432
- Redis: :6379
- Qdrant: :6333
- MinIO: :9000 (API), :9001 (console)
- Nginx: :80 (reverse proxy)

## Development
- `make dev` — start all services
- `make migrate` — run DB migrations
- `make test` — run pytest
- `make logs` — tail all logs
- `make stop` — stop services
- `make clean` — stop + delete volumes

## Tech Stack
FastAPI · Celery · PostgreSQL · Redis · Qdrant · MinIO · Next.js 14 · shadcn/ui
