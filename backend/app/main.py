from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, documents, health
from app.core.config import settings
from app.core.exceptions import RAGBaseException
from app.core.logging import configure_logging, get_logger
from app.core.storage import ensure_bucket_exists
from app.core.vector_store import AsyncQdrantClient, ensure_collection_exists, get_qdrant_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info(
        "RAG SaaS API starting",
        environment=settings.app_env,
        version="0.1.0",
    )

    # --- Storage ---
    await ensure_bucket_exists()

    # --- Vector store ---
    qdrant_client: AsyncQdrantClient = await get_qdrant_client()
    await ensure_collection_exists(qdrant_client)
    app.state.qdrant_client = qdrant_client

    yield

    # --- Shutdown ---
    logger.info("RAG SaaS API shutting down")
    await qdrant_client.close()


app = FastAPI(
    title="RAG SaaS API",
    version="0.1.0",
    description="Production-grade Retrieval-Augmented Generation SaaS API",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
origins = ["*"] if settings.app_env == "development" else [settings.next_public_api_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RAGBaseException)
async def rag_exception_handler(request: Request, exc: RAGBaseException) -> JSONResponse:
    logger.warning(
        "Application error",
        status_code=exc.status_code,
        detail=exc.message,
        path=str(request.url),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "RAG SaaS API"}
