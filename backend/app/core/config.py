from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    secret_key: str = "changeme"

    # Database
    database_url: str = "postgresql+asyncpg://rag:rag@postgres:5432/ragdb"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rag_chunks"

    # MinIO / S3
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "rag-documents"

    # AI
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Frontend
    next_public_api_url: str = "http://localhost:8000"

    # Retrieval config
    top_k_initial: int = 20
    top_k_reranked: int = 6
    rrf_k: int = 60
    score_threshold: float = 0.35

    # Models
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 1536

    # LLM
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1
    context_window: int = 6

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
