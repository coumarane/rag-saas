from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return a new AsyncQdrantClient connected to the configured Qdrant instance."""
    return AsyncQdrantClient(url=settings.qdrant_url)


async def ensure_collection_exists(client: AsyncQdrantClient) -> None:
    """Create the Qdrant collection with dense + sparse vectors if it does not exist.

    Also ensures payload indexes on ``doc_id`` and ``tenant_id`` are present.
    """
    try:
        exists = await client.collection_exists(settings.qdrant_collection)

        if not exists:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.embedding_dims,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(),
                    )
                },
            )
            logger.info(
                "Qdrant collection created",
                collection=settings.qdrant_collection,
                dims=settings.embedding_dims,
            )
        else:
            logger.info(
                "Qdrant collection already exists",
                collection=settings.qdrant_collection,
            )

        # Ensure payload indexes (idempotent — Qdrant ignores duplicate index creation)
        for field in ("doc_id", "tenant_id"):
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.debug("Payload index ensured", field=field, collection=settings.qdrant_collection)

    except Exception as exc:
        logger.error(
            "Failed to ensure Qdrant collection",
            collection=settings.qdrant_collection,
            exc_info=exc,
        )
        raise VectorStoreError(
            f"Failed to initialise collection '{settings.qdrant_collection}'"
        ) from exc


async def upsert_chunks(
    client: AsyncQdrantClient,
    points: list[dict],
) -> None:
    """Upsert a list of chunk points into Qdrant.

    Each element of ``points`` must contain:
      - ``id``            : str or UUID (point ID)
      - ``dense_vector``  : list[float]
      - ``sparse_vector`` : dict with keys ``indices`` (list[int]) and ``values`` (list[float])
      - ``payload``       : dict (must include at minimum ``doc_id`` and ``tenant_id``)
    """
    if not points:
        return

    try:
        structs = [
            PointStruct(
                id=p["id"],
                vector={
                    "dense": p["dense_vector"],
                    "sparse": SparseVector(
                        indices=p["sparse_vector"]["indices"],
                        values=p["sparse_vector"]["values"],
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ]

        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=structs,
            wait=True,
        )
        logger.info(
            "Chunks upserted to Qdrant",
            collection=settings.qdrant_collection,
            count=len(structs),
        )
    except Exception as exc:
        logger.error(
            "Qdrant upsert failed",
            collection=settings.qdrant_collection,
            count=len(points),
            exc_info=exc,
        )
        raise VectorStoreError("Failed to upsert chunks into Qdrant") from exc


async def search_dense(
    client: AsyncQdrantClient,
    query_vector: list[float],
    doc_id: str,
    top_k: int = 20,
) -> list[dict]:
    """Dense vector search filtered by ``doc_id``.

    Returns a list of dicts with keys ``id``, ``score``, and ``payload``.
    """
    try:
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        results = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using="dense",
            query_filter=doc_filter,
            limit=top_k,
            with_payload=True,
        )

        hits = [
            {"id": str(p.id), "score": p.score, "payload": p.payload}
            for p in results.points
        ]
        logger.debug(
            "Dense search completed",
            doc_id=doc_id,
            top_k=top_k,
            hits=len(hits),
        )
        return hits

    except Exception as exc:
        logger.error(
            "Qdrant dense search failed",
            doc_id=doc_id,
            exc_info=exc,
        )
        raise VectorStoreError("Dense vector search failed") from exc


async def search_sparse(
    client: AsyncQdrantClient,
    sparse_vector: dict,
    doc_id: str,
    top_k: int = 20,
) -> list[dict]:
    """Sparse (BM25) vector search filtered by ``doc_id``.

    ``sparse_vector`` must be a dict with keys ``indices`` (list[int]) and
    ``values`` (list[float]).

    Returns a list of dicts with keys ``id``, ``score``, and ``payload``.
    """
    try:
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        query_sv = SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"],
        )

        results = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_sv,
            using="sparse",
            query_filter=doc_filter,
            limit=top_k,
            with_payload=True,
        )

        hits = [
            {"id": str(p.id), "score": p.score, "payload": p.payload}
            for p in results.points
        ]
        logger.debug(
            "Sparse search completed",
            doc_id=doc_id,
            top_k=top_k,
            hits=len(hits),
        )
        return hits

    except Exception as exc:
        logger.error(
            "Qdrant sparse search failed",
            doc_id=doc_id,
            exc_info=exc,
        )
        raise VectorStoreError("Sparse vector search failed") from exc


async def delete_document_vectors(
    client: AsyncQdrantClient,
    doc_id: str,
) -> None:
    """Delete all Qdrant points whose payload ``doc_id`` matches the given value."""
    try:
        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        await client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=FilterSelector(filter=doc_filter),
            wait=True,
        )
        logger.info(
            "Document vectors deleted from Qdrant",
            doc_id=doc_id,
            collection=settings.qdrant_collection,
        )
    except Exception as exc:
        logger.error(
            "Qdrant vector deletion failed",
            doc_id=doc_id,
            exc_info=exc,
        )
        raise VectorStoreError(f"Failed to delete vectors for doc '{doc_id}'") from exc
