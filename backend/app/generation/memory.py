"""Redis-backed conversation memory for multi-turn chat."""

from __future__ import annotations

import json

import redis.asyncio as aioredis

from app.core.config import settings


class ConversationMemory:
    """Stores and retrieves per-conversation message history in Redis.

    Messages are kept in a Redis list (RPUSH / LRANGE) where each element
    is a JSON-encoded ``{"role": str, "content": str}`` object.  At most the
    last 100 messages are retained per conversation.
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    def _key(self, conversation_id: str) -> str:
        return f"conversation:{conversation_id}:messages"

    async def get_history(self, conversation_id: str, last_n: int = 6) -> list[dict]:
        """Return the last *last_n* messages for *conversation_id*.

        Returns an empty list if the conversation does not exist.
        """
        redis = await self._get_redis()
        key = self._key(conversation_id)
        # LRANGE with negative indices: -last_n gives the last N elements
        raw_messages = await redis.lrange(key, -last_n, -1)
        return [json.loads(m) for m in raw_messages]

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Trims the list to the last 100 messages after appending.
        """
        redis = await self._get_redis()
        key = self._key(conversation_id)
        await redis.rpush(key, json.dumps({"role": role, "content": content}))
        # Keep only the last 100 messages
        await redis.ltrim(key, -100, -1)

    async def clear(self, conversation_id: str) -> None:
        """Delete the entire history for *conversation_id*."""
        redis = await self._get_redis()
        await redis.delete(self._key(conversation_id))


# Module-level singleton used by route handlers
conversation_memory = ConversationMemory()
