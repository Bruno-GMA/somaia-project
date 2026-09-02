import asyncio
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from src.config import settings
from src.utils.retry import async_retry_with_backoff


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        concurrency: int | None = None,
        timeout: int | None = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.embed_model
        self.semaphore = asyncio.Semaphore(concurrency or settings.embed_concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout or 60)
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def _reset_session(self) -> None:
        """Force recreate session (e.g., after connection errors)."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        await self._ensure_session()
        async with self.semaphore:
            tasks = [self._embed_one(text) for text in texts]
            return await asyncio.gather(*tasks)

    async def _embed_one(self, text: str) -> list[float]:
        assert self._session is not None
        session = self._session

        async def _do_embed() -> list[float]:
            async with self.semaphore:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                ) as resp:
                    resp.raise_for_status()
                    data: dict[str, Any] = await resp.json()
                    return list(data["embedding"])

        try:
            result = await async_retry_with_backoff(
                _do_embed,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                exceptions=(aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError),
            )
            return result  # type: ignore
        except (TimeoutError, aiohttp.ClientError):
            # Reset session on connection errors for subsequent requests
            await self._reset_session()
            raise

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]
