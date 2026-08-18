import asyncio
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from src.config import settings


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
        async with self._session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        ) as resp:
            resp.raise_for_status()
            data: dict[str, Any] = await resp.json()
            return list(data["embedding"])

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]
