import asyncio
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from src.config import settings


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str: ...

    @abstractmethod
    async def generate_batch(self, prompts: list[str]) -> list[str]: ...


class OllamaLLMProvider(LLMProvider):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        concurrency: int | None = None,
        timeout: int | None = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.chat_model
        self.semaphore = asyncio.Semaphore(concurrency or settings.gen_concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout or settings.request_timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate_batch(self, prompts: list[str]) -> list[str]:
        await self._ensure_session()
        tasks = [self._generate_one(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def _generate_one(self, prompt: str) -> str:
        await self._ensure_session()
        assert self._session is not None
        async with self.semaphore:
            async with self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            ) as resp:
                resp.raise_for_status()
                data: dict[str, Any] = await resp.json()
                return str(data["message"]["content"]).strip()

    async def generate(self, prompt: str) -> str:
        return (await self.generate_batch([prompt]))[0]
