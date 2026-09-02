from typing import Any

import requests

from src.config import settings
from src.utils.retry import retry_with_backoff


def ollama_embed_sync(text: str) -> list[float]:
    def _embed() -> list[float]:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Invalid embedding response")
        return [float(x) for x in embedding]

    return retry_with_backoff(
        _embed,
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        exceptions=(requests.RequestException, ValueError, KeyError),
    )


def ollama_chat_sync(prompt: str) -> str:
    def _chat() -> str:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=settings.request_timeout,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        message = data.get("message")
        if not isinstance(message, dict):
            raise ValueError("Invalid chat response")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Invalid chat response content")
        return content.strip()

    return retry_with_backoff(
        _chat,
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
        exceptions=(requests.RequestException, ValueError, KeyError),
    )


def check_ollama_sync() -> None:
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception:
        print("  Ollama não está acessível em", settings.ollama_base_url)
        print("    Inicie com: ollama serve")
        raise SystemExit(1)

    available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    for model in (settings.chat_model, settings.embed_model):
        base = model.split(":")[0]
        if base not in available:
            print(f"  Modelo '{model}' não encontrado. Baixando…")
            requests.post(
                f"{settings.ollama_base_url}/api/pull",
                json={"name": model},
                timeout=600,
            )
            print(f"  '{model}' baixado.")
