import requests

from src.config import settings


def ollama_embed_sync(text: str) -> list[float]:
    resp = requests.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data: dict[str, object] = resp.json()
    return list(data["embedding"])  # type: ignore[no-any-return, call-overload]


def ollama_chat_sync(prompt: str) -> str:
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
    data: dict[str, object] = resp.json()
    return str(data["message"]["content"]).strip()  # type: ignore[index]


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
