import asyncio
from pathlib import Path

import typer

from src.config import settings
from src.pipeline.async_orchestrator import AsyncPipelineOrchestrator
from src.pipeline.orchestrator import SyncPipelineOrchestrator

app = typer.Typer(
    name="somaia",
    help="RAG de questões matemáticas com Ollama local",
    add_completion=False,
)


def _apply_settings(
    db_path: Path | None,
    top_k: int | None,
    model: str | None,
    embed_model: str | None,
) -> None:
    if db_path:
        settings.db_path = str(db_path)
    if top_k:
        settings.top_k = top_k
    if model:
        settings.chat_model = model
    if embed_model:
        settings.embed_model = embed_model


@app.command()
def run(
    csv_path: Path = typer.Argument(..., help="Caminho do arquivo CSV de entrada"),
    db_path: Path = typer.Option(None, "--db", "-d", help="Caminho do banco SQLite"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Número de exemplos similares"),
    model: str = typer.Option(None, "--model", "-m", help="Modelo de chat do Ollama"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Modelo de embeddings"),
) -> None:
    """Executa o pipeline RAG síncrono no CSV."""
    _apply_settings(db_path, top_k, model, embed_model)

    if not csv_path.exists():
        print(f"  Arquivo não encontrado: {csv_path}")
        raise typer.Exit(1)

    orchestrator = SyncPipelineOrchestrator()
    orchestrator.run(csv_path)


@app.command()
def run_async(
    csv_path: Path = typer.Argument(..., help="Caminho do arquivo CSV de entrada"),
    db_path: Path = typer.Option(None, "--db", "-d", help="Caminho do banco SQLite"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Número de exemplos similares"),
    model: str = typer.Option(None, "--model", "-m", help="Modelo de chat do Ollama"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Modelo de embeddings"),
    embed_concurrency: int = typer.Option(
        None, "--embed-concurrency", help="Concorrência para embeddings"
    ),
    gen_concurrency: int = typer.Option(
        None, "--gen-concurrency", help="Concorrência para geração"
    ),
) -> None:
    """Executa o pipeline RAG assíncrono no CSV (mais rápido)."""
    _apply_settings(db_path, top_k, model, embed_model)

    if embed_concurrency:
        settings.embed_concurrency = embed_concurrency
    if gen_concurrency:
        settings.gen_concurrency = gen_concurrency

    if not csv_path.exists():
        print(f"  Arquivo não encontrado: {csv_path}")
        raise typer.Exit(1)

    orchestrator = AsyncPipelineOrchestrator()
    asyncio.run(orchestrator.run(csv_path))


@app.command()
def evaluate(
    csv_path: Path = typer.Argument(..., help="CSV respondido para avaliar"),
) -> None:
    """Avalia a acurácia das respostas."""
    import subprocess
    subprocess.run(["python", "acerto.py"], check=False)


if __name__ == "__main__":
    app()
