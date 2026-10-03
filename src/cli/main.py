import asyncio
import json
from datetime import UTC
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
    no_rag: bool,
) -> None:
    if db_path:
        settings.db_path = str(db_path)
    if top_k:
        settings.top_k = top_k
    if model:
        settings.chat_model = model
    if embed_model:
        settings.embed_model = embed_model
    settings.use_rag = not no_rag


@app.command()
def run(
    csv_path: Path = typer.Argument(..., help="Caminho do arquivo CSV de entrada"),
    db_path: Path = typer.Option(None, "--db", "-d", help="Caminho do banco SQLite"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Número de exemplos similares"),
    model: str = typer.Option(None, "--model", "-m", help="Modelo de chat do Ollama"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Modelo de embeddings"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Desativa retrieval (gera sem exemplos)"),
) -> None:
    """Executa o pipeline RAG síncrono no CSV."""
    _apply_settings(db_path, top_k, model, embed_model, no_rag)

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
    no_rag: bool = typer.Option(False, "--no-rag", help="Desativa retrieval (gera sem exemplos)"),
) -> None:
    """Executa o pipeline RAG assíncrono no CSV (mais rápido)."""
    _apply_settings(db_path, top_k, model, embed_model, no_rag)

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


@app.command()
def evaluate_retrieval(
    db_path: Path = typer.Argument(..., help="Caminho do arquivo SQLite"),
    top_k: int = typer.Option(None, "--top-k", "-k", min=1),
    embed_model: str = typer.Option(None, "--embed-model", "-e"),
    output: Path = typer.Option(None, "--output", "-o", help="JSON novo; não sobrescreve"),
    qrels_path: Path = typer.Option(None, "--qrels", help="Julgamentos JSON versionados"),
) -> None:
    """Avalia recuperação usando o corpus completo ou qrels explícitos (schema v2)."""
    from datetime import datetime

    from src.db.connection import ConnectionManager
    from src.db.repository import QuestionRepository
    from src.embedding.provider import OllamaEmbeddingProvider
    from src.evaluation.ranking import Qrels, validate_k
    from src.evaluation.retrieval_metrics import aggregate, evaluate_retrieval
    from src.retrieval.hybrid import HybridRetriever

    if not db_path.is_file():
        raise typer.BadParameter(f"Banco inexistente: {db_path}")
    k = top_k if top_k is not None else settings.top_k
    validate_k(k)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = output or Path(f"retrieval_metrics_v2_{stamp}.json")
    if out_path.exists():
        raise typer.BadParameter(f"Saída já existe: {out_path}")
    qrels = Qrels.from_json(qrels_path.read_text(encoding="utf-8")) if qrels_path else None
    settings.db_path = str(db_path)
    if embed_model:
        settings.embed_model = embed_model

    async def collect() -> list[dict[str, object]]:
        cm = ConnectionManager(settings.db_path)
        embedder = OllamaEmbeddingProvider()
        try:
            corpus = QuestionRepository(cm).get_retrieval_corpus()
            if not corpus:
                raise typer.BadParameter("Nenhuma questão elegível no banco")
            if qrels:
                qrels.validate_corpus([str(q.id) for q in corpus])
            retriever = HybridRetriever(cm, embedder)
            per_query = []
            for q in corpus:
                similar = await retriever.retrieve(q.text, k, exclude_id=q.id)
                result = evaluate_retrieval(q, similar, k, corpus=corpus, qrels=qrels)
                result["query_text"] = q.text
                per_query.append(result)
            return per_query
        finally:
            try:
                await embedder.close()
            finally:
                cm.close_all()

    per_query = asyncio.run(collect())
    metrics = aggregate(per_query)
    payload = {
        **metrics.payload,
        "top_k": k,
        "embed_model": settings.embed_model,
        "legacy_compatible": False,
        "per_query": per_query,
    }
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(metrics.summary())
    print(f"Métricas salvas em: {out_path}")


if __name__ == "__main__":
    app()
