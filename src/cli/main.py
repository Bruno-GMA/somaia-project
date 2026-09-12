import asyncio
import json
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
    no_rag: bool = typer.Option(
        False, "--no-rag", help="Desativa retrieval (gera sem exemplos)"
    ),
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
    no_rag: bool = typer.Option(
        False, "--no-rag", help="Desativa retrieval (gera sem exemplos)"
    ),
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
    db_path: Path = typer.Argument(..., help="Caminho do banco SQLite"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Número de exemplos similares"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Modelo de embeddings"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Caminho para gravar retrieval_metrics.json"
    ),
) -> None:
    """Avalia a qualidade do retrieval (métricas IR clássicas)."""
    from src.db.connection import ConnectionManager
    from src.db.repository import QuestionRepository
    from src.embedding.provider import OllamaEmbeddingProvider
    from src.evaluation.retrieval_metrics import aggregate, evaluate_retrieval
    from src.retrieval.hybrid import HybridRetriever

    if not db_path.exists():
        print(f"  Arquivo não encontrado: {db_path}")
        raise typer.Exit(1)

    settings.db_path = str(db_path)
    if top_k:
        settings.top_k = top_k
    if embed_model:
        settings.embed_model = embed_model

    cm = ConnectionManager(settings.db_path)
    qrepo = QuestionRepository(cm)
    embedder = OllamaEmbeddingProvider()
    retriever = HybridRetriever(cm, embedder)

    answered = [q for q in qrepo.get_all() if q.has_answer]
    if not answered:
        print("Nenhuma questão respondida no banco.")
        raise typer.Exit(1)

    print(f"  Avaliando retrieval de {len(answered)} questões (top_k={settings.top_k})...\n")

    per_query: list[dict[str, object]] = []
    for i, q in enumerate(answered, 1):
        similar = asyncio.run(
            retriever.retrieve(q.text, settings.top_k, exclude_id=q.id)
        )
        res = evaluate_retrieval(q, similar, settings.top_k)
        res["query_id"] = q.id
        res["query_text"] = q.text
        per_query.append(res)
        print(f"  [{i}/{len(answered)}]  {q.text[:60]}…", end="\r")

    print()
    metrics = aggregate(per_query)
    print(metrics.summary())

    out_path = output or Path("retrieval_metrics.json")
    payload: dict[str, object] = {
        "metrics": metrics.to_dict(),
        "n_queries": metrics.n_queries,
        "top_k": settings.top_k,
        "per_query": per_query,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Métricas salvas em: {out_path}")

    asyncio.run(embedder.close())
    cm.close_all()


if __name__ == "__main__":
    app()
