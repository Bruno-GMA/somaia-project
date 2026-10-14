import asyncio
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
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
    rerank: bool = False,
    candidate_k: int | None = None,
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
    settings.rerank_enabled = rerank
    if candidate_k is not None:
        settings.rerank_candidate_k = candidate_k
    if rerank and not no_rag:
        settings.validate_reranking()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_answers(db_path: Path) -> dict[str, dict[str, str | None]]:
    """Load per-question outcomes from one isolated mode database."""
    from src.db.connection import ConnectionManager
    from src.db.repository import QuestionRepository

    cm = ConnectionManager(str(db_path))
    try:
        return {
            str(question.id): {
                "answer": question.answer,
                "ground_truth": question.ground_truth,
                "operation": question.operation,
            }
            for question in QuestionRepository(cm).get_all()
        }
    finally:
        cm.close_all()


def _run_paired(
    csv_path: Path, output_dir: Path, *, use_async: bool, repetitions: int
) -> None:
    """Run RAG and no-RAG in fresh directories and export joinable records.

    The corpus is the supplied CSV; callers should provide a corpus/evaluation
    split that does not expose reference answers for the evaluated questions.
    """
    if output_dir.exists():
        raise typer.BadParameter(f"Diretório de saída já existe: {output_dir}")
    output_dir.mkdir(parents=True)
    run_id = output_dir.name
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "project": "SOMAIA",
        "modes": ["rag", "no-rag"],
        "mode_order": ["rag", "no-rag"],
        "repetitions": repetitions,
        "seed": None,
        "async": use_async,
        "generation": {"model": settings.chat_model, "output_budget": None},
        "embedding_model": settings.embed_model,
        "inputs": {"dataset": {"path": str(csv_path.resolve()), "sha256": _file_hash(csv_path)}},
        "notes": {
            "retrieval_metrics_no_rag": "not_applicable",
            "cache_policy": "no response cache; fresh SQLite database and checkpoint per mode",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    original = {name: getattr(settings, name) for name in ("db_path", "use_rag", "artifact_dir")}
    records: list[dict[str, object]] = []
    try:
        for repetition in range(repetitions):
            for mode in ("rag", "no-rag"):
                mode_dir = output_dir / f"repeat-{repetition + 1}" / mode
                mode_dir.mkdir(parents=True)
                input_copy = mode_dir / csv_path.name
                shutil.copy2(csv_path, input_copy)
                db_path = mode_dir / "state.sqlite"
                settings.db_path = str(db_path)
                settings.use_rag = mode == "rag"
                settings.artifact_dir = str(mode_dir)
                started = datetime.now(UTC).isoformat()
                try:
                    if use_async:
                        asyncio.run(AsyncPipelineOrchestrator().run(input_copy))
                    else:
                        SyncPipelineOrchestrator().run(input_copy)
                    status, error = "ok", None
                except Exception as exc:  # retain the other mode and remaining repetitions
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                for query_id, row in _read_answers(db_path).items() if db_path.exists() else []:
                    records.append(
                        {
                            "run_id": run_id,
                            "query_id": query_id,
                            "project": "SOMAIA",
                            "mode": mode,
                            "repetition": repetition + 1,
                            "started_at": started,
                            "status": status if row["answer"] else "missing_answer",
                            "error": error,
                            "answer": row["answer"],
                            "ground_truth": row["ground_truth"],
                            "operation": row["operation"],
                            "retrieval_metrics": None if mode == "no-rag" else "not_collected",
                        }
                    )
    finally:
        settings.db_path = original["db_path"]
        settings.use_rag = original["use_rag"]
        settings.artifact_dir = original["artifact_dir"]

    by_key: dict[tuple[int, str], dict[str, dict[str, object]]] = {}
    for record in records:
        by_key.setdefault((int(record["repetition"]), str(record["query_id"])), {})[
            str(record["mode"])
        ] = record
    pairs = [pair for pair in by_key.values() if set(pair) == {"rag", "no-rag"}]
    summary = {
        "run_id": run_id,
        "total_records": len(records),
        "complete_pairs": len(pairs),
        "valid_pairs": sum(
            pair["rag"]["status"] == pair["no-rag"]["status"] == "ok" for pair in pairs
        ),
        "failed_or_incomplete": len(by_key) - len(pairs)
        + sum(any(value["status"] != "ok" for value in pair.values()) for pair in pairs),
        "quality_delta": None,
        "latency_delta": None,
        "limitations": [
            "Este executor não calcula qualidade/latência por consulta; use os planos de métricas e hardware para preenchê-las.",
            "Métricas de recuperação em no-rag são nulas por não serem aplicáveis.",
        ],
    }
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Comparação concluída. Artefatos: {output_dir}")


@app.command()
def run(
    csv_path: Path = typer.Argument(..., help="Caminho do arquivo CSV de entrada"),
    db_path: Path = typer.Option(None, "--db", "-d", help="Caminho do banco SQLite"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Número de exemplos similares"),
    model: str = typer.Option(None, "--model", "-m", help="Modelo de chat do Ollama"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Modelo de embeddings"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Desativa retrieval (gera sem exemplos)"),
    rerank: bool = typer.Option(False, "--rerank", help="Reordena candidatos recuperados"),
    candidate_k: int = typer.Option(None, "--candidate-k", min=1),
) -> None:
    """Executa o pipeline RAG síncrono no CSV."""
    _apply_settings(db_path, top_k, model, embed_model, no_rag, rerank, candidate_k)

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
    rerank: bool = typer.Option(False, "--rerank", help="Reordena candidatos recuperados"),
    candidate_k: int = typer.Option(None, "--candidate-k", min=1),
) -> None:
    """Executa o pipeline RAG assíncrono no CSV (mais rápido)."""
    _apply_settings(db_path, top_k, model, embed_model, no_rag, rerank, candidate_k)

    if embed_concurrency:
        settings.embed_concurrency = embed_concurrency
    if gen_concurrency:
        settings.gen_concurrency = gen_concurrency

    if not csv_path.exists():
        print(f"  Arquivo não encontrado: {csv_path}")
        raise typer.Exit(1)

    orchestrator = AsyncPipelineOrchestrator()
    asyncio.run(orchestrator.run(csv_path))


@app.command("compare")
def compare(
    csv_path: Path = typer.Argument(..., help="CSV de avaliação"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Novo diretório de artefatos"),
    repetitions: int = typer.Option(1, "--repetitions", min=1),
    async_mode: bool = typer.Option(False, "--async", help="Usa o orquestrador assíncrono"),
) -> None:
    """Compara RAG/no-RAG nas mesmas linhas, com estado isolado por modo."""
    if not csv_path.is_file():
        raise typer.BadParameter(f"CSV inexistente: {csv_path}")
    _run_paired(csv_path, output_dir, use_async=async_mode, repetitions=repetitions)


@app.command()
def evaluate(
    csv_path: Path = typer.Argument(..., help="CSV respondido para avaliar"),
) -> None:
    """Avalia a acurácia das respostas."""
    import subprocess

    subprocess.run([sys.executable, "acerto.py", str(csv_path)], check=False)


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
