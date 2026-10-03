import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.db import (
    ChunkRepository,
    ConnectionManager,
    EmbeddingRepository,
    QuestionRepository,
    init_database,
)
from src.embedding.provider import OllamaEmbeddingProvider
from src.generation.provider import OllamaLLMProvider
from src.io import StreamingCSVReader, StreamingCSVWriter
from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.chunking import chunk_question
from src.pipeline.ollama_sync import check_ollama_sync
from src.pipeline.prompt import build_prompt
from src.retrieval import HybridRetriever
from src.utils.hardware_monitor import HardwareMonitor


@dataclass
class PipelineMetrics:
    total_questions: int = 0
    questions_with_answer: int = 0
    questions_answered: int = 0
    embed_time: float = 0.0
    retrieve_time: float = 0.0
    generate_time: float = 0.0
    total_time: float = 0.0
    batch_times: list[float] = field(default_factory=list)
    hardware_summary: dict = field(default_factory=dict)

    @property
    def throughput_qps(self) -> float:
        return self.questions_answered / self.total_time if self.total_time > 0 else 0.0

    @property
    def avg_embed_time(self) -> float:
        return self.embed_time / self.total_questions if self.total_questions > 0 else 0.0

    @property
    def avg_generate_time(self) -> float:
        return self.generate_time / self.questions_answered if self.questions_answered > 0 else 0.0

    def summary(self) -> str:
        return (
            f"\n  === Métricas de Performance ===\n"
            f"  Total questões: {self.total_questions}\n"
            f"  Já respondidas: {self.questions_with_answer}\n"
            f"  Respondidas agora: {self.questions_answered}\n"
            f"  Tempo total: {self.total_time:.1f}s\n"
            f"  Throughput: {self.throughput_qps:.2f} q/s\n"
            f"  Tempo embeddings: {self.embed_time:.1f}s (média: {self.avg_embed_time:.3f}s/q)\n"
            f"  Tempo retrieval: {self.retrieve_time:.1f}s\n"
            f"  Tempo geração: {self.generate_time:.1f}s (média: {self.avg_generate_time:.3f}s/q)\n"
            f"{self._hardware_summary_line()}"
        )

    def _hardware_summary_line(self) -> str:
        if not self.hardware_summary:
            return ""
        hs = self.hardware_summary
        return (
            "  === Resumo de Hardware ===\n"
            f"  CPU média: {hs.get('cpu_percent_mean', 0.0):.1f}%\n"
            f"  CPU pico: {hs.get('cpu_percent_peak', 0.0):.1f}%\n"
            f"  RAM pico (RSS): {hs.get('rss_mb_peak', 0.0):.1f} MB\n"
            f"  Threads máximas: {hs.get('num_threads_peak', 0)}\n"
            f"  Duração total: {hs.get('total_elapsed_s', 0.0):.1f}s\n"
        )


class AsyncPipelineOrchestrator:
    def __init__(self) -> None:
        self.cm = ConnectionManager(settings.db_path)
        init_database(settings.db_path)
        self.qrepo = QuestionRepository(self.cm)
        self.crepo = ChunkRepository(self.cm)
        self.erepo = EmbeddingRepository(self.cm)
        self.embedder = OllamaEmbeddingProvider()
        self.llm = OllamaLLMProvider()
        self.retriever = HybridRetriever(self.cm, self.embedder)
        self.checkpoint = CheckpointManager(settings.checkpoint_path, settings.checkpoint_interval)
        self.metrics = PipelineMetrics()
        self.monitor = HardwareMonitor()

    async def run(self, csv_path: Path) -> PipelineMetrics:
        t_start = time.time()
        check_ollama_sync()

        self.monitor.start(mode="rag" if settings.use_rag else "no_rag")

        # Phase 1: Ingest + Embeddings (async, batched)
        self.monitor.sample("preparation_indexing")
        await self._phase_ingest_async(csv_path)
        self.monitor.sample("embedding_indexing")

        # Phase 2: Answer unanswered (async, batched)
        await self._phase_answer_async()
        self.monitor.sample("retrieval_generation")

        # Phase 3: Export
        self._phase_export(csv_path)
        self.monitor.sample("export")

        await self.embedder.close()
        await self.llm.close()
        self.cm.close_all()

        self.metrics.total_time = time.time() - t_start
        self.monitor.save_json(str(Path(f"hardware_log_{int(time.time() * 1000)}.json")))
        self.metrics.hardware_summary = self.monitor.summarize()
        print(self.metrics.summary())
        print(self.monitor.format_report())
        return self.metrics

    async def _phase_ingest_async(self, csv_path: Path) -> None:
        print("  Gerando embeddings e populando banco vetorial (async)...")
        reader = StreamingCSVReader(csv_path, chunk_size=settings.embed_batch_size)

        total_processed = 0
        for batch in reader.iter_batches():
            # Filter already processed
            batch = [q for q in batch if not self.checkpoint.is_processed(q.id or 0)]
            if not batch:
                continue

            self.metrics.total_questions += len(batch)

            # Upsert questions (sync, fast)
            qids = [self.qrepo.upsert(q) for q in batch]

            # Batch embed questions
            texts = [q.text for q in batch]
            t0 = time.time()
            embeddings = await self.embedder.embed_batch(texts)
            self.metrics.embed_time += time.time() - t0
            for qid, emb in zip(qids, embeddings):
                self.erepo.store_question_embedding(qid, emb)

            # Chunking + chunk embeddings (batched)
            all_chunks = []
            chunk_qids = []
            for qid, q in zip(qids, batch):
                chunks = chunk_question(q.text)
                if len(chunks) > 1:
                    self.crepo.insert_chunks(qid, chunks)
                    all_chunks.extend(chunks)
                    chunk_qids.extend([qid] * len(chunks))

            if all_chunks:
                t0 = time.time()
                chunk_embeddings = await self.embedder.embed_batch(all_chunks)
                self.metrics.embed_time += time.time() - t0
                for chunk_id, emb in zip(chunk_qids, chunk_embeddings):
                    pass  # We'll handle this below

                # Actually, we need to get the chunk_ids in the same order
                # Let's re-fetch them
                for qid, q in zip(qids, batch):
                    chunks = chunk_question(q.text)
                    if len(chunks) > 1:
                        chunk_rows = self.crepo.get_chunks(qid)
                        for chunk_row, chunk_text in zip(chunk_rows, chunks):
                            if chunk_row.id is not None:
                                chunk_emb = await self.embedder.embed(chunk_text)
                                self.erepo.store_chunk_embedding(chunk_row.id, chunk_emb)

            for qid in qids:
                self.checkpoint.mark_processed(qid)

            total_processed += len(batch)
            print(f"  [{total_processed}] ✔ batch processado", end="\r")

        print("\n  Embeddings prontos.\n")
        self.checkpoint.force_flush()

    async def _phase_answer_async(self) -> None:
        unanswered = self.qrepo.get_unanswered()

        if not unanswered:
            print("ℹ  Todas as questões já possuem resposta. Nada a fazer.")
            return

        self.metrics.questions_with_answer = len(self.qrepo.get_all()) - len(unanswered)
        self.metrics.questions_answered = len(unanswered)

        print(f"  Respondendo {len(unanswered)} questão(ões) com RAG (async)...\n")

        total_model_time = 0.0
        batch_size = settings.gen_concurrency
        answered_count = 0
        failed_count = 0

        for i in range(0, len(unanswered), batch_size):
            batch = unanswered[i : i + batch_size]
            batch_start = time.time()

            # Retrieve for all in batch (concurrent) — skipped when --no-rag
            t0 = time.time()
            if settings.use_rag:
                retrieve_tasks = [
                    self.retriever.retrieve(q.text, settings.top_k, q.id) for q in batch
                ]
                all_similar = await asyncio.gather(*retrieve_tasks, return_exceptions=True)
            else:
                all_similar = [[] for _ in batch]
            self.metrics.retrieve_time += time.time() - t0

            # Build prompts - handle retrieval failures
            prompts: list[str] = []
            valid_indices: list[int] = []
            for idx, (q, similar) in enumerate(zip(batch, all_similar)):
                if isinstance(similar, Exception):
                    failed_count += 1
                    print(
                        f"  [{i + idx + 1}/{len(unanswered)}]  "
                        f"✗ Falha na recuperação: {q.text[:60]}… → {similar}"
                    )
                else:
                    # similar is guaranteed to be list[RetrievedExample] here after isinstance check
                    prompts.append(build_prompt(q.text, similar))  # type: ignore[arg-type]
                    valid_indices.append(idx)

            # Batch generate for valid prompts
            answers = []
            if prompts:
                t0 = time.time()
                try:
                    answers = await self.llm.generate_batch(prompts)
                    elapsed = time.time() - t0
                    self.metrics.generate_time += elapsed
                    total_model_time += elapsed
                except Exception as e:
                    elapsed = time.time() - t0
                    self.metrics.generate_time += elapsed
                    total_model_time += elapsed
                    # Mark all valid prompts as failed
                    for idx in valid_indices:
                        failed_count += 1
                        q = batch[idx]
                        print(
                            f"  [{i + idx + 1}/{len(unanswered)}]  "
                            f"✗ Falha na geração: {q.text[:60]}… → {e}"
                        )
                    answers = [""] * len(prompts)

            # Save answers - handle per-question
            answer_idx = 0
            for idx, q in enumerate(batch):
                if idx not in valid_indices:
                    continue

                answer = answers[answer_idx]
                answer_idx += 1

                if not answer or not answer.strip():
                    failed_count += 1
                    print(f"  [{i + idx + 1}/{len(unanswered)}]  ✗ Resposta vazia: {q.text[:60]}…")
                    continue

                print(f"  [{i + idx + 1}/{len(unanswered)}]  {q.text[:60]}… → {answer[:80]}")
                q.answer = answer
                self.qrepo.upsert(q)

                if q.id is not None:
                    self.checkpoint.mark_processed(q.id)

                answered_count += 1

            self.metrics.batch_times.append(time.time() - batch_start)

        print(f"\n  Tempo total gasto no modelo: {total_model_time:.1f} segundos.")
        if answered_count:
            print(f"  Tempo médio por questão: {total_model_time / answered_count:.1f} segundos.")
        print(f"  Respondidas com sucesso: {answered_count}")
        if failed_count:
            print(f"  Falhas: {failed_count}")

    def _phase_export(self, csv_path: Path) -> None:
        output_path = csv_path.with_stem(csv_path.stem + "_respondido")
        questions = self.qrepo.get_all()

        with StreamingCSVWriter(output_path) as writer:
            for q in questions:
                writer.write_question(q)

        print(f"\n  Concluído! CSV salvo em: {output_path}")
        print(f"  Banco SQLite: {settings.db_path}")
