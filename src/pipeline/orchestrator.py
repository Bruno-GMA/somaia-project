import time
from pathlib import Path

from src.config import settings
from src.db import (
    ChunkRepository,
    ConnectionManager,
    EmbeddingRepository,
    QuestionRepository,
    init_database,
)
from src.io import StreamingCSVReader, StreamingCSVWriter
from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.chunking import chunk_question
from src.pipeline.ollama_sync import check_ollama_sync, ollama_chat_sync, ollama_embed_sync
from src.pipeline.prompt import build_prompt
from src.retrieval.vector_store import VectorStore


class SyncPipelineOrchestrator:
    def __init__(self) -> None:
        self.cm = ConnectionManager(settings.db_path)
        init_database(settings.db_path)
        self.qrepo = QuestionRepository(self.cm)
        self.crepo = ChunkRepository(self.cm)
        self.erepo = EmbeddingRepository(self.cm)
        self.vector_store = VectorStore(self.cm)
        self.checkpoint = CheckpointManager(settings.checkpoint_path, settings.checkpoint_interval)

    def run(self, csv_path: Path) -> None:
        check_ollama_sync()

        # Phase 1: Ingest + Embeddings
        self._phase_ingest(csv_path)

        # Phase 2: Answer unanswered
        self._phase_answer()

        # Phase 3: Export
        self._phase_export(csv_path)

        self.cm.close_all()

    def _phase_ingest(self, csv_path: Path) -> None:
        print("  Gerando embeddings e populando banco vetorial…")
        reader = StreamingCSVReader(csv_path)

        for batch in reader.iter_batches():
            for q in batch:
                qid = q.id or 0
                if self.checkpoint.is_processed(qid):
                    continue

                qid = self.qrepo.upsert(q)

                # Embedding da questão inteira
                if not self.erepo.has_question_embedding(qid):
                    emb = ollama_embed_sync(q.text)
                    self.erepo.store_question_embedding(qid, emb)

                # Chunking para enunciados longos
                chunks = chunk_question(q.text)
                if len(chunks) > 1:
                    chunk_ids = self.crepo.insert_chunks(qid, chunks)
                    if not self.erepo.has_chunk_embeddings(qid):
                        for chunk_id, chunk_text in zip(chunk_ids, chunks):
                            chunk_emb = ollama_embed_sync(chunk_text)
                            self.erepo.store_chunk_embedding(chunk_id, chunk_emb)

                self.checkpoint.mark_processed(qid)
                print(f"  [{qid}] ✔ embedding armazenado", end="\r")

        print("\n  Embeddings prontos.\n")
        self.checkpoint.force_flush()

    def _phase_answer(self) -> None:
        unanswered = self.qrepo.get_unanswered()

        if not unanswered:
            print("ℹ  Todas as questões já possuem resposta. Nada a fazer.")
            return

        print(f"  Respondendo {len(unanswered)} questão(ões) com RAG…\n")

        total_model_time = 0.0
        answered_count = 0
        failed_count = 0

        for i, q in enumerate(unanswered, 1):
            try:
                if settings.use_rag:
                    # Verifica se a questão foi fatiada em chunks
                    has_chunks = False
                    with self.cm.connection() as conn:
                        cur = conn.execute(
                            "SELECT 1 FROM chunks WHERE question_id = ? LIMIT 1", (q.id,)
                        )
                        has_chunks = cur.fetchone() is not None

                    if has_chunks:
                        chunks = chunk_question(q.text)
                        first_chunk_emb = ollama_embed_sync(chunks[0])
                        similar = self.vector_store.retrieve_chunks(
                            first_chunk_emb, settings.top_k, exclude_question_id=q.id
                        )
                    else:
                        q_emb = ollama_embed_sync(q.text)
                        similar = self.vector_store.retrieve_questions(
                            q_emb, settings.top_k, exclude_id=q.id
                        )
                else:
                    similar = []

                prompt = build_prompt(q.text, similar)

                print(f"  [{i}/{len(unanswered)}]  Resolvendo: {q.text[:60]}…")
                t0 = time.time()
                answer = ollama_chat_sync(prompt)
                elapsed = time.time() - t0
                total_model_time += elapsed

                if not answer or not answer.strip():
                    raise ValueError("Modelo retornou resposta vazia")

                print(f"          ({elapsed:.1f}s) → {answer[:80]}")

                q.answer = answer
                self.qrepo.upsert(q)

                if q.id is not None:
                    self.checkpoint.mark_processed(q.id)

                answered_count += 1

            except Exception as e:
                failed_count += 1
                print(f"  [{i}/{len(unanswered)}]  ✗ Falha: {q.text[:60]}… → {e}")

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
