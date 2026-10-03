from typing import Any

from rank_bm25 import BM25Okapi

from src.db.connection import ConnectionManager
from src.embedding.provider import EmbeddingProvider
from src.models.question import Question, RetrievedExample
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        cm: ConnectionManager,
        embed_provider: EmbeddingProvider,
    ):
        self.cm = cm
        self.embed_provider = embed_provider
        self.vector_store = VectorStore(cm)
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict[str, Any]] = []

    def _build_bm25_index(self) -> None:
        with self.cm.connection() as conn:
            cur = conn.execute(

                    "SELECT id, text, operation FROM questions "
                    "WHERE answer IS NOT NULL AND answer != ''"

            )
            rows = cur.fetchall()

        self._bm25_corpus = []
        tokenized_corpus = []
        for row in rows:
            doc = {
                "id": row["id"],
                "text": row["text"],
                "operation": row["operation"],
            }
            self._bm25_corpus.append(doc)
            # Tokenize: split on whitespace and punctuation
            tokens = (row["text"] + " " + (row["operation"] or "")).lower().split()
            tokenized_corpus.append(tokens)

        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)

    def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            self._build_bm25_index()

    async def retrieve(
        self,
        query: str,
        top_k: int,
        exclude_id: int | None = None,
    ) -> list[RetrievedExample]:
        self._ensure_bm25()

        # 1. Vector search (get more candidates for fusion)
        q_emb = await self.embed_provider.embed(query)
        vector_results = self.vector_store.retrieve_questions(q_emb, top_k * 3, exclude_id)

        # 2. BM25 search
        keyword_results = self._bm25_search(query, top_k * 3, exclude_id)

        # 3. Reciprocal Rank Fusion
        fused = self._rrf_fuse(vector_results, keyword_results, k=60)

        # 4. Return top_k
        return fused[:top_k]

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        exclude_id: int | None = None,
    ) -> list[RetrievedExample]:
        if not self._bm25:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            doc = self._bm25_corpus[idx]
            if exclude_id is not None and doc["id"] == exclude_id:
                continue

            with self.cm.connection() as conn:
                cur = conn.execute(
                    (
                        "SELECT id, text, operation, level, question_type, answer, metadata "
                        "FROM questions WHERE id = ?"
                    ),
                    (doc["id"],),
                )
                row = cur.fetchone()
                if row and row["answer"]:
                    q = Question(
                        id=row["id"],
                        text=row["text"],
                        operation=row["operation"],
                        level=row["level"],
                        question_type=row["question_type"],
                        answer=row["answer"],
                    )
                    # Use negative score as distance (lower is better)
                    results.append(RetrievedExample(question=q, distance=-scores[idx]))

        return results

    def _rrf_fuse(
        self,
        vector_results: list[RetrievedExample],
        keyword_results: list[RetrievedExample],
        k: int = 60,
    ) -> list[RetrievedExample]:
        """Reciprocal Rank Fusion to combine vector and keyword search results."""
        # Map question_id -> (rank, result)
        vec_ranks: dict[int, tuple[int, RetrievedExample]] = {}
        for rank, res in enumerate(vector_results):
            if res.question.id is not None:
                vec_ranks[res.question.id] = (rank, res)

        kw_ranks: dict[int, tuple[int, RetrievedExample]] = {}
        for rank, res in enumerate(keyword_results):
            if res.question.id is not None:
                kw_ranks[res.question.id] = (rank, res)

        # Compute RRF scores
        all_ids = set(vec_ranks.keys()) | set(kw_ranks.keys())
        scored: list[tuple[float, RetrievedExample]] = []

        for qid in all_ids:
            rrf_score = 0.0
            if qid in vec_ranks:
                rrf_score += 1.0 / (k + vec_ranks[qid][0] + 1)
            if qid in kw_ranks:
                rrf_score += 1.0 / (k + kw_ranks[qid][0] + 1)

            # Prefer the result with more info (chunk_text if available)
            result = vec_ranks[qid][1] if qid in vec_ranks else kw_ranks[qid][1]
            scored.append((rrf_score, result))

        # Sort by RRF score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [res for _, res in scored]
