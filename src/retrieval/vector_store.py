from sqlite_vec import serialize_float32

from src.db.connection import ConnectionManager
from src.models.question import Question, RetrievedExample


class VectorStore:
    def __init__(self, cm: ConnectionManager):
        self.cm = cm

    def retrieve_questions(
        self,
        query_embedding: list[float],
        top_k: int,
        exclude_id: int | None = None,
    ) -> list[RetrievedExample]:
        blob = serialize_float32(query_embedding)

        with self.cm.connection() as conn:
            if exclude_id is not None:
                sql = (
                    "SELECT v.rowid, v.distance, q.id, q.text, q.operation, q.level, "
                    "q.question_type, q.answer, q.metadata "
                    "FROM vss_questions v "
                    "JOIN questions q ON q.id = v.rowid "
                    "WHERE v.embedding MATCH ? "
                    "AND k = ? "
                    "AND v.rowid != ? "
                    "AND q.answer IS NOT NULL "
                    "AND q.answer != ''"
                )
                cur = conn.execute(sql, (blob, top_k, exclude_id))
            else:
                sql = (
                    "SELECT v.rowid, v.distance, q.id, q.text, q.operation, q.level, "
                    "q.question_type, q.answer, q.metadata "
                    "FROM vss_questions v "
                    "JOIN questions q ON q.id = v.rowid "
                    "WHERE v.embedding MATCH ? "
                    "AND k = ? "
                    "AND q.answer IS NOT NULL "
                    "AND q.answer != ''"
                )
                cur = conn.execute(sql, (blob, top_k))

            results = []
            for row in cur.fetchall():
                q = Question(
                    id=row["id"],
                    text=row["text"],
                    operation=row["operation"],
                    level=row["level"],
                    question_type=row["question_type"],
                    answer=row["answer"],
                )
                results.append(RetrievedExample(question=q, distance=row["distance"]))
            return results

    def retrieve_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        exclude_question_id: int | None = None,
    ) -> list[RetrievedExample]:
        blob = serialize_float32(query_embedding)

        with self.cm.connection() as conn:
            if exclude_question_id is not None:
                sql = (
                    "SELECT v.rowid, v.distance, c.chunk_text, q.id, q.text, q.operation, q.level, "
                    "q.question_type, q.answer, q.metadata "
                    "FROM vss_chunks v "
                    "JOIN chunks c ON c.id = v.rowid "
                    "JOIN questions q ON q.id = c.question_id "
                    "WHERE v.embedding MATCH ? "
                    "AND k = ? "
                    "AND c.question_id != ? "
                    "AND q.answer IS NOT NULL "
                    "AND q.answer != ''"
                )
                cur = conn.execute(sql, (blob, top_k, exclude_question_id))
            else:
                sql = (
                    "SELECT v.rowid, v.distance, c.chunk_text, q.id, q.text, q.operation, q.level, "
                    "q.question_type, q.answer, q.metadata "
                    "FROM vss_chunks v "
                    "JOIN chunks c ON c.id = v.rowid "
                    "JOIN questions q ON q.id = c.question_id "
                    "WHERE v.embedding MATCH ? "
                    "AND k = ? "
                    "AND q.answer IS NOT NULL "
                    "AND q.answer != ''"
                )
                cur = conn.execute(sql, (blob, top_k))

            results = []
            for row in cur.fetchall():
                q = Question(
                    id=row["id"],
                    text=row["text"],
                    operation=row["operation"],
                    level=row["level"],
                    question_type=row["question_type"],
                    answer=row["answer"],
                )
                results.append(
                    RetrievedExample(
                        question=q, distance=row["distance"], chunk_text=row["chunk_text"]
                    )
                )
            return results
