import json
import sqlite3

from src.db.connection import ConnectionManager
from src.models.enums import DifficultyLevel
from src.models.question import Chunk, Question


class QuestionRepository:
    def __init__(self, cm: ConnectionManager):
        self.cm = cm

    def upsert(self, q: Question) -> int:
        with self.cm.connection() as conn:
            if q.id is not None:
                cur = conn.execute("SELECT id FROM questions WHERE id = ?", (q.id,))
                if cur.fetchone():
                    conn.execute(
                        (
                            "UPDATE questions SET text=?, operation=?, level=?, "
                            "question_type=?, answer=?, ground_truth=?, metadata=? WHERE id=?"
                        ),
                        (
                            q.text,
                            q.operation,
                            q.level.value,
                            q.question_type,
                            q.answer,
                            q.ground_truth,
                            json.dumps(q.metadata),
                            q.id,
                        ),
                    )
                    conn.commit()
                    return q.id

            if q.id is not None:
                cur = conn.execute(
                    """INSERT INTO questions (id, text, operation, level, question_type, answer,
                       ground_truth, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        q.id,
                        q.text,
                        q.operation,
                        q.level.value,
                        q.question_type,
                        q.answer,
                        q.ground_truth,
                        json.dumps(q.metadata),
                    ),
                )
            else:
                cur = conn.execute(
                    """INSERT INTO questions (text, operation, level, question_type, answer,
                       ground_truth, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        q.text,
                        q.operation,
                        q.level.value,
                        q.question_type,
                        q.answer,
                        q.ground_truth,
                        json.dumps(q.metadata),
                    ),
                )
            conn.commit()
            qid = cur.lastrowid
            if qid is None:
                raise RuntimeError("Failed to insert question")
            qid = int(qid)

            # Update FTS
            conn.execute(
                "INSERT INTO fts_questions (rowid, text, operation) VALUES (?, ?, ?)",
                (qid, q.text, q.operation or ""),
            )
            conn.commit()

            return qid

    def get_by_id(self, qid: int) -> Question | None:
        with self.cm.connection() as conn:
            cur = conn.execute(
                (
                    "SELECT id, text, operation, level, question_type, answer, ground_truth, metadata "
                    "FROM questions WHERE id = ?"
                ),
                (qid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_question(row)

    def get_by_text(self, text: str) -> Question | None:
        with self.cm.connection() as conn:
            cur = conn.execute(
                (
                    "SELECT id, text, operation, level, question_type, answer, ground_truth, metadata "
                    "FROM questions WHERE text = ?"
                ),
                (text,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_question(row)

    def get_unanswered(self, limit: int | None = None) -> list[Question]:
        with self.cm.connection() as conn:
            sql = (
                "SELECT id, text, operation, level, question_type, answer, ground_truth, metadata "
                "FROM questions WHERE answer IS NULL OR answer = ''"
            )
            if limit:
                sql += f" LIMIT {limit}"
            cur = conn.execute(sql)
            return [self._row_to_question(row) for row in cur.fetchall()]

    def get_all(self, limit: int | None = None) -> list[Question]:
        with self.cm.connection() as conn:
            sql = (
                "SELECT id, text, operation, level, question_type, answer, ground_truth, metadata FROM questions"
            )
            if limit:
                sql += f" LIMIT {limit}"
            cur = conn.execute(sql)
            return [self._row_to_question(row) for row in cur.fetchall()]

    def _row_to_question(self, row: sqlite3.Row) -> Question:
        return Question(
            id=row["id"],
            text=row["text"],
            operation=row["operation"],
            level=DifficultyLevel.from_str(row["level"] or ""),
            question_type=row["question_type"],
            answer=row["answer"],
            ground_truth=row["ground_truth"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class ChunkRepository:
    def __init__(self, cm: ConnectionManager):
        self.cm = cm

    def insert_chunks(self, question_id: int, chunks: list[str]) -> list[int]:
        with self.cm.connection() as conn:
            chunk_ids = []
            for idx, chunk_text in enumerate(chunks):
                cur = conn.execute(
                    (
                        "INSERT OR IGNORE INTO chunks "
                        "(question_id, chunk_index, chunk_text) VALUES (?, ?, ?)"
                    ),
                    (question_id, idx, chunk_text),
                )
                if cur.lastrowid:
                    chunk_ids.append(cur.lastrowid)
                else:
                    # Already exists, fetch the id
                    cur = conn.execute(
                        "SELECT id FROM chunks WHERE question_id = ? AND chunk_index = ?",
                        (question_id, idx),
                    )
                    row = cur.fetchone()
                    if row:
                        chunk_ids.append(row["id"])
            conn.commit()
            return chunk_ids

    def get_chunks(self, question_id: int) -> list[Chunk]:
        with self.cm.connection() as conn:
            cur = conn.execute(
                (
                    "SELECT id, question_id, chunk_index, chunk_text "
                    "FROM chunks WHERE question_id = ? ORDER BY chunk_index"
                ),
                (question_id,),
            )
            return [
                Chunk(
                    id=row["id"],
                    question_id=row["question_id"],
                    index=row["chunk_index"],
                    text=row["chunk_text"],
                )
                for row in cur.fetchall()
            ]


class EmbeddingRepository:
    def __init__(self, cm: ConnectionManager):
        self.cm = cm

    def store_question_embedding(self, qid: int, embedding: list[float]) -> None:
        from sqlite_vec import serialize_float32

        with self.cm.connection() as conn:
            blob = serialize_float32(embedding)
            conn.execute("DELETE FROM vss_questions WHERE rowid = ?", (qid,))
            conn.execute("INSERT INTO vss_questions (rowid, embedding) VALUES (?, ?)", (qid, blob))
            conn.commit()

    def store_chunk_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        from sqlite_vec import serialize_float32

        with self.cm.connection() as conn:
            blob = serialize_float32(embedding)
            conn.execute("DELETE FROM vss_chunks WHERE rowid = ?", (chunk_id,))
            conn.execute(
                "INSERT INTO vss_chunks (rowid, embedding) VALUES (?, ?)", (chunk_id, blob)
            )
            conn.commit()

    def has_question_embedding(self, qid: int) -> bool:
        with self.cm.connection() as conn:
            cur = conn.execute("SELECT 1 FROM vss_questions WHERE rowid = ?", (qid,))
            return cur.fetchone() is not None

    def has_chunk_embeddings(self, qid: int) -> bool:
        with self.cm.connection() as conn:
            cur = conn.execute(
                """
                SELECT 1 FROM vss_chunks v
                JOIN chunks c ON c.id = v.rowid
                WHERE c.question_id = ?
                LIMIT 1
                """,
                (qid,),
            )
            return cur.fetchone() is not None
