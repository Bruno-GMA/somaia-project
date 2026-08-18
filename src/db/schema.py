SCHEMA_QUESTIONS = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    operation TEXT,
    level TEXT,
    question_type TEXT,
    answer TEXT,
    ground_truth TEXT,
    metadata TEXT DEFAULT '{}'
)
"""

SCHEMA_VSS_QUESTIONS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vss_questions USING vec0(
    embedding float[{768}] distance_metric=cosine
)
"""

SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    UNIQUE(question_id, chunk_index)
)
"""

SCHEMA_VSS_CHUNKS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vec0(
    embedding float[{768}] distance_metric=cosine
)
"""

SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_questions USING fts5(
    text,
    operation,
    content='questions',
    content_rowid='id'
)
"""

ALL_SCHEMAS = [
    SCHEMA_QUESTIONS,
    SCHEMA_VSS_QUESTIONS,
    SCHEMA_CHUNKS,
    SCHEMA_VSS_CHUNKS,
    SCHEMA_FTS,
]

MIGRATIONS: list[str] = [
    # Future migrations would go here
]


def init_database(db_path: str) -> None:
    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    for schema in ALL_SCHEMAS:
        conn.execute(schema)

    conn.commit()
    conn.close()
