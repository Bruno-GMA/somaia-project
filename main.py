
import csv
import sqlite3
import sys
import time
from pathlib import Path

import requests
import sqlite_vec
from sqlite_vec import serialize_float32


# CONFIGURAÇÕES

OLLAMA_BASE_URL   = "http://localhost:11434"
CHAT_MODEL        = "gemma2:2b"          #escolha modelo para responder  (ollama run gemma2:2b)
EMBED_MODEL       = "nomic-embed-text"   # modelo para embeddings (ollama run nomic-embed-text)
EMBED_DIM         = 768                  # dimensão dos vetores do nomic-embed-text
DB_PATH           = "math_rag.db"
ANSWER_COLUMN     = "resposta"
TOP_K             = 3                    # nº de exemplos similares usados como contexto
CHUNK_THRESHOLD   = 256                  # nº de palavras acima do qual a questão é fatiada em chunks
CHUNK_OVERLAP     = 32                   # sobreposição em palavras entre chunks consecutivos


# ──────────────────────────────────────────────
# OLLAMA — helpers
# ──────────────────────────────────────────────

def ollama_embed(text: str) -> list[float]:
    """Retorna o vetor de embedding de um texto via Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def ollama_chat(prompt: str) -> str:
    """Envia uma mensagem ao modelo de chat e retorna a resposta completa."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def check_ollama():
    """Verifica se o Ollama está rodando e se os modelos necessários existem."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception:
        print("  Ollama não está acessível em", OLLAMA_BASE_URL)
        print("    Inicie com: ollama serve")
        sys.exit(1)

    available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    for model in (CHAT_MODEL, EMBED_MODEL):
        base = model.split(":")[0]
        if base not in available:
            print(f"  Modelo '{model}' não encontrado. Baixando…")
            requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model},
                timeout=600,
            )
            print(f"  '{model}' baixado.")


# BANCO DE DADOS (SQLite + sqlite-vec)

def init_db(db_path: str) -> sqlite3.Connection:
    """Cria/abre o banco, carrega a extensão sqlite-vec e garante que as tabelas existem."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")

    # Tabela principal — sem embedding (vetores agora vivem na virtual table)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT    NOT NULL,
            answer    TEXT
        )
    """)

    # Virtual table vetorial — busca por similaridade de cosseno no próprio banco
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vss_questions USING vec0(
            embedding float[{EMBED_DIM}] distance_metric=cosine
        )
    """)

    # Tabela de chunks — para suportar enunciados longos fatiados
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL REFERENCES questions(id),
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT    NOT NULL,
            UNIQUE(question_id, chunk_index)
        )
    """)
    # Virtual table vetorial para os chunks
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vec0(
            embedding float[{EMBED_DIM}] distance_metric=cosine
        )
    """)
    conn.commit()
    return conn


def upsert_question(conn: sqlite3.Connection, question: str, answer: str | None = None) -> int:
    """Insere ou atualiza uma questão. Retorna o id da linha."""
    cur = conn.execute("SELECT id FROM questions WHERE question = ?", (question,))
    row = cur.fetchone()
    if row:
        if answer is not None:
            conn.execute(
                "UPDATE questions SET answer = ? WHERE id = ?",
                (answer, row[0]),
            )
            conn.commit()
        return row[0]
    cur = conn.execute(
        "INSERT INTO questions (question, answer) VALUES (?, ?)",
        (question, answer),
    )
    conn.commit()
    return cur.lastrowid


def store_embedding(conn: sqlite3.Connection, question_id: int, embedding: list[float]):
    """Armazena o vetor de embedding como BLOB binário na virtual table vss_questions."""
    blob = serialize_float32(embedding)
    # vec0 não suporta UPDATE — remove e reinsere
    conn.execute("DELETE FROM vss_questions WHERE rowid = ?", (question_id,))
    conn.execute(
        "INSERT INTO vss_questions (rowid, embedding) VALUES (?, ?)",
        (question_id, blob),
    )
    conn.commit()


def store_chunk_embedding(conn: sqlite3.Connection, chunk_id: int, embedding: list[float]):
    """Armazena o embedding de um chunk na virtual table vss_chunks."""
    blob = serialize_float32(embedding)
    conn.execute("DELETE FROM vss_chunks WHERE rowid = ?", (chunk_id,))
    conn.execute(
        "INSERT INTO vss_chunks (rowid, embedding) VALUES (?, ?)",
        (chunk_id, blob),
    )
    conn.commit()


def has_embedding(conn: sqlite3.Connection, question_id: int) -> bool:
    """Verifica se uma questão já possui embedding na virtual table."""
    cur = conn.execute("SELECT 1 FROM vss_questions WHERE rowid = ?", (question_id,))
    return cur.fetchone() is not None


def has_chunk_embeddings(conn: sqlite3.Connection, question_id: int) -> bool:
    """Verifica se os chunks de uma questão já possuem embeddings."""
    cur = conn.execute(
        """
        SELECT 1 FROM vss_chunks v
        JOIN chunks c ON c.id = v.rowid
        WHERE c.question_id = ?
        LIMIT 1
        """,
        (question_id,),
    )
    return cur.fetchone() is not None



# CHUNKING


def chunk_question(text: str, max_words: int = CHUNK_THRESHOLD, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide um enunciado longo em chunks de palavras com sobreposição.
    Retorna uma lista com um único chunk (o texto original) quando cabe no limite."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks



# RECUPERAÇÃO (VSS — busca vetorial no banco)


def retrieve_similar_vss(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int,
    exclude_id: int | None = None,
) -> list[dict]:
    """Retorna os top_k documentos mais similares à query usando sqlite-vec.

    Faz a busca por cosseno diretamente no banco, sem carregar embeddings em memória.
    Apenas questões que já possuem resposta são consideradas como exemplos.
    """
    blob = serialize_float32(query_embedding)

    if exclude_id is not None:
        sql = """
            SELECT v.rowid, v.distance, q.question, q.answer
            FROM vss_questions v
            JOIN questions q ON q.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              AND v.rowid != ?
              AND q.answer IS NOT NULL
              AND q.answer != ''
        """
        cur = conn.execute(sql, (blob, top_k, exclude_id))
    else:
        sql = """
            SELECT v.rowid, v.distance, q.question, q.answer
            FROM vss_questions v
            JOIN questions q ON q.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              AND q.answer IS NOT NULL
              AND q.answer != ''
        """
        cur = conn.execute(sql, (blob, top_k))

    return [
        {"id": row[0], "distance": row[1], "question": row[2], "answer": row[3]}
        for row in cur.fetchall()
    ]


def retrieve_similar_chunks_vss(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int,
    exclude_question_id: int | None = None,
) -> list[dict]:
    """Busca exemplos similares via chunks (para enunciados longos).
    Retorna a questão pai de cada chunk encontrado."""
    blob = serialize_float32(query_embedding)

    if exclude_question_id is not None:
        sql = """
            SELECT v.rowid, v.distance, c.chunk_text, q.id, q.question, q.answer
            FROM vss_chunks v
            JOIN chunks c ON c.id = v.rowid
            JOIN questions q ON q.id = c.question_id
            WHERE v.embedding MATCH ?
              AND k = ?
              AND c.question_id != ?
              AND q.answer IS NOT NULL
              AND q.answer != ''
        """
        cur = conn.execute(sql, (blob, top_k, exclude_question_id))
    else:
        sql = """
            SELECT v.rowid, v.distance, c.chunk_text, q.id, q.question, q.answer
            FROM vss_chunks v
            JOIN chunks c ON c.id = v.rowid
            JOIN questions q ON q.id = c.question_id
            WHERE v.embedding MATCH ?
              AND k = ?
              AND q.answer IS NOT NULL
              AND q.answer != ''
        """
        cur = conn.execute(sql, (blob, top_k))

    return [
        {
            "id": row[3],
            "distance": row[1],
            "question": row[4],
            "answer": row[5],
            "chunk_text": row[2],
        }
        for row in cur.fetchall()
    ]



# PROMPT RAG


def build_prompt(question: str, similar: list[dict]) -> str:
    examples = ""
    for i, doc in enumerate(similar, 1):
        examples += f"\nExemplo {i}:\nQuestão: {doc['question']}\nResposta: {doc['answer']}\n"

    system = (
        "Você é um especialista em matemática. "
        "Resolva a questão passo a passo e apresente apenas a resposta final de forma clara. "
        "Seja preciso e conciso."
    )

    if examples:
        return (
            f"{system}\n\n"
            f"A seguir, alguns exemplos de questões semelhantes resolvidas:{examples}\n"
            f"Agora resolva:\nQuestão: {question}\nResposta:"
        )
    return f"{system}\n\nQuestão: {question}\nResposta:"


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def detect_question_column(headers: list[str]) -> str:
    """Tenta encontrar automaticamente a coluna de questões."""
    candidates = ["operation", "operação", "pergunta", "problema", "enunciado"]
    lower = [h.lower().strip() for h in headers]
    for c in candidates:
        if c in lower:
            return headers[lower.index(c)]
    # fallback: primeira coluna
    return headers[0]


def process_csv(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"  Arquivo não encontrado: {csv_path}")
        sys.exit(1)

    # ── Ler CSV ──────────────────────────────
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print("  CSV vazio.")
        return

    question_col = detect_question_column(headers)
    print(f"  Coluna de questões detectada: '{question_col}'")
    print(f"  {len(rows)} questão(ões) encontrada(s).\n")


    if ANSWER_COLUMN not in headers:
        headers.append(ANSWER_COLUMN)


    conn = init_db(DB_PATH)


    # ── Fase 1: Inserir questões, gerar embeddings e chunks ──
    print("  Gerando embeddings e populando banco vetorial…")
    question_ids: dict[str, int] = {}
    for i, row in enumerate(rows, 1):
        q = row[question_col].strip()
        existing_answer = row.get(ANSWER_COLUMN, "").strip() or None
        qid = upsert_question(conn, q, existing_answer)
        question_ids[q] = qid

        # Embedding da questão inteira — apenas se ainda não existir
        if not has_embedding(conn, qid):
            emb = ollama_embed(q)
            store_embedding(conn, qid, emb)

        # Chunking opcional para enunciados longos
        chunks = chunk_question(q)
        if len(chunks) > 1:
            for chunk_idx, chunk_text in enumerate(chunks):
                conn.execute(
                    "INSERT OR IGNORE INTO chunks (question_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
                    (qid, chunk_idx, chunk_text),
                )
            conn.commit()
            if not has_chunk_embeddings(conn, qid):
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_row = conn.execute(
                        "SELECT id FROM chunks WHERE question_id = ? AND chunk_index = ?",
                        (qid, chunk_idx),
                    ).fetchone()
                    if chunk_row:
                        chunk_emb = ollama_embed(chunk_text)
                        store_chunk_embedding(conn, chunk_row[0], chunk_emb)

        print(f"  [{i}/{len(rows)}] ✔ embedding armazenado", end="\r")

    print("\n Embeddings prontos.\n")

    # ── Fase 2: RAG — responder questões sem resposta ──
    to_answer = [r for r in rows if not r.get(ANSWER_COLUMN, "").strip()]

    if not to_answer:
        print("ℹ  Todas as questões já possuem resposta. Nada a fazer.")
    else:
        print(f"  Respondendo {len(to_answer)} questão(ões) com RAG (sqlite-vec)…\n")

    total_model_time = 0.0  # tempo total do modelo

    for i, row in enumerate(rows, 1):
        q = row[question_col].strip()
        existing = row.get(ANSWER_COLUMN, "").strip()

        if existing:
            row[ANSWER_COLUMN] = existing
            print(f"  [{i}/{len(rows)}]  já respondida — pulando")
            continue

        qid = question_ids[q]

        # Se a questão foi fatiada em chunks, busca pelos chunks;
        # caso contrário, busca direto pela vss_questions
        has_chunks = conn.execute(
            "SELECT 1 FROM chunks WHERE question_id = ? LIMIT 1", (qid,)
        ).fetchone() is not None

        if has_chunks:
            # Gera embedding do primeiro chunk (ou do texto completo se curto)
            chunks = chunk_question(q)
            first_chunk_emb = ollama_embed(chunks[0])
            similar = retrieve_similar_chunks_vss(conn, first_chunk_emb, TOP_K, exclude_question_id=qid)
        else:
            q_emb = ollama_embed(q)
            similar = retrieve_similar_vss(conn, q_emb, TOP_K, exclude_id=qid)

        prompt = build_prompt(q, similar)

        print(f"  [{i}/{len(rows)}]  Resolvendo: {q[:60]}…")
        t0 = time.time()
        answer = ollama_chat(prompt)
        elapsed = time.time() - t0
        total_model_time += elapsed
        print(f"          ({elapsed:.1f}s) → {answer[:80]}")

        row[ANSWER_COLUMN] = answer
        upsert_question(conn, q, answer)

    conn.close()

    # ── Fase 3: salvar CSV com respostas ──
    output_path = path.with_stem(path.stem + "_respondido")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Concluído! CSV salvo em: {output_path}")
    print(f"  Banco SQLite: {DB_PATH}")
    print(f"  Tempo total gasto no modelo: {total_model_time:.1f} segundos.")
    print(f" Tempo médio por questão: {total_model_time / len(to_answer):.1f} segundos." if to_answer else "")



# ENTRY POINT


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rag_math.py <caminho_do_arquivo.csv>")
        print("Exemplo: python rag_math.py questoes.csv")
        sys.exit(0)

    check_ollama()
    process_csv(sys.argv[1])