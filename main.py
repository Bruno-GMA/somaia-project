
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import requests


# CONFIGURAÇÕES

OLLAMA_BASE_URL   = "http://localhost:11434"
CHAT_MODEL        = "gemma2:2b"          #escolha modelo para responder  (ollama run gemma2:2b)
EMBED_MODEL       = "nomic-embed-text"# modelo para embeddings (ollama run nomic-embed-text)
DB_PATH           = "math_rag.db"
ANSWER_COLUMN     = "resposta"
TOP_K             = 3                 # nº de exemplos similares usados como contexto


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


# BANCO DE DADOS (SQLite)

def init_db(db_path: str) -> sqlite3.Connection:
    """Cria/abre o banco e garante que as tabelas existem."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT    NOT NULL,
            answer    TEXT,
            embedding TEXT                        -- JSON array de floats
        )
    """)
    conn.commit()
    return conn


def upsert_question(conn: sqlite3.Connection, question: str, answer: str | None = None) -> int:
    """Insere ou atualiza uma questão. Retorna o id da linha."""
    cur = conn.execute("SELECT id FROM questions WHERE question = ?", (question,))
    row = cur.fetchone()
    if row:
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
    conn.execute(
        "UPDATE questions SET embedding = ? WHERE id = ?",
        (json.dumps(embedding), question_id),
    )
    conn.commit()


def get_all_with_embeddings(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT id, question, answer, embedding FROM questions WHERE embedding IS NOT NULL"
    )
    rows = []
    for row in cur.fetchall():
        rows.append({
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "embedding": np.array(json.loads(row[3]), dtype=np.float32),
        })
    return rows



# SIMILARIDADE & RECUPERAÇÃO


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve_similar(
    query_embedding: list[float],
    corpus: list[dict],
    top_k: int,
    exclude_id: int | None = None,
) -> list[dict]:
    """Retorna os top_k documentos mais similares à query."""
    qv = np.array(query_embedding, dtype=np.float32)
    scored = [
        (cosine_similarity(qv, doc["embedding"]), doc)
        for doc in corpus
        if doc["id"] != exclude_id and doc["answer"]  # só exemplos já resolvidos
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]



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


    print("  Gerando embeddings e populando banco…")
    question_ids: dict[str, int] = {}
    for i, row in enumerate(rows, 1):
        q = row[question_col].strip()
        existing_answer = row.get(ANSWER_COLUMN, "").strip() or None
        qid = upsert_question(conn, q, existing_answer)
        question_ids[q] = qid

        # Embedding apenas se ainda não existir
        cur = conn.execute("SELECT embedding FROM questions WHERE id = ?", (qid,))
        if cur.fetchone()[0] is None:
            emb = ollama_embed(q)
            store_embedding(conn, qid, emb)
        print(f"  [{i}/{len(rows)}] ✔ embedding armazenado", end="\r")

    print("\n Embeddings prontos.\n")

    # ── Fase 2: RAG — responder questões sem resposta ──
    corpus = get_all_with_embeddings(conn)
    to_answer = [r for r in rows if not r.get(ANSWER_COLUMN, "").strip()]

    if not to_answer:
        print("ℹ  Todas as questões já possuem resposta. Nada a fazer.")
    else:
        print(f"  Respondendo {len(to_answer)} questão(ões) com RAG…\n")

    for i, row in enumerate(rows, 1):
        q = row[question_col].strip()
        existing = row.get(ANSWER_COLUMN, "").strip()

        if existing:
            row[ANSWER_COLUMN] = existing
            print(f"  [{i}/{len(rows)}]  já respondida — pulando")
            continue

        qid = question_ids[q]
        cur = conn.execute("SELECT embedding FROM questions WHERE id = ?", (qid,))
        emb_json = cur.fetchone()[0]
        q_emb = json.loads(emb_json) if emb_json else ollama_embed(q)

        similar = retrieve_similar(q_emb, corpus, top_k=TOP_K, exclude_id=qid)
        prompt  = build_prompt(q, similar)

        print(f"  [{i}/{len(rows)}]  Resolvendo: {q[:60]}…")
        t0 = time.time()
        answer = ollama_chat(prompt)
        elapsed = time.time() - t0
        print(f"          ({elapsed:.1f}s) → {answer[:80]}")

        row[ANSWER_COLUMN] = answer
        upsert_question(conn, q, answer)

        # Atualiza corpus com a nova resposta para próximas iterações
        for doc in corpus:
            if doc["id"] == qid:
                doc["answer"] = answer

    conn.close()

    # ── Fase 3: salvar CSV com respostas ──
    output_path = path.with_stem(path.stem + "_respondido")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Concluído! CSV salvo em: {output_path}")
    print(f"  Banco SQLite: {DB_PATH}")



# ENTRY POINT


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rag_math.py <caminho_do_arquivo.csv>")
        print("Exemplo: python rag_math.py questoes.csv")
        sys.exit(0)

    check_ollama()
    process_csv(sys.argv[1])