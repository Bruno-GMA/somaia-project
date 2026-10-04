# Spec: Flag `--no-rag`

## Objetivo

Adicionar a flag `--no-rag` aos comandos `run` e `run_async` da CLI, permitindo executar o
pipeline de geração **sem** a etapa de recuperação de exemplos (retrieval). O modelo passa a
responder a questão diretamente, sem exemplos similares injetados no prompt.

O propósito é viabilizar **ablation study**: comparar o desempenho do pipeline com RAG
(linha de base atual) versus sem RAG, usando a mesma infraestrutura de ingest/embeddings.

## Escopo

- Desativa **apenas a etapa de retrieval** (consulta ao `VectorStore`/`HybridRetriever`).
- Mantém **intactos** ingest, embeddings, chunking, checkpoint e exportação. O banco vetorial
  continua sendo populado normalmente, permitindo rodar RAG posteriormente sobre o mesmo DB.
- O prompt de resposta passa a ser construído **sem exemplos** (branch "sem exemplos" do
  `build_prompt`).

## Fora de escopo

- Não desativa embeddings/BM25/chunking.
- Não altera o comportamento do comando `evaluate`.
- Não remove o banco vetorial nem os dados persistidos.

## Arquivos afetados

| Arquivo | Mudança |
|---------|---------|
| `src/config.py` | Adicionar campo `use_rag: bool = True` ao `Settings`. |
| `src/cli/main.py` | Adicionar flag `--no-rag` aos comandos `run` e `run_async`; propagar para `settings.use_rag`. |
| `src/pipeline/prompt.py` | `build_prompt(question, similar, use_rag=True)`: quando `use_rag=False`, ignora os exemplos. |
| `src/pipeline/orchestrator.py` | Em `_phase_answer`, se `not settings.use_rag`, pular retrieval e chamar `build_prompt(q.text, [])`. |
| `src/pipeline/async_orchestrator.py` | Em `_phase_answer_async`, se `not settings.use_rag`, pular o `asyncio.gather` de retrieval e chamar `build_prompt(q.text, [])`. |

## Interface / Assinaturas

### `config.py`

```python
class Settings(BaseSettings):
    use_rag: bool = True
```

### `cli/main.py`

```python
@app.command()
def run(
    ...,
    no_rag: bool = typer.Option(False, "--no-rag", help="Desativa retrieval (gera sem exemplos)"),
) -> None:
    ...
    settings.use_rag = not no_rag
```

Mesma flag em `run_async`.

### `prompt.py`

```python
def build_prompt(question: str, similar: list[RetrievedExample], use_rag: bool = True) -> str:
    if not use_rag:
        similar = []
    ...
```

## Comportamento esperado

- `somaia run data.csv --no-rag` gera respostas sem exemplos recuperados.
- A saída (CSV respondido, banco SQLite) é idêntica em estrutura ao modo normal.
- Sem a flag, o comportamento atual é preservado (default `use_rag=True`).

## Critérios de aceite

- A flag está disponível em `run` e `run_async` e documentada no `README`.
- Com `--no-rag`, o prompt enviado ao Ollama não contém a seção "Exemplos".
- Sem `--no-rag`, o pipeline se comporta exatamente como antes (regressão coberta por testes).

## Testes

- `tests/test_prompt.py`: `build_prompt` retorna prompt sem exemplos quando `use_rag=False`.
- Teste de integração (mock do retriever) garantindo que `retrieve` **não** é chamado com a
  flag ativa, e que `build_prompt` é chamado com lista vazia.