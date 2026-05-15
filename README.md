> RAG de questões matemáticas conectado ao **Ollama local** — sem nuvem, sem API paga, 100% na sua máquina.

---

##  O que é o Sistema?

 É um pipeline de **Retrieval-Augmented Generation (RAG)** que:

1. Recebe um arquivo `.csv` com questões matemáticas
1.1 O Repositório vem com 3 `.csv` de exemplo deixe apenas um, ou adicione o seu
2. Armazena cada questão em um banco **SQLite** com seu respectivo **embedding vetorial**
3. Para cada questão sem resposta, busca exemplos similares já resolvidos (busca semântica por cosseno)
4. Monta um prompt contextualizado e envia ao **LLM local via Ollama**
5. Salva as respostas geradas de volta no `.csv`, na última coluna

Tudo isso **offline**, usando modelos que rodam diretamente na sua máquina.

---

##  Funcionalidades

| Funcionalidade | Descrição |
|---|---|
|  **Banco SQLite** | Questões, respostas e embeddings persistidos localmente |
|  **Embeddings semânticos** | Gerados via `nomic-embed-text` pelo Ollama |
|  **Busca por similaridade** | Cosine similarity para encontrar os exemplos mais relevantes |
|  **Geração com RAG** | Prompt enriquecido com exemplos similares antes de chamar o LLM |
|  **Idepotente** | Questões já respondidas são puladas em execuções futuras |
|  **Compatibilidade Ollama** | Suporte automático a `/api/chat` e fallback para `/api/generate` |
|  **CSV de saída** | Gera `<nome>_respondido.csv` com coluna `resposta` preenchida |
|  **Detecção automática** | Identifica a coluna de questões automaticamente pelo nome |

---

##  Estrutura do projeto

```
SOMAIA-PROJECT/
├── rag_math.py            # Script principal
├── requirements.txt       # Dependências Python
├── questoes_exemplo.csv   # CSV de exemplo
├── math_rag.db            # Banco SQLite (gerado na primeira execução)
└── README.md
```

---

##  Formato do CSV de entrada

A coluna de questões pode ter qualquer um desses nomes (case-insensitive):

```
operacao | operação | operation | pergunta | problema | enunciado
```

Se nenhum for encontrado, a **primeira coluna** é usada automaticamente.

Questões que já possuírem conteúdo na coluna `resposta` são **ignoradas** — o script não regera respostas desnecessariamente.

**Exemplo mínimo:**

```csv
questao
"Quanto é 2 + 2?"
"Resolva: 3x + 6 = 21"
"Qual é a raiz quadrada de 144?"
```

---

## Pré-requisitos

- Python **3.11+**
- [Ollama](https://ollama.com) instalado e rodando
- Modelos No ollama e `nomic-embed-text` baixados

---

##  Como rodar

### 1. Acesse diretório do projeto

```bash
cd localmind
```
### 2. Inicie o Ollama e baixe os modelos

```bash
# Em outro terminal, Iniciar o servidor Ollama
ollama serve
# Baixar modelos necessários
ollama pull gemma2:2b #pode ser outro SLM
ollama pull nomic-embed-text
```

### 3. Instale as dependências

```bash
# Volute no mesmo terminal do diretório localmind
python -m venv venv
.\venv\Scripts\activate (windows)
source venv/bin/activate (linux debian)
pip install -r requirements.txt
```

### 4. Execute o LocalMind

```bash
python main.py dataset_teste.csv
```

O arquivo de saída será salvo automaticamente como `questoes_respondido.csv`.

---

##  Configurações

Edite as constantes no topo do `rag_math.py` conforme necessário:

```python
OLLAMA_BASE_URL = "http://localhost:11434"  # URL do Ollama
CHAT_MODEL      = "llama3"                  # Modelo baixado
EMBED_MODEL     = "nomic-embed-text"        # Modelo de embeddings
DB_PATH         = "math_rag.db"             # Caminho do banco SQLite
ANSWER_COLUMN   = "resposta"                # Nome da coluna de saída no CSV
TOP_K           = 3                         # Nº de exemplos similares no contexto
```

> Você pode trocar `gemma2` por qualquer outro modelo disponível no Ollama, como `mistral`, `gemma3` ou `phi3`.

---

##  Fluxo interno

```
CSV de entrada
     │
     ▼
Inserir questões no SQLite
     │
     ▼
Gerar embeddings (nomic-embed-text)
     │
     ▼
Para cada questão sem resposta:
  ├─ Buscar TOP_K mais similares (cosine similarity)
  ├─ Montar prompt RAG com exemplos
  └─ Chamar SLM via Ollama
     │
     ▼
Salvar respostas no SQLite e no CSV
     │
     ▼
questoes_respondido.csv
```

---


> O SQLite já vem embutido no Python — nenhuma instalação extra necessária.

---

##  Solução de problemas

| Erro | Causa provável | Solução |
|---|---|---|
| ` Ollama não está acessível` | Servidor não iniciado | Execute `ollama serve` |
| `405 Method Not Allowed` | Versão antiga do Ollama | O script faz fallback automático; ou rode `ollama update` |
| `Model not found` | Modelo não baixado | Execute `ollama pull <modelo>` |
| `timeout` | Modelo muito pesado | Aumente o `timeout` nas chamadas ou use um modelo menor |



<p align="center">Feito para pesquisa!</p>