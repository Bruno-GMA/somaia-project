# Métricas de recuperação — schema v2

## Definições e compatibilidade

Esta versão substitui o contrato anterior. Não compare diretamente recall/nDCG de
artefatos sem `schema_version: 2` com os novos resultados: o recall antigo do SOMAIA
usava o denominador de precision e seu IDCG considerava apenas o ranking recuperado.

- `precision_at_k`: relevantes únicos recuperados / k, mesmo em listas curtas.
- `recall_at_k`: relevantes recuperados / total de relevantes elegíveis conhecidos.
- `mrr_at_k`: 1 / posição do primeiro relevante no top-k; a agregação é a média.
- `ndcg_at_k`: DCG / IDCG, ganho `2^grau - 1`, desconto `log2(rank + 1)`;
  IDCG usa os maiores graus do corpus elegível, inclusive documentos não recuperados.
- `hit_at_k`: 1 quando algum relevante aparece no top-k, senão 0.

Graus são inteiros não negativos; qualquer grau positivo é relevante para as métricas
binárias. O cálculo exige `k > 0`, IDs únicos, válidos e pertencentes ao corpus.
O conjunto recuperado nunca é usado para inferir o denominador do recall.

## Contrato de qrels

O argumento `--qrels` recebe JSON UTF-8; IDs de consulta e documento são strings:

```json
{
  "schema_version": 1,
  "source": "anotacao_humana",
  "version": "v1",
  "corpus_id": "corpus-avaliacao-v1",
  "qrels": {
    "consulta_1": {"documento_a": 2, "documento_b": 1, "documento_c": 0},
    "consulta_sem_relevantes": {}
  }
}
```

`source`, `version` e `corpus_id` são obrigatórios. `corpus_id` é a versão declarada
pelo autor; o relatório também inclui SHA-256 dos IDs elegíveis, que não substitui
um hash dos textos/modelos. IDs anotados fora do corpus são rejeitados. Chaves JSON
duplicadas, graus negativos/fracionários/booleanos e IDs inválidos são rejeitados.

Consulta ausente em `qrels` é `unjudged`: todas as métricas ficam `null` e não entram
na média. Consulta explicitamente presente sem relevantes conhecidos é `no_relevant`:
métricas zero, com status e cobertura registrados. Um documento não julgado recebe
`null` em `relevance_grades` e ganho zero na computação; isso **não prova irrelevância**.
Com anotações incompletas, recall e IDCG se referem apenas à relevância conhecida.
Reporte `judgment_coverage`, `n_judged` e `n_unjudged_retrieved` junto às métricas.

## Artefatos e interpretação

Cada consulta registra `query_id`, `k`, fonte/versão da relevância, corpus declarado,
IDs recuperados, graus, total elegível/relevante, exclusões, cobertura e métricas.
Agregados são médias macro com `valid_counts`, `status_counts`, fontes, versões e k.
Sem observações válidas, a média é `null`, não zero. Os artefatos históricos são
preservados; novos experimentos usam novos caminhos.

Separe consultas e corpus, documentando perguntas idênticas ou parafraseadas e o
que se pretende medir. Recuperar o ID esperado de uma pergunta já indexada mede
localização, não necessariamente generalização nem utilidade pedagógica. Não derive
qrels das respostas do próprio sistema. Cobertura total de um adaptador automático
não equivale a cobertura de anotação humana.

## Verificação

O caso canônico possui cinco documentos relevantes elegíveis e um irrelevante. Um
ranking top-2 com apenas um relevante deve produzir precision=0,5 e recall=0,2.
Testes adicionais cobrem relevância graduada, relevante ideal não recuperado, ordem,
lista curta/vazia, nenhuma relevância, qrels incompletos, exclusão da própria consulta,
IDs inválidos/duplicados e agregação sem observações.

## Integração SOMAIA

`QuestionRepository.get_retrieval_corpus()` retorna perguntas com `answer IS NOT NULL
AND answer != ''`, o mesmo universo elegível do ramo BM25 do retriever híbrido.
A consulta é excluída tanto do ranking quanto dos denominadores. Documentos sem
embedding ainda podem ser elegíveis pelo ramo lexical.

Sem `--qrels`, o adaptador usa pseudo-relevância Level+Type (grau 1), com operation
igual elevando o grau a 2. A fonte é `pseudo_relevance_level_type_operation`; isso
mede concordância de metadados, não relevância humana. Com `--qrels`, as chaves de
consulta/documento são os IDs da tabela `questions` convertidos para strings.

A API `evaluate_retrieval(query, retrieved, k, *, corpus, qrels=None)` exige o corpus
completo. A saída `retriever_distances` preserva as distâncias originais: valores
híbridos podem incluir score BM25 negativo e não são rotulados como cosseno ou RRF.

```bash
# Execute na raiz somaia-project; requer Ollama para embeddings.
python -m src.cli.main evaluate-retrieval math_rag.db -k 3
python -m src.cli.main evaluate-retrieval math_rag.db -k 3 --qrels qrels.json -o experimento-v2.json
python -m pytest tests/test_ranking.py tests/test_retrieval_metrics.py tests/test_retrieval_cli.py -q
```

O caminho padrão é `retrieval_metrics_v2_<timestamp>.json`. Um `--output` já existente
é rejeitado antes de chamar o modelo. As consultas usam um único event loop e os
recursos são fechados mesmo quando a avaliação falha. Os testes CLI usam SQLite e
sqlite-vec reais, substituindo apenas a chamada de embeddings; não validam a
qualidade dos embeddings do modelo real.
