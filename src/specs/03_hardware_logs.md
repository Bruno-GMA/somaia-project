# Spec: Logs de Hardware/Energia

> Atualização schema v2: o contrato vigente separa processo Python, sistema e
> processo Ollama. CPU/RAM são recursos observados, não energia. Ver
> `../../specs/evidencias/04-correcao-matematica-hardware.md`.

## Objetivo

Instrumentar o pipeline para capturar e persistir métricas de **consumo de hardware**
(CPU, memória RAM, threads, latência) usando a biblioteca `psutil`. O objetivo é evidenciar o
perfil de recursos da aplicação, justificando futuramente a viabilidade de portar a arquitetura
para um ambiente **mobile** (recursos limitados).

## Contexto / Decisões

- **Escopo de medição**: CPU, memória residente (RSS), número de threads do processo, duração.
- **Sem medição de energia explícita (RAPL/etc.)**: a leitura de energia via
  `/sys/class/powercap` (Intel RAPL) não é garantida em todos os sistemas. A medição baseia-se em
  `psutil` (portátil). Isso é documentado como limitação.

## Escopo

- Novo módulo de monitoramento (`HardwareMonitor`).
- Integração nos dois orquestradores (sync e async).
- Saída persistida em **`hardware_log.json`** (snapshots com timestamps) além de resumo em stdout.

## Fora de escopo

- Medição de emissões de CO2 (ex.: codecarbon).
- Telemetria remota / envio a serviço externo.

## Arquivos afetados

| Arquivo | Mudança |
|---------|---------|
| `src/utils/hardware_monitor.py` | **Novo**: classe `HardwareMonitor`. |
| `src/pipeline/orchestrator.py` | Instanciar monitor, coletar snapshots, gravar JSON e imprimir resumo ao final. |
| `src/pipeline/async_orchestrator.py` | Idem ao sync. |
| `src/pipeline/async_orchestrator.py` (`PipelineMetrics`) | Incluir/carregar dados do monitor no `summary()`. |
| `requirements.txt` / `pyproject.toml` | Adicionar `psutil`. |

## Interface / Assinaturas

```python
class HardwareMonitor:
    def __init__(self) -> None: ...
    def start(self) -> None: ...
    def snapshot(self) -> dict: ...
    def summarize(self) -> dict: ...
    def format_report(self) -> str: ...
    def save_json(self, path: str) -> None: ...
```

Campos de um snapshot:

```python
{
    "timestamp": 1694457600.0,
    "cpu_percent": 42.5,          # CPU total do processo/sistema
    "rss_mb": 512.3,              # memória residente (MB)
    "num_threads": 8,             # threads do processo
    "elapsed_s": 12.4,            # desde o start
}
```

## Comportamento esperado

- O monitor é iniciado no começo de `run`/`run_async`.
- Snapshots são coletados por fase do pipeline (ingest, retrieval, geração).
- Ao final, `hardware_log.json` contém a lista de snapshots e um resumo (média/pico de CPU, RAM pico, threads máximas, duração total).

## Saída JSON (esqueleto)

```json
{
  "summary": {
    "cpu_percent_mean": 38.2,
    "cpu_percent_peak": 91.0,
    "rss_mb_peak": 780.4,
    "num_threads_peak": 12,
    "total_elapsed_s": 214.7
  },
  "snapshots": [
    { "timestamp": 1694457600.0, "cpu_percent": 42.5, "rss_mb": 512.3, "num_threads": 8 }
  ]
}
```

## Critérios de aceite

- `hardware_log.json` é gerado ao final do pipeline (sync e async).
- O resumo de hardware aparece no output final.
- `psutil` está declarado em `requirements.txt` e `pyproject.toml`.
- Métricas são plausíveis e não estouram (valores negativos/NaN tratados).

## Testes

- `tests/test_hardware_monitor.py`: usa `pytest-mock` para mockar `psutil` e validar
  `snapshot`/`summarize`/`format_report` sem depender do hardware real.
- Valida que `save_json` escreve um arquivo JSON parseável.
