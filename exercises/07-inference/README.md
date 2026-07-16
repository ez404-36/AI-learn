# Практика 07 — Инференс и производительность

Опирается на теорию: [07-inference](../../07-inference/README.md).
**Требуется LM Studio** (chat-модель, streaming endpoint).

## Упражнения

### ex1_latency_vs_throughput — латентность vs throughput

Измеряете латентность и tokens/sec одного запроса, затем общий throughput
при нескольких **параллельных** запросах (`concurrent.futures`). Видите на
практике конфликт из §2: латентность одного запроса ≠ суммарный throughput
(суть continuous batching, §4).

```bash
uv run python 07-inference/ex1_latency_vs_throughput/work.py
```

### ex2_streaming_ttft — стриминг и TTFT

Сравниваете нестриминговый вызов и стриминговый (`stream=True`), измеряя
**TTFT** (время до первого токена) отдельно от общего времени (§5).
Стриминг не ускоряет генерацию, но резко снижает воспринимаемую задержку.

```bash
uv run python 07-inference/ex2_streaming_ttft/work.py
```

> Как запускать упражнения (`setup_practice.py`, `work.py`, `solution.py`) —
> см. [общий README практики](../README.md#3-конвенция-файлов).

> Метрики зависят от вашего железа и модели — важны не абсолютные числа, а
> соотношения (латентность vs throughput, TTFT vs полное время).
