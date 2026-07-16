"""Упражнение 07.1 — латентность vs throughput (LM Studio).

Опирается на §2 и §4 теории 07: латентность — задержка ОДНОГО запроса;
throughput — сколько запросов/токенов в секунду выдаёт система суммарно по
ВСЕМ пользователям. Их часто путают, а провайдеры называют «X ток/сек»,
не уточняя, о какой из величин речь.

Здесь мы измеряем:
  1. латентность и tokens/sec одного последовательного запроса;
  2. общий throughput при нескольких ПАРАЛЛЕЛЬНЫХ запросах
     (`concurrent.futures`) — и видим, что суммарный throughput выше, хотя
     каждый отдельный запрос может стать чуть медленнее.

Требуется: LM Studio с CHAT-моделью.

Задача:
  1. timed_completion: вернуть (текст, длительность_сек, число_токенов).
  2. sequential_throughput: N запросов подряд.
  3. parallel_throughput: N запросов параллельно (ThreadPoolExecutor).

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 07-inference/ex1_latency_vs_throughput/work.py
  4. Сверка:  07-inference/ex1_latency_vs_throughput/solution.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

PROMPT = "Перечисли 5 планет Солнечной системы одним предложением."


def timed_completion(prompt: str) -> tuple[str, float, int]:
    """Сделать один запрос, вернуть (текст, секунды, число_токенов_ответа)."""
    client = get_client()
    model = first_model_id(client)
    # TODO: замерить time.perf_counter() вокруг chat.completions.create;
    #       число токенов взять из resp.usage.completion_tokens
    raise NotImplementedError


def sequential_throughput(n: int) -> float:
    """N запросов ПОДРЯД. Вернуть суммарный throughput (токены/сек)."""
    # TODO: суммировать токены и время, вернуть tokens/total_time
    raise NotImplementedError


def parallel_throughput(n: int, workers: int = 4) -> float:
    """N запросов ПАРАЛЛЕЛЬНО. Вернуть суммарный throughput (токены/сек)."""
    # TODO: ThreadPoolExecutor; замерить общее время стены (wall-clock)
    #       вокруг всех запросов, поделить суммарные токены на него
    raise NotImplementedError


def main() -> None:
    try:
        text, secs, toks = timed_completion(PROMPT)
        seq = sequential_throughput(4)
        par = parallel_throughput(4)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Один запрос: {secs:.2f} c, {toks} токенов, "
          f"{toks / secs:.1f} ток/сек (латентность)")
    print(f"Throughput последовательно: {seq:.1f} ток/сек")
    print(f"Throughput параллельно:     {par:.1f} ток/сек")
    print("[OK] ex1: сравните латентность одного запроса и общий throughput.")


if __name__ == "__main__":
    main()
