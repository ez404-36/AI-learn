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

Что нужно знать:
    1. time.perf_counter() — точные "часы" для замера длительности:
        вызвать до и после операции, разница = время выполнения в
        секундах (не привязаны к календарному времени, только для
        интервалов).
    2. resp.usage.completion_tokens — сервер сам возвращает статистику
        по токенам ответа в поле .usage объекта, полученного от
        chat.completions.create.
    3. ThreadPoolExecutor(max_workers=workers) — пул потоков: позволяет
        выполнить несколько блокирующих сетевых запросов одновременно
        (используется как контекстный менеджер); метод .map(fn, iterable)
        запускает fn для каждого элемента параллельно и собирает
        результаты в исходном порядке.
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
    """Сделать один запрос и замерить его длительность.

    Args:
        prompt: текст запроса к модели.

    Returns:
        Кортеж (текст, секунды, число_токенов_ответа).
    """
    client = get_client()
    model = first_model_id(client)
    # TODO
    raise NotImplementedError


def sequential_throughput(n: int) -> float:
    """N запросов ПОДРЯД.

    Args:
        n: сколько запросов сделать.

    Returns:
        Суммарный throughput (токены/сек): суммировать токены и время,
        вернуть tokens/total_time.
    """
    # TODO
    raise NotImplementedError


def parallel_throughput(n: int, workers: int = 4) -> float:
    """N запросов ПАРАЛЛЕЛЬНО.

    Args:
        n: сколько запросов сделать.
        workers: размер пула потоков (сколько запросов идёт одновременно).

    Returns:
        Суммарный throughput (токены/сек) при выполнении n запросов
        параллельно с помощью пула из workers потоков.
    """
    # TODO
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

    assert text.strip(), "пустой текст ответа"
    assert secs > 0, secs
    assert toks > 0, toks
    assert seq > 0, seq
    assert par > 0, par
    print("[OK] ex1: сравните латентность одного запроса и общий throughput.")


if __name__ == "__main__":
    main()
