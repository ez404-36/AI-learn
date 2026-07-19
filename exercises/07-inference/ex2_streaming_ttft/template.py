"""Упражнение 07.2 — стриминг и TTFT (LM Studio).

Опирается на §5 теории 07: стриминг не делает модель быстрее (общее время
генерации то же), но резко улучшает ВОСПРИНИМАЕМУЮ скорость — первые слова
появляются почти сразу (после TTFT), и пользователь начинает читать, пока
хвост ещё генерируется.

Здесь мы сравниваем два вызова к одной модели:
  - нестриминг: время до появления ЛЮБОГО текста = всё время генерации;
  - стриминг: измеряем TTFT (время до первого чанка) отдельно от общего.

Требуется: LM Studio с CHAT-моделью (streaming endpoint).

Задача:
  1. blocking_call: обычный вызов, вернуть (текст, total_sec).
  2. streaming_call: stream=True, вернуть (текст, ttft_sec, total_sec).

Что нужно знать:
    1. С параметром stream=True chat.completions.create возвращает не
        один ответ, а итерируемый объект: генерация приходит частями
        (чанками) по мере готовности, а не целиком в конце.
    2. В цикле `for chunk in stream:` каждый chunk.choices[0].delta — это
        "приращение" ответа (может быть пустым на служебных чанках);
        delta.content — очередной кусочек текста (или None).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

PROMPT = "Расскажи короткую историю про кота-программиста (3-4 предложения)."


def blocking_call(prompt: str) -> tuple[str, float]:
    """Нестриминговый вызов.

    Args:
        prompt: текст запроса к модели.

    Returns:
        Кортеж (текст, total_sec). Замерьте время всего вызова
        chat.completions.create (без stream).
    """
    client = get_client()
    model = first_model_id(client)
    # TODO
    raise NotImplementedError


def streaming_call(prompt: str) -> tuple[str, float, float]:
    """Стриминговый вызов.

    TTFT — время до ПЕРВОГО непустого чанка контента.

    Args:
        prompt: текст запроса к модели.

    Returns:
        Кортеж (текст, ttft_sec, total_sec), где ttft — время до первого
        непустого чанка контента.
    """
    client = get_client()
    model = first_model_id(client)
    # TODO
    raise NotImplementedError


def main() -> None:
    try:
        block_text, block_total = blocking_call(PROMPT)
        _, ttft, stream_total = streaming_call(PROMPT)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Нестриминг: первый текст через {block_total:.2f} c (= всё время)")
    print(f"Стриминг:   TTFT {ttft:.2f} c, полный ответ {stream_total:.2f} c")

    # blocking_call должен вернуть непустой текст и положительное время.
    assert block_text.strip(), "пустой текст ответа"
    assert block_total > 0, block_total

    # TTFT заметно меньше общего времени — вот выигрыш в воспринимаемой скорости.
    assert ttft < stream_total, (ttft, stream_total)
    print("[OK] ex2: стриминг отдаёт первый токен задолго до конца генерации.")


if __name__ == "__main__":
    main()
