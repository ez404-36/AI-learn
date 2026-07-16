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

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 07-inference/ex2_streaming_ttft/work.py
  4. Сверка:  07-inference/ex2_streaming_ttft/solution.py
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
    """Нестриминговый вызов. Вернуть (текст, total_sec)."""
    client = get_client()
    model = first_model_id(client)
    # TODO: замерить время всего вызова chat.completions.create (без stream)
    raise NotImplementedError


def streaming_call(prompt: str) -> tuple[str, float, float]:
    """Стриминговый вызов. Вернуть (текст, ttft_sec, total_sec).

    TTFT — время до ПЕРВОГО непустого чанка контента.
    """
    client = get_client()
    model = first_model_id(client)
    # TODO: stream=True; в цикле по чанкам зафиксировать момент первого
    #       непустого delta.content как ttft, собрать полный текст,
    #       по завершении — total
    raise NotImplementedError


def main() -> None:
    try:
        _, block_total = blocking_call(PROMPT)
        _, ttft, stream_total = streaming_call(PROMPT)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Нестриминг: первый текст через {block_total:.2f} c (= всё время)")
    print(f"Стриминг:   TTFT {ttft:.2f} c, полный ответ {stream_total:.2f} c")
    # TTFT заметно меньше общего времени — вот выигрыш в воспринимаемой скорости.
    assert ttft < stream_total, (ttft, stream_total)
    print("[OK] ex2: стриминг отдаёт первый токен задолго до конца генерации.")


if __name__ == "__main__":
    main()
