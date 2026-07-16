"""Эталонное решение упражнения 07.2 — стриминг и TTFT."""

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
    client = get_client()
    model = first_model_id(client)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    total = time.perf_counter() - start
    return (resp.choices[0].message.content or "").strip(), total


def streaming_call(prompt: str) -> tuple[str, float, float]:
    client = get_client()
    model = first_model_id(client)
    start = time.perf_counter()
    ttft = -1.0
    chunks: list[str] = []
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft < 0:
                ttft = time.perf_counter() - start  # первый непустой чанк
            chunks.append(delta)
    total = time.perf_counter() - start
    if ttft < 0:
        ttft = total
    return "".join(chunks).strip(), ttft, total


def main() -> None:
    try:
        block_text, block_total = blocking_call(PROMPT)
        text, ttft, stream_total = streaming_call(PROMPT)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Ответ (стриминг): {text}")
    print(f"\nНестриминг: первый текст через {block_total:.2f} c (= всё время)")
    print(f"Стриминг:   TTFT {ttft:.2f} c, полный ответ {stream_total:.2f} c")
    print(
        f"Воспринимаемое ускорение: пользователь видит первые слова в "
        f"{stream_total / ttft:.1f}x раньше конца генерации."
    )

    # blocking_call должен вернуть непустой текст и положительное время.
    assert block_text.strip(), "пустой текст ответа"
    assert block_total > 0, block_total

    assert ttft < stream_total, (ttft, stream_total)
    print("[OK] ex2_solution: стриминг не меняет общее время, но снижает TTFT.")


if __name__ == "__main__":
    main()
