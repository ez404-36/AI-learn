"""Эталонное решение упражнения 02.2 — роли и temperature."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

SYSTEM_PROMPT = "Ты — краткий ассистент. Отвечай одним коротким предложением."


def ask(user_text: str, temperature: float) -> str:
    client = get_client()
    model = first_model_id(client)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    question = "Придумай название для кофейни."
    try:
        a0 = ask(question, temperature=0.0)
        b0 = ask(question, temperature=0.0)
        samples = [ask(question, temperature=1.0) for _ in range(3)]
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"temperature=0: '{a0}' / '{b0}'")
    print("temperature=1:")
    for s in samples:
        print(f"  - {s}")

    # ask() должен возвращать непустой текст ответа при любой temperature.
    assert a0.strip(), "пустой ответ при temperature=0"
    assert b0.strip(), "пустой ответ при temperature=0"
    assert all(s.strip() for s in samples), "пустой ответ при temperature=1"

    # Наблюдение (не жёсткий assert — модели различаются):
    # при temperature=0 ответы обычно совпадают, при 1 — чаще различаются.
    unique = len(set(samples))
    print(
        f"[OK] ex2_solution: уникальных ответов при temperature=1: "
        f"{unique}/{len(samples)}."
    )


if __name__ == "__main__":
    main()
