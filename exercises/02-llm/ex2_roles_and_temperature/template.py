"""Упражнение 02.2 — роли и temperature (LM Studio).

Опирается на §5 (temperature) и §8 (промпт по ролям system/user) теории 02.
Собираем `messages` с ролями и смотрим, как `temperature` влияет на
разброс ответов: при 0 модель детерминирована, при высокой — вариативна.

Требуется: LM Studio с загруженной CHAT-моделью.

Задача:
  1. ask: отправить messages с заданной temperature, вернуть текст ответа.
  2. В main: сравнить разброс при temperature=0 и temperature=1.
"""

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
    """Отправить system+user сообщения в чат-модель, вернуть ответ.

    Args:
        user_text: реплика пользователя.
        temperature: степень случайности сэмплирования: 0 почти
            детерминирован, 1 — заметный разброс формулировок.

    Returns:
        client.chat.completions.create(model=model, messages=[...],
        temperature=temperature).choices[0].message.content — текст
        первого варианта ответа (choices — список альтернатив, обычно
        из одного элемента). messages — список реплик диалога по ролям:
        "system" — инструкция модели (как себя вести), "user" — реплика
        пользователя, "assistant" — прошлый ответ модели (для истории).
    """
    # get_client() — настроенный клиент openai.OpenAI, направленный на
    # локальный сервер LM Studio (не проверяет доступность сразу).
    client = get_client()
    # first_model_id(client) — id первой загруженной в LM Studio модели
    # (заодно проверяет, что сервер отвечает).
    model = first_model_id(client)
    # TODO
    raise NotImplementedError


def main() -> None:
    question = "Придумай название для кофейни."
    try:
        # temperature=0 → детерминированно: два вызова должны совпасть
        a0 = ask(question, temperature=0.0)
        b0 = ask(question, temperature=0.0)
        # высокая temperature → разброс
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
    print("[OK] ex2: сравните детерминизм при 0 и разброс при 1.")


if __name__ == "__main__":
    main()
