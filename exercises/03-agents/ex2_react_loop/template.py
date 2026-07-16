"""Упражнение 03.2 — ReAct-цикл (LM Studio).

Опирается на §3 теории 03: паттерн ReAct = цикл Thought → Action →
Observation, который «крутит» рантайм агента, пока задача не решена.

Здесь мы реализуем ReAct вручную поверх LM Studio: модель в текстовом
формате пишет Thought и Action, наш рантайм парсит Action, выполняет
инструмент, возвращает Observation обратно в контекст и повторяет.

Инструменты: `lookup` (заглушка «база фактов») и `calc` (калькулятор).
Задача-пример требует двух шагов: узнать два факта → вычесть.

Требуется: LM Studio с инструктивной chat-моделью.

Задача:
  1. run_tool: выполнить действие вида 'calc[1703-1147]' / 'lookup[Москва]'.
  2. react: цикл, парсящий ответ модели, до строки 'Final: ...'.
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

# Мини-"база знаний" для инструмента lookup (заглушка вместо реального поиска).
FACTS = {
    "москва": "Москва основана в 1147 году.",
    "санкт-петербург": "Санкт-Петербург основан в 1703 году.",
}

SYSTEM_PROMPT = """Ты — агент, работающий по протоколу ReAct.
Доступные инструменты:
  lookup[запрос]  — вернуть факт о городе из базы.
  calc[выражение] — посчитать арифметику, например calc[1703-1147].
На каждом шаге выводи РОВНО одну строку одного из видов:
  Thought: <рассуждение>
  Action: lookup[...]   или   Action: calc[...]
Когда знаешь ответ, выведи строку:
  Final: <число или короткий ответ>
Не выводи ничего лишнего."""


def run_tool(action: str) -> str:
    """Выполнить действие 'name[arg]' и вернуть Observation.

    Args:
        action: строка вида 'calc[1+2]' или 'lookup[Москва]'.

    Returns:
        Текст-наблюдение (Observation): распарсить name и arg из строки;
        для 'lookup' искать в FACTS (по нижнему регистру), для 'calc' —
        посчитать арифметику.
    """
    # TODO
    raise NotImplementedError


def react(question: str, max_steps: int = 6) -> str:
    """ReAct-цикл: Thought → Action → Observation, до финального ответа.

    Args:
        question: вопрос пользователя.
        max_steps: максимум шагов цикла до принудительной остановки.

    Returns:
        Финальный ответ (то, что после 'Final:'). До max_steps раз:
          - вызвать модель (temperature=0), взять первую строку ответа;
          - если строка начинается с 'Final:' → вернуть остаток;
          - если 'Action:' → выполнить run_tool, добавить в messages ответ
            ассистента и Observation как user-сообщение;
          - если 'Thought:' → просто добавить в контекст и продолжить.
    """
    client = get_client()
    model = first_model_id(client)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # TODO
    raise NotImplementedError


def main() -> None:
    # Оффлайн-проверка инструментов без LLM:
    assert run_tool("calc[1703-1147]") == "556"
    assert "1147" in run_tool("lookup[Москва]")

    question = (
        "На сколько лет Санкт-Петербург моложе Москвы? "
        "Используй инструменты, ответь числом."
    )
    try:
        answer = react(question)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] LLM-часть пропущена: {exc}", file=sys.stderr)
        print("[OK] ex2_react_loop: инструменты run_tool корректны (calc/lookup).")
        return

    print(f"Финальный ответ агента: {answer}")
    assert "556" in answer, answer  # 1703 - 1147 = 556
    print("[OK] ex2_react_loop: агент прошёл цикл Thought→Action→Observation.")


if __name__ == "__main__":
    main()
