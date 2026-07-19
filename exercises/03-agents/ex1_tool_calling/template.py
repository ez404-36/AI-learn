"""Упражнение 03.1 — tool calling (LM Studio).

Опирается на §2 теории 03: LLM сама не выполняет код — она возвращает
структурированный запрос на вызов инструмента, а рантайм выполняет его и
кладёт результат обратно в контекст (сообщение с ролью `tool`).

Здесь мы описываем инструмент `calculator` по OpenAI-схеме, отправляем в
LM Studio, выполняем запрошенный вызов руками и возвращаем результат.

Требуется: LM Studio с моделью, поддерживающей function calling.

Задача:
  1. calculator: посчитать простое арифметическое выражение (реальный tool).
  2. TOOLS_SCHEMA: описать инструмент в формате OpenAI tools.
  3. run_with_tools: цикл «запрос → tool_call → выполнить → вернуть tool → финальный ответ».

Что нужно знать:
    1. Первый вызов chat.completions.create с параметром tools=TOOLS_SCHEMA
        сообщает модели, какие функции ей доступны: вместо обычного текста
        она может попросить их вызвать.
    2. message.tool_calls — список запросов на вызов функций, у каждого
        свои .id, .function.name, .function.arguments.
    3. Чтобы вернуть результат вызова модели, в messages добавляются два
        сообщения: {"role": "assistant", ..., "tool_calls": [...]} (то,
        что попросила модель) и {"role": "tool", "tool_call_id": tc.id,
        "content": результат} (ответ на конкретный вызов — id должен
        совпадать с id запроса).
    4. Второй вызов chat.completions.create без tools возвращает финальный
        текстовый ответ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)


def calculator(expression: str) -> str:
    """Безопасно посчитать арифметическое выражение вида '2 + 2 * 3'.

    Args:
        expression: строка с выражением (разрешены + - * / и скобки).

    Returns:
        Строка с результатом вычисления.
    """
    # TODO: разберите выражение БЕЗ eval (например через ast) и верните
    #       строку с результатом.
    raise NotImplementedError


# TODO: опишите инструмент calculator в формате OpenAI tools:
# список из одного dict {"type": "function", "function": {name, description,
# parameters: {...JSON Schema...}}}
# JSON Schema — стандартный формат описания структуры JSON-объекта
# (какие поля есть, какого они типа, какие обязательны); здесь им
# описываются аргументы функции, чтобы модель знала, что и как передавать.
TOOLS_SCHEMA: list[dict] = []


def run_with_tools(question: str) -> str:
    """Полный цикл tool calling: запрос → tool_call → выполнить → финальный ответ.

    Args:
        question: вопрос пользователя.

    Returns:
        Финальный текстовый ответ модели после выполнения всех запрошенных
        инструментов.
    """
    client = get_client()
    model = first_model_id(client)
    messages: list[dict] = [{"role": "user", "content": question}]
    # TODO
    raise NotImplementedError


def main() -> None:
    # Оффлайн-проверка самого инструмента и его схемы (без LLM):
    assert calculator("17 * 23 + 5") == "396"
    assert TOOLS_SCHEMA[0]["type"] == "function"
    assert TOOLS_SCHEMA[0]["function"]["name"] == "calculator"
    assert "expression" in TOOLS_SCHEMA[0]["function"]["parameters"]["properties"]

    try:
        answer = run_with_tools("Сколько будет 17 * 23 + 5? Ответь числом.")
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] LLM-часть пропущена: {exc}", file=sys.stderr)
        print("[OK] ex1_tool_calling: инструмент calculator корректен (17*23+5=396).")
        return

    print(f"Ответ модели: {answer}")
    assert "396" in answer, answer  # 17*23+5 = 396
    print("[OK] ex1_tool_calling: модель воспользовалась калькулятором.")


if __name__ == "__main__":
    main()
