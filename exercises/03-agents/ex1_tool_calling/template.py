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

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 03-agents/ex1_tool_calling/work.py
  4. Сверка:  03-agents/ex1_tool_calling/solution.py
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
    """Безопасно посчитать арифметическое выражение вида '2 + 2 * 3'."""
    # TODO: разберите выражение БЕЗ eval (например через ast) и верните
    #       строку с результатом. Разрешены + - * / и скобки.
    raise NotImplementedError


# TODO: опишите инструмент calculator в формате OpenAI tools:
# список из одного dict {"type": "function", "function": {name, description,
# parameters: {...JSON Schema...}}}
TOOLS_SCHEMA: list[dict] = []


def run_with_tools(question: str) -> str:
    """Полный цикл tool calling, вернуть финальный текстовый ответ."""
    client = get_client()
    model = first_model_id(client)
    messages: list[dict] = [{"role": "user", "content": question}]
    # TODO:
    #  1. Первый вызов chat.completions.create с tools=TOOLS_SCHEMA.
    #  2. Если модель вернула tool_calls — для каждого выполнить calculator,
    #     добавить в messages сообщение assistant с tool_calls и сообщения
    #     с role="tool" (tool_call_id + результат).
    #  3. Второй вызов без tools — получить финальный ответ.
    raise NotImplementedError


def main() -> None:
    try:
        answer = run_with_tools("Сколько будет 17 * 23 + 5? Ответь числом.")
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Ответ модели: {answer}")
    assert "396" in answer, answer  # 17*23+5 = 396
    print("[OK] ex1_tool_calling: модель воспользовалась калькулятором.")


if __name__ == "__main__":
    main()
