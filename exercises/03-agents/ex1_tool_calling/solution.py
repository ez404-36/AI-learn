"""Эталонное решение упражнения 03.1 — tool calling.

Безопасный калькулятор реализован через `ast` (без `eval`), чтобы модель
не могла спровоцировать выполнение произвольного кода — это как раз тот
самый «рантайм контролирует выполнение», о котором говорит §2 теории 03.
"""

from __future__ import annotations

import ast
import json
import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("недопустимое выражение")


def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    # int-результат печатаем без .0
    return str(int(result)) if result == int(result) else str(result)


TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Вычисляет арифметическое выражение (+ - * / и скобки).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Выражение, например '17 * 23 + 5'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def run_with_tools(question: str) -> str:
    client = get_client()
    model = first_model_id(client)
    messages: list[dict] = [{"role": "user", "content": question}]

    first = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS_SCHEMA, temperature=0.0
    )
    msg = first.choices[0].message

    if not msg.tool_calls:
        return (msg.content or "").strip()

    # Ответ ассистента с запросом инструментов кладём обратно в контекст.
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
    )

    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = calculator(args["expression"])
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )

    final = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0
    )
    return (final.choices[0].message.content or "").strip()


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
        print("[OK] ex1_solution: инструмент calculator корректен (17*23+5=396).")
        return

    print(f"Ответ модели: {answer}")
    assert "396" in answer, answer
    print("[OK] ex1_solution: модель воспользовалась калькулятором.")


if __name__ == "__main__":
    main()
