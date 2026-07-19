"""Эталонное решение упражнения 03.2 — ReAct-цикл."""

from __future__ import annotations

import ast
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

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

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _calc(expr: str) -> str:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("bad expr")

    r = ev(ast.parse(expr, mode="eval").body)
    return str(int(r)) if r == int(r) else str(r)


def run_tool(action: str) -> str:
    m = re.match(r"\s*(\w+)\[(.*)\]\s*$", action)
    if not m:
        return f"Ошибка: не удалось разобрать действие '{action}'"
    name, arg = m.group(1), m.group(2).strip()
    if name == "lookup":
        return FACTS.get(arg.lower(), f"Нет данных о '{arg}'")
    if name == "calc":
        try:
            return _calc(arg)
        except Exception as exc:  # неверное выражение от модели
            return f"Ошибка вычисления: {exc}"
    return f"Неизвестный инструмент '{name}'"


def react(question: str, max_steps: int = 6) -> str:
    client = get_client()
    model = first_model_id(client)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, stop=["\n"]
        )
        line = (resp.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": line})

        if line.startswith("Final:"):
            return line[len("Final:"):].strip()

        if line.startswith("Action:"):
            action = line[len("Action:"):].strip()
            observation = run_tool(action)
            messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )
        # Thought: — просто продолжаем цикл; строка уже в контексте.

    return "не удалось получить Final за отведённые шаги"


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
        print("[OK] ex2_solution: инструменты run_tool корректны (calc/lookup).")
        return

    print(f"Финальный ответ агента: {answer}")
    assert "556" in answer, answer
    print("[OK] ex2_solution: агент прошёл цикл Thought→Action→Observation.")


if __name__ == "__main__":
    main()
