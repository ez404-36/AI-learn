"""Эталонное решение упражнения 05.2 — граф с условным ребром на LangGraph."""

from __future__ import annotations

import ast
import operator
import re
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    BASE_URL,
    LMStudioUnavailableError,
    first_model_id,
)

SYSTEM = (
    "Ты решаешь арифметику пошагово. Если нужно вычислить — выведи РОВНО "
    "строку 'Action: calc[<выражение>]'. Когда знаешь ответ — выведи "
    "'Final: <число>'. Ничего лишнего."
)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def calc(expr: str) -> str:
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("bad")
    r = ev(ast.parse(expr, mode="eval").body)
    return str(int(r)) if r == int(r) else str(r)


class AgentState(TypedDict):
    question: str
    scratch: list[str]
    last: str
    answer: str


def make_model():
    from langchain_openai import ChatOpenAI

    model_id = first_model_id()
    return ChatOpenAI(base_url=BASE_URL, api_key="lm-studio",
                      model=model_id, temperature=0, stop=["\n"])


def think(state: AgentState) -> AgentState:
    from langchain_core.messages import HumanMessage, SystemMessage

    model = make_model()
    convo = "\n".join(state["scratch"])
    user = f"Вопрос: {state['question']}\n{convo}".strip()
    resp = model.invoke([SystemMessage(SYSTEM), HumanMessage(user)])
    line = str(resp.content).strip().splitlines()[0] if resp.content else ""
    state["last"] = line
    return state


def tool(state: AgentState) -> AgentState:
    m = re.search(r"calc\[(.*?)\]", state["last"])
    obs = calc(m.group(1)) if m else "Ошибка"
    state["scratch"].append(state["last"])
    state["scratch"].append(f"Observation: {obs}")
    return state


def route(state: AgentState) -> str:
    from langgraph.graph import END

    if "Action:" in state["last"]:
        return "tool"
    # финальная ветка: вынуть число/текст после Final:
    last = state["last"]
    state["answer"] = last.split("Final:", 1)[-1].strip() if "Final:" in last else last
    return END


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("tool", tool)
    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", route, {"tool": "tool", END: END})
    graph.add_edge("tool", "think")
    return graph.compile()


def main() -> None:
    # Оффлайн-проверка инструмента:
    assert calc("(12 + 8) * 3") == "60"

    try:
        app = build_graph()
        result = app.invoke(
            {"question": "Сколько будет (12 + 8) * 3?",
             "scratch": [], "last": "", "answer": ""},
            {"recursion_limit": 20},
        )
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] LLM-часть пропущена: {exc}", file=sys.stderr)
        print("[OK] ex2_solution: инструмент calc и структура графа корректны.")
        return

    print("Трасса:", *result["scratch"], sep="\n  ")
    print(f"Ответ: {result['answer']}")
    assert "60" in result["answer"], result["answer"]
    print("[OK] ex2_solution: граф с условным ребром решил задачу.")


if __name__ == "__main__":
    main()
