"""Упражнение 05.2 — граф с условным ребром на LangGraph (LM Studio).

Опирается на §4 теории 05: в LangGraph приложение — это граф состояний с
узлами (шаги), рёбрами (переходы) и условными рёбрами (ветвление). Так
удобно реализовать ReAct-цикл: узел «думать» → узел «инструмент» → ребро
назад к «думать», пока задача не решена.

Здесь граф решает арифметический вопрос: узел `think` просит модель либо
вызвать инструмент `calc`, либо дать финальный ответ; условное ребро
направляет поток в узел `tool` или в конец.

Требуется: LM Studio с CHAT-моделью.

Задача:
  1. think: вызвать модель, записать её строку в state.
  2. route: условное ребро — вернуть "tool" или END.
  3. build_graph: собрать StateGraph с узлами и условным ребром.
"""

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
    scratch: list[str]  # накопленный диалог (Action/Observation)
    last: str           # последняя строка модели
    answer: str


def make_model():
    from langchain_openai import ChatOpenAI

    model_id = first_model_id()
    return ChatOpenAI(base_url=BASE_URL, api_key="lm-studio",
                      model=model_id, temperature=0, stop=["\n"])


def think(state: AgentState) -> AgentState:
    """Узел: спросить модель, что делать дальше; записать строку в state."""
    from langchain_core.messages import HumanMessage, SystemMessage

    model = make_model()
    convo = "\n".join(state["scratch"])
    user = f"Вопрос: {state['question']}\n{convo}".strip()
    # TODO: model.invoke([SystemMessage(SYSTEM), HumanMessage(user)]),
    #       положить .content (первую строку) в state["last"], вернуть state
    raise NotImplementedError


def tool(state: AgentState) -> AgentState:
    """Узел: выполнить calc[...] из последней строки, добавить Observation."""
    m = re.search(r"calc\[(.*?)\]", state["last"])
    obs = calc(m.group(1)) if m else "Ошибка"
    state["scratch"].append(state["last"])
    state["scratch"].append(f"Observation: {obs}")
    return state


def route(state: AgentState) -> str:
    """Условное ребро: 'tool' если модель просит Action, иначе END."""
    from langgraph.graph import END

    # TODO: вернуть "tool" если в state["last"] есть "Action:",
    #       иначе записать финальный ответ в state["answer"] и вернуть END
    raise NotImplementedError


def build_graph():
    """Собрать граф: think → (route) → tool → think, либо END."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    # TODO: add_node("think", think); add_node("tool", tool);
    #       add_edge(START, "think");
    #       add_conditional_edges("think", route, {"tool": "tool", END: END});
    #       add_edge("tool", "think"); return graph.compile()
    raise NotImplementedError


def main() -> None:
    try:
        app = build_graph()
        result = app.invoke(
            {"question": "Сколько будет (12 + 8) * 3?",
             "scratch": [], "last": "", "answer": ""},
            {"recursion_limit": 20},
        )
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print("Трасса:", *result["scratch"], sep="\n  ")
    print(f"Ответ: {result['answer']}")
    assert "60" in result["answer"], result["answer"]  # (12+8)*3 = 60
    print("[OK] ex2_langgraph_agent: граф с условным ребром решил задачу.")


if __name__ == "__main__":
    main()
