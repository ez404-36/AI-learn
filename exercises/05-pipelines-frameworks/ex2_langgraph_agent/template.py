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

Что нужно знать:
    1. SystemMessage/HumanMessage — обёртки LangChain над ролями сообщений
        (аналог {"role": "system", ...} / {"role": "user", ...} из raw
        OpenAI API): SystemMessage(text) — системная инструкция,
        HumanMessage(text) — реплика пользователя. model.invoke([...])
        возвращает объект сообщения с полем .content (текст ответа).
    2. StateGraph(AgentState) — граф, состояние которого — словарь по
        схеме AgentState; узлы получают state и возвращают обновлённый
        state. START/END — специальные маркеры входа и выхода графа.
    3. graph.add_node(name, fn) — зарегистрировать функцию fn как узел
        графа с именем name. graph.add_edge(from, to) — обычное ребро.
        graph.add_conditional_edges(name, route_fn, mapping) — условное
        ребро: после узла name вызвать route_fn(state), и по результату
        (ключу из mapping) перейти в соответствующий узел.
        graph.compile() — собрать граф в исполняемое приложение с методом
        .invoke(initial_state).
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
    """Посчитать простое арифметическое выражение.

    Args:
        expr: строка с выражением (+ - * / и скобки).

    Returns:
        Строка с результатом вычисления.
    """
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
    """Создать ChatOpenAI, направленный в LM Studio.

    Returns:
        Настроенная модель ChatOpenAI (temperature=0, stop=["\\n"] —
        остановка генерации на первом переносе строки, чтобы модель
        выводила ровно одну строку за раз).
    """
    from langchain_openai import ChatOpenAI

    model_id = first_model_id()
    return ChatOpenAI(base_url=BASE_URL, api_key="lm-studio",
                      model=model_id, temperature=0, stop=["\n"])


def think(state: AgentState) -> AgentState:
    """Узел: спросить модель, что делать дальше; записать строку в state.

    Args:
        state: текущее состояние агента (вопрос, накопленный диалог,
            последняя строка модели, финальный ответ).

    Returns:
        Обновлённый state с новой строкой модели в state["last"].
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model = make_model()
    convo = "\n".join(state["scratch"])
    user = f"Вопрос: {state['question']}\n{convo}".strip()
    # TODO
    raise NotImplementedError


def tool(state: AgentState) -> AgentState:
    """Узел: выполнить calc[...] из последней строки, добавить Observation.

    Args:
        state: текущее состояние агента.

    Returns:
        Обновлённый state с добавленными в scratch строками Action и
        Observation.
    """
    m = re.search(r"calc\[(.*?)\]", state["last"])
    obs = calc(m.group(1)) if m else "Ошибка"
    state["scratch"].append(state["last"])
    state["scratch"].append(f"Observation: {obs}")
    return state


def route(state: AgentState) -> str:
    """Условное ребро: 'tool' если модель просит Action, иначе END.

    Args:
        state: текущее состояние агента.

    Returns:
        "tool", если в state["last"] есть "Action:", иначе END (записав
        финальный ответ в state["answer"]).
    """
    from langgraph.graph import END

    # TODO
    raise NotImplementedError


def build_graph():
    """Собрать граф: think → (route) → tool → think, либо END.

    Returns:
        Скомпилированное приложение LangGraph (объект с методом
        .invoke(initial_state)).
    """
    # StateGraph(AgentState) — граф, состояние которого — словарь по схеме
    # AgentState; узлы получают state и возвращают обновлённый state.
    # START/END — специальные маркеры входа и выхода графа.
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    # TODO
    raise NotImplementedError


def main() -> None:
    from langgraph.graph import END

    # Оффлайн-проверка инструмента и условного ребра (без LLM):
    assert calc("(12 + 8) * 3") == "60"

    state_tool: AgentState = {
        "question": "", "scratch": [], "last": "Action: calc[1+1]", "answer": ""
    }
    assert route(state_tool) == "tool"

    state_final: AgentState = {
        "question": "", "scratch": [], "last": "Final: 42", "answer": ""
    }
    assert route(state_final) == END
    assert state_final["answer"] == "42"

    try:
        app = build_graph()
        result = app.invoke(
            {"question": "Сколько будет (12 + 8) * 3?",
             "scratch": [], "last": "", "answer": ""},
            {"recursion_limit": 20},
        )
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] LLM-часть пропущена: {exc}", file=sys.stderr)
        print("[OK] ex2_langgraph_agent: инструмент calc и структура графа корректны.")
        return

    print("Трасса:", *result["scratch"], sep="\n  ")
    print(f"Ответ: {result['answer']}")
    assert "60" in result["answer"], result["answer"]  # (12+8)*3 = 60
    print("[OK] ex2_langgraph_agent: граф с условным ребром решил задачу.")


if __name__ == "__main__":
    main()
