"""Упражнение 05.1 — цепочка на LangChain (LM Studio).

Опирается на §3 теории 05: LangChain соединяет компоненты оператором `|`
в пайплайн. `ChatOpenAI` — единая обёртка над моделью; чтобы направить её
в LM Studio, задаём `base_url`.

Собираем двухшаговую цепочку:
  1. черновик ответа по теме;
  2. сжатие черновика в одно предложение.

Требуется: LM Studio с CHAT-моделью.

Задача:
  1. make_model: создать ChatOpenAI, направленный в LM Studio.
  2. summarize_chain: prompt_draft | model | ... | prompt_squeeze | model.

Что нужно знать:
    1. ChatOpenAI(base_url=..., api_key=..., model=..., temperature=...) —
        конструктор обёртки LangChain над OpenAI-совместимым чат-API;
        base_url направляет её на LM Studio вместо облачного OpenAI,
        api_key — заглушка (LM Studio не проверяет ключ).
    2. Оператор `|` соединяет шаги пайплайна: выход слева становится
        входом справа (как в Unix-пайпах).
    3. ChatPromptTemplate.from_template("... {var} ...") — шаблон
        промпта: {var} подставляется при вызове .invoke({"var": значение}).
    4. StrOutputParser() — достаёт из ответа модели обычную строку (без
        него результат остался бы объектом сообщения с доп. полями).
    5. .invoke(входной_словарь) прогоняет его через всю цепочку и
        возвращает финальный результат.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    BASE_URL,
    LMStudioUnavailableError,
    first_model_id,
)


def make_model():
    """Создать ChatOpenAI, направленный в LM Studio.

    Returns:
        Модель ChatOpenAI, направленная на LM Studio, temperature=0.
    """
    from langchain_openai import ChatOpenAI

    model_id = first_model_id()  # заодно проверит доступность сервера
    # TODO
    raise NotImplementedError


def run(topic: str) -> str:
    """Двухшаговая цепочка: черновик → сжатие в одно предложение.

    Args:
        topic: тема, которую нужно объяснить и сжать.

    Returns:
        Итоговое сжатое предложение — результат прохода темы через
        двухшаговую цепочку (черновик → сжатие).
    """
    # StrOutputParser — достаёт из ответа модели обычную строку (без этого
    # результат остался бы объектом сообщения с доп. полями).
    from langchain_core.output_parsers import StrOutputParser
    # ChatPromptTemplate.from_template("... {var} ...") — шаблон промпта:
    # {var} подставляется при вызове .invoke({"var": значение}).
    from langchain_core.prompts import ChatPromptTemplate

    model = make_model()
    # TODO
    raise NotImplementedError


def main() -> None:
    try:
        model = make_model()
        result = run("что такое RAG")
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    # make_model должен вернуть модель, направленную в LM Studio.
    assert model.openai_api_base == BASE_URL, model.openai_api_base
    assert model.temperature == 0, model.temperature

    print(f"Итог цепочки: {result}")
    assert result.strip(), "пустой результат"
    print("[OK] ex1_langchain_chain: двухшаговая цепочка отработала.")


if __name__ == "__main__":
    main()
