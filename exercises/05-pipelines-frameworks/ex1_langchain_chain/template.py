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
        ChatOpenAI(base_url=BASE_URL, api_key="lm-studio", model=model_id,
        temperature=0). ChatOpenAI — обёртка LangChain над
        OpenAI-совместимым чат-API; ведёт себя как любая LangChain-модель
        (можно соединять оператором | ), но под капотом ходит туда же,
        куда наш openai.OpenAI — в LM Studio.
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
        Итоговое сжатое предложение. Соберите цепочку через оператор |
        (соединяет шаги пайплайна: выход слева становится входом справа,
        как в Unix-пайпах):
          draft = ChatPromptTemplate.from_template("Объясни {topic} в 3
                  предложениях") | model | StrOutputParser()
          squeeze = ChatPromptTemplate.from_template(
                    "Сожми это в ОДНО предложение:\n{draft}") | model |
                    StrOutputParser()
        Затем вызовите draft.invoke({"topic": topic}), потом
        squeeze.invoke({"draft": <результат draft>}) — .invoke() прогоняет
        входной словарь через всю цепочку и возвращает финальный результат.
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
