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
    """Вернуть ChatOpenAI, направленный в LM Studio."""
    from langchain_openai import ChatOpenAI

    model_id = first_model_id()  # заодно проверит доступность сервера
    # TODO: вернуть ChatOpenAI(base_url=BASE_URL, api_key="lm-studio",
    #                          model=model_id, temperature=0)
    raise NotImplementedError


def run(topic: str) -> str:
    """Двухшаговая цепочка: черновик → сжатие в одно предложение."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    model = make_model()
    # TODO: соберите цепочку через оператор | :
    #   draft = ChatPromptTemplate.from_template("Объясни {topic} в 3 предложениях")
    #           | model | StrOutputParser()
    #   squeeze = ChatPromptTemplate.from_template(
    #             "Сожми это в ОДНО предложение:\n{draft}") | model | StrOutputParser()
    #   Затем прогоните draft, потом squeeze.
    raise NotImplementedError


def main() -> None:
    try:
        result = run("что такое RAG")
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Итог цепочки: {result}")
    assert result.strip(), "пустой результат"
    print("[OK] ex1_langchain_chain: двухшаговая цепочка отработала.")


if __name__ == "__main__":
    main()
