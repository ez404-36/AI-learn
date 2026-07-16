"""Эталонное решение упражнения 05.1 — цепочка на LangChain."""

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
    from langchain_openai import ChatOpenAI

    model_id = first_model_id()
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key="lm-studio",
        model=model_id,
        temperature=0,
    )


def run(topic: str) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    model = make_model()
    parser = StrOutputParser()

    draft_chain = (
        ChatPromptTemplate.from_template("Объясни {topic} в 3 предложениях.")
        | model
        | parser
    )
    squeeze_chain = (
        ChatPromptTemplate.from_template(
            "Сожми это в ОДНО короткое предложение:\n{draft}"
        )
        | model
        | parser
    )

    draft = draft_chain.invoke({"topic": topic})
    return squeeze_chain.invoke({"draft": draft})


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
    print("[OK] ex1_solution: двухшаговая цепочка (черновик → сжатие) отработала.")


if __name__ == "__main__":
    main()
