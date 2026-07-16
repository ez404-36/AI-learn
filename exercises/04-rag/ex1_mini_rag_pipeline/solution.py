"""Эталонное решение упражнения 04.1 — мини-RAG-пайплайн.

Дополнительно демонстрирует разницу «с RAG» / «без RAG»: без контекста
модель не знает вымышленного факта (и честно скажет это или сгаллюцинирует),
а с извлечённым контекстом отвечает точно (§5 «используй только контекст»).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

EMBED_MODEL = "nomic-embed-text"

DOCS = [
    "Проект «Гларнак» стартовал в компании Кувельти в 2019 году.",
    "Кодовое имя внутреннего ускорителя вывода — «Зимородок-7».",
    "Ускоритель «Зимородок-7» сокращает латентность инференса на 42%.",
    "Столовая в офисе работает с 9:00 до 18:00.",
    "Годовой отчёт публикуется в первый вторник марта.",
]


def embed_text(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    client = get_client()
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=float)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(question: str, docs: list[str], top_k: int = 2) -> list[str]:
    q = embed_text(question)
    scored = [(cos_sim(q, embed_text(d)), d) for d in docs]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def _generate(question: str, context_chunks: list[str] | None) -> str:
    client = get_client()
    model = first_model_id(client)
    if context_chunks:
        context = "\n".join(f"- {c}" for c in context_chunks)
        prompt = (
            "Ответь на вопрос, используя ТОЛЬКО контекст ниже. "
            "Если ответа нет в контексте — так и скажи.\n\n"
            f"Контекст:\n{context}\n\nВопрос: {question}"
        )
    else:
        prompt = question
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def answer(question: str, context_chunks: list[str]) -> str:
    return _generate(question, context_chunks)


def main() -> None:
    question = "Насколько ускоритель Зимородок-7 сокращает латентность?"
    try:
        without_rag = _generate(question, None)
        chunks = retrieve(question, DOCS, top_k=2)
        with_rag = answer(question, chunks)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Ответ БЕЗ RAG (модель не знает факта): {without_rag}")
    print("\nНайденные чанки:")
    for c in chunks:
        print(f"  - {c}")
    print(f"\nОтвет С RAG: {with_rag}")

    assert "42" in with_rag, with_rag
    print("\n[OK] ex1_solution: RAG дал точный ответ из извлечённого контекста.")


if __name__ == "__main__":
    main()
