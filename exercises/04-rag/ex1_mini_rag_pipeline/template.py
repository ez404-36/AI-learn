"""Упражнение 04.1 — мини-RAG-пайплайн (LM Studio).

Опирается на §3–§6 теории 04: полный путь «чанкинг → эмбеддинг → поиск
top-K (cos_sim) → сборка промпта → генерация». Демонстрируем главную
пользу RAG: ответ на вопрос про ВЫМЫШЛЕННЫЙ факт возможен только когда
факт лежит в контексте (без RAG модель его знать не может).

Требуется: LM Studio с CHAT-моделью и EMBEDDING-моделью.

Задача:
  1. embed_text: получить вектор через LM Studio.
  2. cos_sim: косинусная близость (раздел 02/04).
  3. retrieve: top-K ближайших чанков к запросу.
  4. answer: собрать промпт «отвечай только по контексту» и сгенерировать.
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

# Мини-база знаний с ВЫМЫШЛЕННЫМ фактом, которого нет в обучении модели.
DOCS = [
    "Проект «Гларнак» стартовал в компании Кувельти в 2019 году.",
    "Кодовое имя внутреннего ускорителя вывода — «Зимородок-7».",
    "Ускоритель «Зимородок-7» сокращает латентность инференса на 42%.",
    "Столовая в офисе работает с 9:00 до 18:00.",
    "Годовой отчёт публикуется в первый вторник марта.",
]


def embed_text(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    """Получить embedding строки через LM Studio.

    Args:
        text: строка, для которой нужен вектор.
        model: имя embedding-модели в LM Studio.

    Returns:
        Вектор embedding'а как np.ndarray.
    """
    # TODO
    raise NotImplementedError


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов.

    Args:
        a: первый вектор.
        b: второй вектор.

    Returns:
        Косинусная близость a и b (раздел 02/04).
    """
    # TODO
    raise NotImplementedError


def retrieve(question: str, docs: list[str], top_k: int = 2) -> list[str]:
    """Найти top_k документов, ближайших по смыслу к вопросу.

    Args:
        question: вопрос, для которого ищем релевантный контекст.
        docs: набор документов-кандидатов.
        top_k: сколько ближайших документов вернуть.

    Returns:
        Список из top_k документов, отсортированных по убыванию cos_sim
        к вопросу. Заэмбедите вопрос и docs, отсортируйте по cos_sim.
    """
    # TODO
    raise NotImplementedError


def answer(question: str, context_chunks: list[str]) -> str:
    """Сгенерировать ответ, опираясь ТОЛЬКО на переданный контекст.

    Args:
        question: вопрос пользователя.
        context_chunks: релевантные чанки контекста (из retrieve).

    Returns:
        Текстовый ответ модели. Соберите промпт с инструкцией «используй
        только контекст ниже» (см. §5 теории) и вызовите
        chat.completions.create.
    """
    client = get_client()
    model = first_model_id(client)
    # TODO
    raise NotImplementedError


def _without_rag(question: str) -> str:
    """Готовый вспомогательный код: спросить модель БЕЗ контекста, чтобы
    показать, что вымышленный факт ей неизвестен (см. докстринг модуля).
    Реализовывать не нужно — используется только для демонстрации.

    Args:
        question: вопрос пользователя.

    Returns:
        Текстовый ответ модели без доступа к контексту.
    """
    client = get_client()
    model = first_model_id(client)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    # Оффлайн-проверка cos_sim (без LM Studio):
    assert cos_sim(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cos_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0

    question = "Насколько ускоритель Зимородок-7 сокращает латентность?"
    try:
        without_rag = _without_rag(question)
        chunks = retrieve(question, DOCS, top_k=2)
        with_rag = answer(question, chunks)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Ответ БЕЗ RAG (модель не знает факта): {without_rag}")
    print("\nНайденные чанки:")
    for c in chunks:
        print(f"  - {c}")

    # retrieve должен найти релевантный чанк про сам ускоритель.
    assert any("Зимородок-7" in c and "42%" in c for c in chunks), chunks

    print(f"\nОтвет с RAG: {with_rag}")
    assert "42" in with_rag, with_rag
    print("[OK] ex1_mini_rag: ответ основан на извлечённом контексте.")


if __name__ == "__main__":
    main()
