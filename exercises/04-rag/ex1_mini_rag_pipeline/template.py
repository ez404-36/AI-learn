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

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 04-rag/ex1_mini_rag_pipeline/work.py
  4. Сверка:  04-rag/ex1_mini_rag_pipeline/solution.py
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
    """Вернуть embedding строки как np.ndarray."""
    # TODO
    raise NotImplementedError


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость."""
    # TODO
    raise NotImplementedError


def retrieve(question: str, docs: list[str], top_k: int = 2) -> list[str]:
    """Вернуть top_k документов, ближайших по смыслу к вопросу."""
    # TODO: заэмбедить вопрос и docs, отсортировать по cos_sim, взять top_k
    raise NotImplementedError


def answer(question: str, context_chunks: list[str]) -> str:
    """Сгенерировать ответ, опираясь ТОЛЬКО на переданный контекст."""
    client = get_client()
    model = first_model_id(client)
    # TODO: собрать промпт с инструкцией «используй только контекст ниже»
    #       (см. §5 теории) и вызвать chat.completions.create
    raise NotImplementedError


def main() -> None:
    question = "Насколько ускоритель Зимородок-7 сокращает латентность?"
    try:
        chunks = retrieve(question, DOCS, top_k=2)
        with_rag = answer(question, chunks)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print("Найденные чанки:")
    for c in chunks:
        print(f"  - {c}")
    print(f"Ответ с RAG: {with_rag}")
    assert "42" in with_rag, with_rag
    print("[OK] ex1_mini_rag: ответ основан на извлечённом контексте.")


if __name__ == "__main__":
    main()
