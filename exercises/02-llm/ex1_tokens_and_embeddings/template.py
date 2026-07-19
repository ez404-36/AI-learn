"""Упражнение 02.1 — эмбеддинги и поиск по смыслу (LM Studio).

Опирается на §2–§3 теории 02: текст → вектор чисел (embedding), близкий
смысл → близкие векторы. Здесь мы получаем реальные эмбеддинги от локальной
embedding-модели в LM Studio и строим наивный «ближайший сосед».

Требуется: LM Studio с загруженной EMBEDDING-моделью (например
nomic-embed-text). См. exercises/README.md, раздел 2.

Задача:
  1. embed_text: получить вектор для строки через /v1/embeddings.
  2. cos_sim: косинусная близость (как в §3 теории).
  3. nearest: для запроса найти ближайшую по смыслу фразу из набора.

Что нужно знать:
    1. client.embeddings.create(model=model, input=text) — запрос к
        эндпоинту /v1/embeddings, возвращает объект resp с полем .data
        (список результатов по каждой входной строке); resp.data[0].embedding
        — сам вектор (обычный list[float]); np.array(...) оборачивает его
        в np.ndarray для арифметики (cos_sim).
    2. np.dot(a, b) — скалярное произведение векторов (сумма a[i]*b[i]).
    3. np.linalg.norm(v) — длина (норма) вектора: sqrt(сумма v[i]**2).
    4. np.argmax(list_or_array) — индекс наибольшего элемента (не само
        значение, а его позиция).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import LMStudioUnavailableError, ensure_server

# Имя embedding-модели в LM Studio. Можно оставить как есть — LM Studio
# использует загруженную embedding-модель. При необходимости замените.
EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    """Получить embedding строки через LM Studio.

    Args:
        text: строка, для которой нужен вектор.
        model: имя embedding-модели в LM Studio.

    Returns:
        Вектор embedding'а как np.ndarray (float).
    """
    # ensure_server() — наш хелпер из common/lmstudio_client.py: возвращает
    # настроенный клиент openai.OpenAI и сразу проверяет, что LM Studio
    # доступен (иначе бросает понятную LMStudioUnavailableError).
    client = ensure_server()
    # TODO
    raise NotImplementedError


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов (формула из §3 теории 02).

    Args:
        a: первый вектор.
        b: второй вектор.

    Returns:
        cos_sim = dot(a, b) / (norm(a) * norm(b)).
    """
    # TODO
    raise NotImplementedError


def nearest(query: str, phrases: list[str]) -> str:
    """Найти фразу из `phrases`, ближайшую к `query` по смыслу.

    Args:
        query: запрос, для которого ищем ближайшую по смыслу фразу.
        phrases: набор фраз-кандидатов.

    Returns:
        Элемент phrases с наибольшим cos_sim к query.
    """
    # TODO
    raise NotImplementedError


def main() -> None:
    # Оффлайн-проверка cos_sim (без LM Studio):
    assert cos_sim(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cos_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert cos_sim(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0

    phrases = [
        "кошка спит на диване",
        "котёнок играет с клубком",
        "экскаватор роет котлован",
        "самосвал везёт щебень",
    ]
    try:
        result = nearest("маленький кот", phrases)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    # Ожидаем, что ближайшим окажется что-то про кошек, а не про технику.
    assert "кот" in result or "кошка" in result, result
    print(f"[OK] ex1: ближайшая к 'маленький кот' фраза → '{result}'")


if __name__ == "__main__":
    main()
