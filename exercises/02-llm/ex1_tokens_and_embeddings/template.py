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
    """Вернуть embedding строки как np.ndarray (float)."""
    client = ensure_server()
    # TODO: client.embeddings.create(model=model, input=text)
    #       → взять .data[0].embedding, обернуть в np.array
    raise NotImplementedError


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов."""
    # TODO (формула из §3 теории 02)
    raise NotImplementedError


def nearest(query: str, phrases: list[str]) -> str:
    """Вернуть фразу из `phrases`, ближайшую к `query` по смыслу."""
    # TODO: заэмбедить query и все phrases, вернуть argmax по cos_sim
    raise NotImplementedError


def main() -> None:
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
