"""Эталонное решение упражнения 02.1 — эмбеддинги и поиск по смыслу."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import LMStudioUnavailableError, ensure_server

EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    client = ensure_server()
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=float)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest(query: str, phrases: list[str]) -> str:
    q = embed_text(query)
    vecs = [embed_text(p) for p in phrases]
    scores = [cos_sim(q, v) for v in vecs]
    return phrases[int(np.argmax(scores))]


def main() -> None:
    phrases = [
        "кошка спит на диване",
        "котёнок играет с клубком",
        "экскаватор роет котлован",
        "самосвал везёт щебень",
    ]
    try:
        q = embed_text("маленький кот")
        vecs = [embed_text(p) for p in phrases]
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    scores = [cos_sim(q, v) for v in vecs]
    ranked = sorted(zip(phrases, scores), key=lambda t: t[1], reverse=True)
    print("Ранжирование по близости к 'маленький кот':")
    for phrase, score in ranked:
        print(f"  {score:+.3f}  {phrase}")

    best = ranked[0][0]
    assert "кот" in best or "кошка" in best, best
    print(f"[OK] ex1_solution: ближайшая → '{best}'")


if __name__ == "__main__":
    main()
