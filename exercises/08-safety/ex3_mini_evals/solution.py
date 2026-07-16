"""Эталонное решение упражнения 08.3 — мини-eval-набор."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

DATASET = [
    ("Столица Франции?", "Париж"),
    ("Сколько будет 2+2?", "4"),
    ("Химический символ золота?", "Au"),
    ("В каком году человек впервые высадился на Луну?", "1969"),
    ("Автор романа 'Война и мир'?", "Толстой"),
]


def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".!?").strip()


def exact_match(prediction: str, reference: str) -> bool:
    return normalize(reference) in normalize(prediction)


def model_answer(question: str) -> str:
    client = get_client()
    model = first_model_id(client)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Отвечай максимально кратко."},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def llm_judge(question: str, reference: str, prediction: str) -> bool:
    client = get_client()
    model = first_model_id(client)
    judge_prompt = (
        "Ты — строгий экзаменатор. Верен ли ОТВЕТ по смыслу, "
        "учитывая ЭТАЛОН? Ответь строго одним словом: YES или NO.\n\n"
        f"Вопрос: {question}\nЭталон: {reference}\nОтвет: {prediction}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    return "YES" in verdict


def run_evals() -> tuple[float, float]:
    exact_hits = 0
    judge_hits = 0
    for question, reference in DATASET:
        prediction = model_answer(question)
        if exact_match(prediction, reference):
            exact_hits += 1
        if llm_judge(question, reference, prediction):
            judge_hits += 1
    n = len(DATASET)
    return exact_hits / n, judge_hits / n


def main() -> None:
    assert exact_match("Это Париж.", "париж")
    assert not exact_match("Лондон", "Париж")

    try:
        # Прямая проверка llm_judge на явно верном и явно неверном ответе.
        assert llm_judge("Столица Франции?", "Париж", "Париж") is True
        assert llm_judge("Столица Франции?", "Париж", "Берлин") is False

        exact_score, judge_score = run_evals()
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] LLM-часть пропущена: {exc}", file=sys.stderr)
        print("[OK] ex3_solution: метрика exact_match корректна.")
        return

    print(f"Exact match:   {exact_score:.0%}")
    print(f"LLM-as-judge:  {judge_score:.0%}")
    assert judge_score >= exact_score - 1e-9
    print(
        "[OK] ex3_solution: judge ловит верные по смыслу ответы, "
        "которые exact-match пропускает (§5)."
    )


if __name__ == "__main__":
    main()
