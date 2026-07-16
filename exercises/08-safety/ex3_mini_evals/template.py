"""Упражнение 08.3 — мини-eval-набор (LM Studio).

Опирается на §5 теории 08: evals — «юнит-тесты» для недетерминированной
системы. Нельзя оценивать ответ LLM простым «прошёл/не прошёл» по точному
совпадению — нужны и более гибкие метрики. Здесь мы совмещаем два подхода:
  1. exact match (нормализованное точное совпадение);
  2. LLM-as-a-judge — второй вызов той же модели оценивает ответ по смыслу.

Требуется: LM Studio с CHAT-моделью.

Задача:
  1. exact_match: нормализовать и сравнить строки.
  2. llm_judge: спросить модель, верен ли ответ по смыслу (YES/NO).
  3. run_evals: прогнать датасет, вернуть (exact_score, judge_score).

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 08-safety/ex3_mini_evals/work.py
  4. Сверка:  08-safety/ex3_mini_evals/solution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

# Мини eval-датасет: (вопрос, эталонный ответ).
DATASET = [
    ("Столица Франции?", "Париж"),
    ("Сколько будет 2+2?", "4"),
    ("Химический символ золота?", "Au"),
    ("В каком году человек впервые высадился на Луну?", "1969"),
    ("Автор романа 'Война и мир'?", "Толстой"),
]


def normalize(text: str) -> str:
    """Привести строку к нижнему регистру, убрать пунктуацию/пробелы по краям."""
    return text.strip().lower().rstrip(".!?").strip()


def exact_match(prediction: str, reference: str) -> bool:
    """Нормализованное точное вхождение эталона в предсказание."""
    # TODO: вернуть normalize(reference) in normalize(prediction)
    raise NotImplementedError


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
    """Второй вызов модели: верен ли prediction по смыслу? (True/False)."""
    # TODO: собрать промпт-судью, попросить ответить строго YES или NO,
    #       вернуть True, если в ответе есть YES
    raise NotImplementedError


def run_evals() -> tuple[float, float]:
    """Прогнать датасет. Вернуть (доля exact_match, доля judge=YES)."""
    # TODO: для каждого (q, ref): получить ответ модели, посчитать обе метрики
    raise NotImplementedError


def main() -> None:
    # Оффлайн-проверка exact_match:
    assert exact_match("Это Париж.", "париж")
    assert not exact_match("Лондон", "Париж")

    try:
        exact_score, judge_score = run_evals()
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] LLM-часть пропущена: {exc}", file=sys.stderr)
        print("[OK] ex3: метрика exact_match корректна.")
        return

    print(f"Exact match:   {exact_score:.0%}")
    print(f"LLM-as-judge:  {judge_score:.0%}")
    # Judge обычно >= exact, т.к. ловит верные по смыслу, но иначе
    # сформулированные ответы.
    assert judge_score >= exact_score - 1e-9
    print("[OK] ex3_mini_evals: две метрики прогнаны по датасету.")


if __name__ == "__main__":
    main()
