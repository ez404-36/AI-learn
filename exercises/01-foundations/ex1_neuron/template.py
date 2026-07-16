"""Упражнение 01.1 — нейрон и нелинейность (без LLM).

Опирается на разделы §2–§3 теории 01: нейрон = взвешенная сумма + функция
активации, а нелинейность (ReLU) — причина, по которой глубокая сеть умнее
одного линейного слоя.

Задача:
  1. Реализовать `relu` и `neuron` (взвешенная сумма + ReLU).
  2. Собрать двуслойную сеть ДВАЖДЫ: с активацией и без неё, и показать,
     что без активации два слоя сворачиваются в один линейный.

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 01-foundations/ex1_neuron/work.py
  4. Сверка:  01-foundations/ex1_neuron/solution.py
"""

from __future__ import annotations


def relu(z: float) -> float:
    """Вернуть z, если z > 0, иначе 0."""
    # TODO: реализуйте ReLU (см. §3 теории)
    raise NotImplementedError


def neuron(inputs: list[float], weights: list[float], bias: float) -> float:
    """Один нейрон: взвешенная сумма входов + bias, затем ReLU."""
    # TODO: посчитайте z = sum(x*w) + bias и верните relu(z)
    raise NotImplementedError


def linear_unit(inputs: list[float], weights: list[float], bias: float) -> float:
    """Тот же нейрон, но БЕЗ активации — чистая линейная операция."""
    # TODO: верните взвешенную сумму без relu
    raise NotImplementedError


def main() -> None:
    # 1. Проверка ReLU на таблице из §3
    assert [relu(z) for z in [5, 0.3, 0, -2, -100]] == [5, 0.3, 0, 0, 0]

    # 2. Нейрон "гасит" отрицательную сумму
    assert neuron([1.0, 2.0], [0.5, -1.0], 0.0) == 0.0  # z = -1.5 → 0
    assert neuron([1.0, 2.0], [0.5, 1.0], 0.0) == 2.5

    print("[OK] ex1_neuron: нейрон и ReLU работают как ожидается")


if __name__ == "__main__":
    main()
