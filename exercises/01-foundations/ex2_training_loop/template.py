"""Упражнение 01.2 — цикл обучения линейной регрессии вручную (без LLM).

Опирается на §5 теории 01 (четыре шага: forward → loss → backprop → шаг
оптимизатора). Здесь backprop считается РУКАМИ через формулу градиента
MSE — без autograd (реальный PyTorch-вариант будет в 05/ex3).

Модель: y = w*x + b. Датасет синтетический: y = 2x + 1 + шум.

Задача:
  1. forward: предсказать y_pred по x.
  2. mse_loss: средний квадрат ошибки.
  3. gradients: аналитические производные MSE по w и b.
  4. train: собрать цикл из шагов и вернуть подобранные w, b.

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 01-foundations/ex2_training_loop/work.py
  4. Сверка:  01-foundations/ex2_training_loop/solution.py
"""

from __future__ import annotations

import numpy as np


def forward(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Прямой проход: y = w*x + b."""
    # TODO
    raise NotImplementedError


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Средний квадрат ошибки."""
    # TODO
    raise NotImplementedError


def gradients(
    x: np.ndarray, y_true: np.ndarray, w: float, b: float
) -> tuple[float, float]:
    """Вернуть (dL/dw, dL/db) для MSE и модели w*x+b.

    Подсказка: L = mean((w*x+b - y)^2).
      dL/dw = mean(2*(y_pred - y)*x)
      dL/db = mean(2*(y_pred - y))
    """
    # TODO
    raise NotImplementedError


def train(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
) -> tuple[float, float]:
    """Цикл обучения: вернуть подобранные (w, b)."""
    w, b = 0.0, 0.0
    # TODO: на каждой эпохе посчитать градиенты и сделать шаг:
    #   w -= lr * dw;  b -= lr * db
    raise NotImplementedError


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 50)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.05, size=x.shape)

    w, b = train(x, y)
    assert abs(w - 2.0) < 0.1, f"w={w}"
    assert abs(b - 1.0) < 0.1, f"b={b}"
    print(f"[OK] ex2_training_loop: выучено w≈{w:.3f}, b≈{b:.3f} (истина 2, 1)")


if __name__ == "__main__":
    main()
