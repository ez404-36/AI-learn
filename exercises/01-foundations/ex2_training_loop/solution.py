"""Эталонное решение упражнения 01.2 — цикл обучения + демонстрация overfitting.

Первая часть: линейная регрессия градиентным спуском (четыре шага из §5).
Вторая часть (`demo_overfitting`): показываем главного врага ML из §5 —
переобучение. Полином высокой степени на 6 точках идеально ложится на
обучающую выборку (train loss → 0), но на новых точках ошибка велика.
"""

from __future__ import annotations

import numpy as np


def forward(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def gradients(
    x: np.ndarray, y_true: np.ndarray, w: float, b: float
) -> tuple[float, float]:
    y_pred = forward(x, w, b)
    diff = y_pred - y_true
    dw = float(np.mean(2 * diff * x))
    db = float(np.mean(2 * diff))
    return dw, db


def train(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
) -> tuple[float, float]:
    w, b = 0.0, 0.0
    for _ in range(epochs):
        dw, db = gradients(x, y, w, b)
        w -= lr * dw
        b -= lr * db
    return w, b


def demo_overfitting() -> tuple[float, float]:
    """Вернуть (train_loss, test_loss) для переобученного полинома.

    6 обучающих точек, полином степени 5 → проходит через ВСЕ точки
    (train_loss ≈ 0), но дико осциллирует между ними (большой test_loss).
    """
    rng = np.random.default_rng(1)
    true_fn = lambda t: np.sin(t)  # noqa: E731

    x_train = np.linspace(-3, 3, 6)
    y_train = true_fn(x_train) + rng.normal(0, 0.01, size=x_train.shape)

    x_test = np.linspace(-3, 3, 60)
    y_test = true_fn(x_test)

    degree = 5  # степень = точек-1 → интерполяция без остатка
    # np.polyfit(x, y, degree) — подобрать коэффициенты многочлена заданной
    # степени методом наименьших квадратов (возвращает коэффициенты).
    coeffs = np.polyfit(x_train, y_train, degree)

    # np.polyval(coeffs, x) — вычислить значения этого многочлена в точках x.
    train_loss = mse_loss(np.polyval(coeffs, x_train), y_train)
    test_loss = mse_loss(np.polyval(coeffs, x_test), y_test)
    return train_loss, test_loss


def main() -> None:
    # mse_loss на простом примере (без обучения)
    assert mse_loss(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0

    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 50)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.05, size=x.shape)

    w, b = train(x, y)
    assert abs(w - 2.0) < 0.1, f"w={w}"
    assert abs(b - 1.0) < 0.1, f"b={b}"
    print(f"[OK] обучение: w≈{w:.3f}, b≈{b:.3f} (истина 2, 1)")

    train_loss, test_loss = demo_overfitting()
    # Переобучение: почти идеально на train, заметно хуже на test.
    assert train_loss < 1e-3
    assert test_loss > train_loss * 100
    print(
        f"[OK] overfitting: train_loss={train_loss:.2e}, "
        f"test_loss={test_loss:.2e} — модель зазубрила 6 точек, "
        "но плохо обобщает."
    )


if __name__ == "__main__":
    main()
