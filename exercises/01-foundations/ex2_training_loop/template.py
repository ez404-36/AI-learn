"""Упражнение 01.2 — цикл обучения линейной регрессии вручную (без LLM).

Опирается на §5 теории 01 (четыре шага: forward → loss → backprop → шаг
оптимизатора). Здесь backprop считается РУКАМИ через формулу градиента
MSE (Mean Squared Error, «средний квадрат ошибки» — популярный loss для
регрессии: усредняем квадраты разниц между предсказанием и истиной) —
без autograd (autograd — механизм автоматического дифференцирования,
который сам считает производные; здесь мы делаем его работу вручную,
чтобы понять, что происходит внутри).

Модель: y = w*x + b. Датасет синтетический: y = 2x + 1 + шум.

Задача:
  1. forward: предсказать y_pred по x.
  2. mse_loss: средний квадрат ошибки.
  3. gradients: аналитические производные MSE по w и b.
  4. train: собрать цикл из шагов и вернуть подобранные w, b.

Что нужно знать:
    1. np.ndarray - массив чисел NumPy.
        Операторы *, +, -, / применяются к нему поэлементно ко всем значениям сразу.
        Вторым аргументом может быть либо число, либо другой np.ndarray.
    2. np.mean(np.ndarray) - среднее арифметическое элементов массива
    3. dL/dw, dL/db — частная производная (partial derivative) loss L по
        параметру w (соответственно b): показывает, насколько изменится L
        при малом изменении w, если b оставить зафиксированным (и наоборот
        для dL/db). Знак производной задаёт направление шага: если dL/dw
        положительна, увеличение w увеличивает loss, значит нужно двигаться
        в обратную сторону — отсюда `w -= lr * dL/dw`.

"""

from __future__ import annotations

import numpy as np


def forward(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Прямой проход: y = w*x + b.

    Args:
        x: массив со входными значениями
        w: вес (наклон прямой).
        b: bias (свободное слагаемое, смещение).

    Returns:
        Массив предсказаний.
    """
    # TODO: реализуйте
    raise NotImplementedError


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Средний квадрат ошибки (MSE).

    Args:
        y_pred: предсказанные значения.
        y_true: истинные значения.

    Returns:
        Среднее арифметическое квадратов разниц
    """
    # TODO: реализуйте
    raise NotImplementedError


def gradients(
    x: np.ndarray, y_true: np.ndarray, w: float, b: float
) -> tuple[float, float]:
    """Вернуть (dL/dw, dL/db) для MSE и модели w*x+b.

    Подсказка: L = mean((w*x+b - y)^2).
      dL/dw = mean(2*(y_pred - y)*x)
      dL/db = mean(2*(y_pred - y))

    Args:
        x: входные значения.
        y_true: истинные значения.
        w: текущий вес.
        b: текущий bias.

    Returns:
        Кортеж (dL/dw, dL/db) — частные производные MSE по w и b.
    """
    # TODO: реализуйте
    raise NotImplementedError


def train(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
) -> tuple[float, float]:
    """Цикл обучения: вернуть подобранные (w, b).

    Args:
        x: входные значения датасета.
        y: истинные значения датасета.
        lr: learning rate — скорость обучения: насколько сильно сдвигать
            w и b на каждом шаге в сторону антиградиента.
        epochs: epoch — один полный проход обучения по всем примерам
            данных; здесь каждая эпоха = один шаг градиентного спуска.

    Returns:
        Кортеж (w, b) — подобранные параметры модели после обучения.
    """
    w, b = 0.0, 0.0
    # TODO: на каждой эпохе посчитать градиенты и сделать шаг:
    #   w -= lr * dw;  b -= lr * db
    raise NotImplementedError


def main() -> None:
    # np.array(list) — создать массив NumPy из обычного Python-списка.
    # mse_loss на простом примере (без обучения)
    assert mse_loss(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0

    # np.random.default_rng(seed) — генератор случайных чисел с фиксированным
    # seed (одинаковый результат при каждом запуске — для воспроизводимости).
    rng = np.random.default_rng(0)
    # np.linspace(start, stop, num) — num точек, равномерно распределённых
    # от start до stop включительно (здесь: 50 точек от -1 до 1).
    x = np.linspace(-1, 1, 50)
    # x.shape — форма массива (кортеж размеров по каждой оси, здесь (50,)).
    y = 2.0 * x + 1.0 + rng.normal(0, 0.05, size=x.shape)

    w, b = train(x, y)
    assert abs(w - 2.0) < 0.1, f"w={w}"
    assert abs(b - 1.0) < 0.1, f"b={b}"
    print(f"[OK] ex2_training_loop: выучено w≈{w:.3f}, b≈{b:.3f} (истина 2, 1)")


if __name__ == "__main__":
    main()
