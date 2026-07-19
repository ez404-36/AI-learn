"""Упражнение 01.1 — нейрон и нелинейность (без LLM).

Опирается на разделы §2–§3 теории 01: нейрон = взвешенная сумма + функция
активации, а нелинейность (ReLU) — причина, по которой глубокая сеть умнее
одного линейного слоя.

Задача:
  1. Реализовать `relu`, `neuron` (взвешенная сумма + ReLU) и `linear_unit`
     (то же самое, но без активации).
  2. Остальное (two_linear_layers, collapsed_single_layer, two_relu_layers)
     уже реализовано — это демонстрация на ваших функциях: без активации
     два слоя сворачиваются в один линейный, а с ReLU — нет.
"""

from __future__ import annotations


def linear_unit(inputs: list[float], weights: list[float], bias: float) -> float:
    """Чистая линейная операция.

    Args:
        inputs: входные значения x_1..x_n.
        weights: веса w_1..w_n.
        bias: свободное слагаемое (смещение).

    Returns:
        Взвешенная сумма: sum(x_i * w_i) + bias.
    """
    # TODO: реализуйте
    raise NotImplementedError


def relu(z: float) -> float:
    """Функция активации ReLU.

    Args:
        z: взвешенная сумма входов нейрона (до активации).

    Returns:
        z, если z > 0, иначе 0.
    """
    # TODO: реализуйте (см. §3 теории)
    raise NotImplementedError


def neuron(inputs: list[float], weights: list[float], bias: float) -> float:
    """Один нейрон: взвешенная сумма входов + bias, затем ReLU.

    Args:
        inputs: входные значения x_1..x_n.
        weights: веса w_1..w_n (по одному на каждый вход).
        bias: свободное слагаемое (смещение) — прибавляется к сумме
            независимо от входов.

    Returns:
        relu(sum(x_i * w_i) + bias).
    """
    # TODO
    raise NotImplementedError


def two_linear_layers(x: float) -> float:
    """Слой1 (без активации) → слой2 (без активации). Оба линейны.

    Это готовый демонстрационный код (веса подобраны произвольно) — он
    использует уже реализованные вами linear_unit/neuron, чтобы показать
    эффект из §3 теории. Реализовывать эту функцию не нужно.

    Args:
        x: входное значение.

    Returns:
        Результат прохода через два линейных слоя подряд.
    """
    h = linear_unit([x], [2.0], 1.0)  # h = 2x + 1
    return linear_unit([h], [3.0], -4.0)  # y = 3h - 4 = 6x - 1


def collapsed_single_layer(x: float) -> float:
    """Тот же результат ОДНИМ линейным слоем (коэффициенты — результат
    подстановки h из two_linear_layers: y = 3*(2x+1) - 4 = 6x - 1).

    Args:
        x: входное значение.

    Returns:
        Результат прохода через один эквивалентный линейный слой.
    """
    return linear_unit([x], [6.0], -1.0)


def two_relu_layers(x: float) -> float:
    """Те же слои, но с ReLU между ними — уже нелинейно.

    Args:
        x: входное значение.

    Returns:
        Результат прохода через два слоя с ReLU между ними.
    """
    h = neuron([x], [2.0], 1.0)
    return neuron([h], [3.0], -4.0)


def main() -> None:
    # 1. Проверка ReLU на таблице из §3
    assert [relu(z) for z in [5, 0.3, 0, -2, -100]] == [5, 0.3, 0, 0, 0]

    # 2. Нейрон "гасит" отрицательную сумму
    assert neuron([1.0, 2.0], [0.5, -1.0], 0.0) == 0.0  # z = -1.5 → 0
    assert neuron([1.0, 2.0], [0.5, 1.0], 0.0) == 2.5

    # 3. linear_unit НЕ гасит отрицательную сумму (в отличие от neuron)
    assert linear_unit([1.0, 2.0], [0.5, -1.0], 0.0) == -1.5
    assert linear_unit([1.0, 2.0], [0.5, 1.0], 0.0) == 2.5

    # 4. Без активации 2 слоя == 1 слой (для любого x)
    for x in [-3.0, 0.0, 1.5, 10.0]:
        assert two_linear_layers(x) == collapsed_single_layer(x)

    # 5. С ReLU эквивалентность ломается: при x, дающем отрицательный h,
    #    результат отличается от линейного случая.
    x_neg = -2.0  # h_lin = 2*(-2)+1 = -3 → ReLU обнулит
    assert two_linear_layers(x_neg) != two_relu_layers(x_neg)

    print("[OK] ex1_neuron: нейрон и ReLU работают как ожидается")


if __name__ == "__main__":
    main()
