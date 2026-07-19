"""Эталонное решение упражнения 01.1 — нейрон и нелинейность.

Ключевая демонстрация: без функции активации два линейных слоя
эквивалентны одному линейному слою (см. §3 «Почему без неё сеть не
работает»). С ReLU это перестаёт быть верным — сеть получает способность
описывать не-прямые зависимости.
"""

from __future__ import annotations


def linear_unit(inputs: list[float], weights: list[float], bias: float) -> float:
    return sum(x * w for x, w in zip(inputs, weights)) + bias


def relu(z: float) -> float:
    return z if z > 0 else 0


def neuron(inputs: list[float], weights: list[float], bias: float) -> float:
    z = sum(x * w for x, w in zip(inputs, weights)) + bias
    return relu(z)


def two_linear_layers(x: float) -> float:
    """Слой1 (без активации) → слой2 (без активации). Оба линейны."""
    h = linear_unit([x], [2.0], 1.0)   # h = 2x + 1
    return linear_unit([h], [3.0], -4.0)  # y = 3h - 4 = 6x - 1


def collapsed_single_layer(x: float) -> float:
    """Тот же результат ОДНИМ линейным слоем: y = 6x - 1."""
    return linear_unit([x], [6.0], -1.0)


def two_relu_layers(x: float) -> float:
    """Те же слои, но с ReLU между ними — уже нелинейно."""
    h = neuron([x], [2.0], 1.0)
    return neuron([h], [3.0], -4.0)


def main() -> None:
    # 1. ReLU
    assert [relu(z) for z in [5, 0.3, 0, -2, -100]] == [5, 0.3, 0, 0, 0]

    # 2. Нейрон гасит отрицательную сумму
    assert neuron([1.0, 2.0], [0.5, -1.0], 0.0) == 0.0
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

    print("[OK] ex1_neuron solution: без активации слои сворачиваются в один,")
    print("     с ReLU — нет. Именно поэтому нужна нелинейность.")


if __name__ == "__main__":
    main()
