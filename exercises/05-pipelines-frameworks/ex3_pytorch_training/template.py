"""Упражнение 05.3 — реальный шаг обучения на PyTorch (без LLM).

Опирается на §6 теории 05 и §5 теории 01. Тот же цикл обучения, что вы
делали руками в 01/ex2, но теперь на настоящем PyTorch: `loss.backward()`
считает градиенты через autograd, `optimizer.step()` обновляет веса.

Учим `nn.Linear(1, 1)` восстанавливать зависимость y = 2x + 1.

Задача:
  1. build: создать модель, loss-функцию и оптимизатор.
  2. train_step: forward → loss → backward → step → zero_grad (§6 теории).
  3. train: прогнать несколько эпох, вернуть выученные w, b.

Что нужно знать:
    1. nn.MSELoss() — готовый объект функции потерь (среднее квадратов
        ошибки), вызывается как loss_fn(pred, target).
    2. torch.optim.SGD(model.parameters(), lr=0.1) — оптимизатор
        градиентного спуска; model.parameters() — все обучаемые тензоры
        модели (веса, bias), lr — скорость обучения.
    3. model(x) вызывает forward, возвращает предсказание (тензор с
        историей операций для последующего backward).
    4. loss.backward() — autograd: считает градиенты loss по ВСЕМ
        параметрам модели (заполняет .grad у каждого параметра из
        model.parameters()).
    5. optimizer.step() — обновляет веса модели на основе .grad (для SGD:
        w -= lr * w.grad). optimizer.zero_grad() — обнуляет .grad перед
        следующим шагом (иначе градиенты накапливаются).
    6. .item() достаёт из тензора-скаляра обычное Python-число (float).
"""

from __future__ import annotations

import torch
from torch import nn


def build() -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Создать модель, loss-функцию и оптимизатор.

    Returns:
        Кортеж (модель Linear(1,1), MSELoss, SGD-оптимизатор).
    """
    # nn.Linear(in_features, out_features) — готовый линейный слой PyTorch
    # y = W @ x + b; веса W и b создаются автоматически со requires_grad=True
    # (то есть PyTorch сам будет считать по ним градиенты).
    model = nn.Linear(1, 1)
    # TODO
    raise NotImplementedError


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Один шаг обучения: forward → loss → backward → step → zero_grad.

    Args:
        model: обучаемая модель.
        loss_fn: функция потерь.
        optimizer: оптимизатор.
        x: входные данные батча.
        y: истинные значения батча.

    Returns:
        Значение loss (float) за этот шаг обучения (forward → loss →
        backward → step → zero_grad, см. §6 теории).
    """
    # TODO
    raise NotImplementedError


def train(epochs: int = 300) -> tuple[float, float]:
    """Обучить модель на y=2x+1.

    Args:
        epochs: epoch — один полный проход обучения по всем примерам
            данных; здесь каждая эпоха = один шаг градиентного спуска.

    Returns:
        Кортеж (w, b) — выученные параметры модели.
    """
    # torch.manual_seed(seed) — фиксирует случайность (инициализацию весов
    # и т.п.) для воспроизводимого результата, аналог np.random.default_rng.
    torch.manual_seed(0)
    # torch.linspace — как np.linspace, но возвращает тензор PyTorch.
    # .unsqueeze(1) добавляет ось: было (50,), стало (50, 1) — nn.Linear
    # ожидает вход вида (batch, features): batch — размер пакета примеров,
    # обрабатываемых за один forward-проход (здесь все 50 точек — один
    # батч), features — число признаков одного примера (здесь 1: сам x).
    x = torch.linspace(-1, 1, 50).unsqueeze(1)
    y = 2.0 * x + 1.0

    model, loss_fn, optimizer = build()
    for _ in range(epochs):
        train_step(model, loss_fn, optimizer, x, y)

    # model.weight / model.bias — обученные параметры слоя nn.Linear
    # (тензоры формы (1,1) и (1,)); .item() достаёт из них Python-число.
    w = model.weight.item()
    b = model.bias.item()
    return w, b


def main() -> None:
    w, b = train()
    print(f"Выучено: w≈{w:.3f}, b≈{b:.3f} (истина 2, 1)")
    assert abs(w - 2.0) < 0.1, w
    assert abs(b - 1.0) < 0.1, b
    print("[OK] ex3_pytorch_training: autograd + optimizer сошлись к решению.")


if __name__ == "__main__":
    main()
