"""Упражнение 05.3 — реальный шаг обучения на PyTorch (без LLM).

Опирается на §6 теории 05 и §5 теории 01. Тот же цикл обучения, что вы
делали руками в 01/ex2, но теперь на настоящем PyTorch: `loss.backward()`
считает градиенты через autograd, `optimizer.step()` обновляет веса.

Учим `nn.Linear(1, 1)` восстанавливать зависимость y = 2x + 1.

Задача:
  1. build: создать модель, loss-функцию и оптимизатор.
  2. train_step: forward → loss → backward → step → zero_grad (§6 теории).
  3. train: прогнать несколько эпох, вернуть выученные w, b.

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 05-pipelines-frameworks/ex3_pytorch_training/work.py
  4. Сверка:  05-pipelines-frameworks/ex3_pytorch_training/solution.py
"""

from __future__ import annotations

import torch
from torch import nn


def build() -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Вернуть (модель Linear(1,1), MSELoss, SGD-оптимизатор)."""
    model = nn.Linear(1, 1)
    # TODO: loss_fn = nn.MSELoss(); optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    raise NotImplementedError


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Один шаг обучения. Вернуть значение loss (float)."""
    # TODO (пять строк из §6 теории):
    #   pred = model(x)
    #   loss = loss_fn(pred, y)
    #   loss.backward()
    #   optimizer.step()
    #   optimizer.zero_grad()
    #   return loss.item()
    raise NotImplementedError


def train(epochs: int = 300) -> tuple[float, float]:
    """Обучить модель на y=2x+1, вернуть выученные (w, b)."""
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 50).unsqueeze(1)
    y = 2.0 * x + 1.0

    model, loss_fn, optimizer = build()
    for _ in range(epochs):
        train_step(model, loss_fn, optimizer, x, y)

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
