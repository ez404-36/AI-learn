"""Эталонное решение упражнения 05.3 — шаг обучения на PyTorch."""

from __future__ import annotations

import torch
from torch import nn


def build() -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, loss_fn, optimizer


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    pred = model(x)               # forward pass
    loss = loss_fn(pred, y)       # ошибка
    loss.backward()               # autograd → градиенты (backprop)
    optimizer.step()              # шаг градиентного спуска
    optimizer.zero_grad()         # обнулить градиенты перед след. шагом
    return loss.item()


def train(epochs: int = 300) -> tuple[float, float]:
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 50).unsqueeze(1)
    y = 2.0 * x + 1.0

    model, loss_fn, optimizer = build()
    for _ in range(epochs):
        train_step(model, loss_fn, optimizer, x, y)

    return model.weight.item(), model.bias.item()


def main() -> None:
    w, b = train()
    print(f"Выучено: w≈{w:.3f}, b≈{b:.3f} (истина 2, 1)")
    assert abs(w - 2.0) < 0.1, w
    assert abs(b - 1.0) < 0.1, b
    print("[OK] ex3_solution: autograd + optimizer сошлись к y=2x+1.")


if __name__ == "__main__":
    main()
