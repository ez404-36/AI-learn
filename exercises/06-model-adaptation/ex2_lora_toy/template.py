"""Упражнение 06.2 — игрушечная LoRA-обёртка (без LLM).

Опирается на §2 теории 06: LoRA (Low-Rank Adaptation) замораживает
исходные веса W и обучает только маленькую низкоранговую добавку
ΔW = B @ A (rank r ≪ dim — ранг матриц-множителей намного меньше
размерности исходных весов). Число обучаемых параметров падает в
десятки-сотни раз.

Здесь мы оборачиваем `nn.Linear` LoRA-адаптером на PyTorch и сравниваем
число обучаемых параметров: full fine-tune (полное дообучение — меняются
ВСЕ веса модели) vs LoRA.

Задача:
  1. LoRALinear.__init__: заморозить base, создать обучаемые A (r×in) и B (out×r).
  2. forward: base(x) + (x @ A^T) @ B^T * scaling.
  3. trainable_params: посчитать число обучаемых параметров.
"""

from __future__ import annotations

import torch
from torch import nn


def trainable_params(module: nn.Module) -> int:
    """Число параметров с requires_grad=True.

    Args:
        module: модуль PyTorch (например, nn.Linear или LoRALinear).

    Returns:
        Суммарное число обучаемых элементов во всех параметрах модуля.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    """Linear с замороженной базой и обучаемым низкоранговым адаптером."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        """Создать LoRA-обёртку над nn.Linear.

        Args:
            in_features: размерность входа базового Linear-слоя.
            out_features: размерность выхода базового Linear-слоя.
            rank: rank адаптера — ранг матриц A/B, намного меньше
                in_features/out_features (чем меньше rank, тем меньше
                обучаемых параметров).
        """
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        # TODO:
        #   self.base.requires_grad_(False) — заморозить base: у всех его
        #       параметров requires_grad=False, значит autograd не будет
        #       считать по ним градиенты и optimizer их не тронет.
        #   nn.Parameter(tensor) — оборачивает обычный tensor так, чтобы
        #       PyTorch считал его ОБУЧАЕМЫМ параметром модуля (попадает
        #       в module.parameters(), requires_grad=True по умолчанию).
        #   torch.randn(rows, cols) — тензор случайных чисел (для A).
        #   torch.zeros(rows, cols) — тензор нулей (для B, стандарт LoRA).
        #   Создайте обучаемые параметры self.A формы (rank, in_features)
        #   — случайная инициализация, умноженная на 0.01, и self.B формы
        #   (out_features, rank) — нули (стандарт LoRA).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход: база + низкоранговая добавка.

        Args:
            x: входной тензор формы (batch, in_features).

        Returns:
            База + низкоранговая LoRA-добавка (см. формулу ΔW = B@A в
            докстринге модуля).
        """
        # TODO
        raise NotImplementedError


def main() -> None:
    in_f, out_f, rank = 512, 512, 4
    full = nn.Linear(in_f, out_f)
    lora = LoRALinear(in_f, out_f, rank=rank)

    full_tp = trainable_params(full)
    lora_tp = trainable_params(lora)

    x = torch.randn(2, in_f)
    out = lora(x)
    assert out.shape == (2, out_f)  # forward работает

    # torch.allclose(a, b) — True, если тензоры совпадают с точностью до
    # погрешности округления (аналог np.allclose, надёжнее, чем ==).
    # B=0 → в начале обучения адаптер не меняет выход базы (стандарт LoRA).
    assert torch.allclose(out, lora.base(x))

    print(f"Full fine-tune обучаемых параметров: {full_tp}")
    print(f"LoRA (rank={rank}) обучаемых параметров: {lora_tp}")
    print(f"Экономия: x{full_tp / lora_tp:.0f}")
    assert lora_tp < full_tp / 50  # на два порядка меньше
    print("[OK] ex2_lora_toy: LoRA обучает малую долю параметров.")


if __name__ == "__main__":
    main()
