"""Эталонное решение упражнения 06.2 — игрушечная LoRA-обёртка.

ΔW = B @ A, где A (rank×in), B (out×rank). B инициализируется нулями,
поэтому в начале обучения адаптер не меняет выход базы (важное свойство
LoRA: старт с нулевой добавки). Обучаемы только A и B — база заморожена.
"""

from __future__ import annotations

import torch
from torch import nn


def trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.requires_grad_(False)  # веса базы заморожены
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.A.T) @ self.B.T  # низкоранговая добавка
        return self.base(x) + delta


def main() -> None:
    in_f, out_f, rank = 512, 512, 4
    full = nn.Linear(in_f, out_f)
    lora = LoRALinear(in_f, out_f, rank=rank)

    full_tp = trainable_params(full)
    lora_tp = trainable_params(lora)

    x = torch.randn(2, in_f)
    out = lora(x)
    assert out.shape == (2, out_f)

    # B=0 → в начале адаптер не меняет выход базы.
    assert torch.allclose(out, lora.base(x))

    print(f"Full fine-tune обучаемых параметров: {full_tp}")
    print(f"LoRA (rank={rank}) обучаемых параметров: {lora_tp}")
    print(f"Экономия: x{full_tp / lora_tp:.0f}")
    assert lora_tp < full_tp / 50
    print("[OK] ex2_solution: LoRA учит ~доли процента параметров, база заморожена.")


if __name__ == "__main__":
    main()
