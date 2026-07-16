"""Упражнение 06.2 — игрушечная LoRA-обёртка (без LLM).

Опирается на §2 теории 06: LoRA замораживает исходные веса W и обучает
только маленькую низкоранговую добавку ΔW = B @ A (rank r ≪ dim). Число
обучаемых параметров падает в десятки-сотни раз.

Здесь мы оборачиваем `nn.Linear` LoRA-адаптером на PyTorch и сравниваем
число обучаемых параметров: full fine-tune vs LoRA.

Задача:
  1. LoRALinear.__init__: заморозить base, создать обучаемые A (r×in) и B (out×r).
  2. forward: base(x) + (x @ A^T) @ B^T * scaling.
  3. trainable_params: посчитать число обучаемых параметров.

Как работать:
  1. `python setup_practice.py` создаст рядом work.py (копию этого файла).
  2. Пишите код в work.py (он игнорируется git).
  3. Запуск:  uv run python 06-model-adaptation/ex2_lora_toy/work.py
  4. Сверка:  06-model-adaptation/ex2_lora_toy/solution.py
"""

from __future__ import annotations

import torch
from torch import nn


def trainable_params(module: nn.Module) -> int:
    """Число параметров с requires_grad=True."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    """Linear с замороженной базой и обучаемым низкоранговым адаптером."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        # TODO: заморозить base (requires_grad_(False)),
        #       создать обучаемые nn.Parameter A (rank×in) и B (out×rank);
        #       A инициализировать случайно, B — нулями (стандарт LoRA).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: base(x) + (x @ A.T) @ B.T
        raise NotImplementedError


def main() -> None:
    in_f, out_f, rank = 512, 512, 4
    full = nn.Linear(in_f, out_f)
    lora = LoRALinear(in_f, out_f, rank=rank)

    full_tp = trainable_params(full)
    lora_tp = trainable_params(lora)

    x = torch.randn(2, in_f)
    assert lora(x).shape == (2, out_f)  # forward работает

    print(f"Full fine-tune обучаемых параметров: {full_tp}")
    print(f"LoRA (rank={rank}) обучаемых параметров: {lora_tp}")
    print(f"Экономия: x{full_tp / lora_tp:.0f}")
    assert lora_tp < full_tp / 50  # на два порядка меньше
    print("[OK] ex2_lora_toy: LoRA обучает малую долю параметров.")


if __name__ == "__main__":
    main()
