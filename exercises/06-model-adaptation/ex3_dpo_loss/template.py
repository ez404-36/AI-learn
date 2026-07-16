"""Упражнение 06.3 — DPO loss на игрушечных лог-вероятностях (без LLM).

Опирается на §5 теории 06: DPO учит модель напрямую на парах ответов
«лучше (chosen) / хуже (rejected)», без отдельной reward-модели и RL.

Формула DPO loss для одной пары:
    loss = -log( sigmoid( beta * ( (lp_chosen  - lp_ref_chosen)
                                  -(lp_rejected - lp_ref_rejected) ) ) )
где lp_* — суммарные log-вероятности ответа под обучаемой (policy) и
опорной (ref) моделью. Минимизация loss повышает разрыв в пользу chosen.

Задача:
  1. dpo_loss: реализовать формулу на PyTorch.
  2. Проверить: если policy сильнее предпочитает chosen, чем ref, loss мал.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    lp_chosen: torch.Tensor,
    lp_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Средний DPO loss по батчу пар предпочтений."""
    # TODO:
    #   pi_logratios  = lp_chosen - lp_rejected
    #   ref_logratios = ref_chosen - ref_rejected
    #   logits = pi_logratios - ref_logratios
    #   loss = -F.logsigmoid(beta * logits)
    #   return loss.mean()
    raise NotImplementedError


def main() -> None:
    # Опорная модель одинаково оценивает chosen и rejected.
    ref_chosen = torch.tensor([-2.0, -2.0])
    ref_rejected = torch.tensor([-2.0, -2.0])

    # Случай A: policy СИЛЬНЕЕ предпочитает chosen → loss маленький.
    good = dpo_loss(
        lp_chosen=torch.tensor([-1.0, -1.0]),
        lp_rejected=torch.tensor([-3.0, -3.0]),
        ref_chosen=ref_chosen,
        ref_rejected=ref_rejected,
    )
    # Случай B: policy предпочитает rejected → loss большой.
    bad = dpo_loss(
        lp_chosen=torch.tensor([-3.0, -3.0]),
        lp_rejected=torch.tensor([-1.0, -1.0]),
        ref_chosen=ref_chosen,
        ref_rejected=ref_rejected,
    )

    print(f"loss (policy предпочитает chosen): {good.item():.4f}")
    print(f"loss (policy предпочитает rejected): {bad.item():.4f}")
    assert good.item() < bad.item()
    print("[OK] ex3_dpo_loss: loss ниже, когда модель предпочитает лучший ответ.")


if __name__ == "__main__":
    main()
