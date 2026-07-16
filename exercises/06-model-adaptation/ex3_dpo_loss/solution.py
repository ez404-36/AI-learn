"""Эталонное решение упражнения 06.3 — DPO loss."""

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
    pi_logratios = lp_chosen - lp_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def main() -> None:
    ref_chosen = torch.tensor([-2.0, -2.0])
    ref_rejected = torch.tensor([-2.0, -2.0])

    good = dpo_loss(
        lp_chosen=torch.tensor([-1.0, -1.0]),
        lp_rejected=torch.tensor([-3.0, -3.0]),
        ref_chosen=ref_chosen,
        ref_rejected=ref_rejected,
    )
    bad = dpo_loss(
        lp_chosen=torch.tensor([-3.0, -3.0]),
        lp_rejected=torch.tensor([-1.0, -1.0]),
        ref_chosen=ref_chosen,
        ref_rejected=ref_rejected,
    )
    # Нейтральный случай: policy = ref → logits=0 → loss = -log(0.5).
    neutral = dpo_loss(
        lp_chosen=torch.tensor([-2.0]),
        lp_rejected=torch.tensor([-2.0]),
        ref_chosen=torch.tensor([-2.0]),
        ref_rejected=torch.tensor([-2.0]),
    )

    print(f"loss (предпочитает chosen):   {good.item():.4f}")
    print(f"loss (предпочитает rejected): {bad.item():.4f}")
    print(f"loss (нейтрально):            {neutral.item():.4f} ≈ -log(0.5)")

    assert good.item() < neutral.item() < bad.item()
    assert abs(neutral.item() - torch.log(torch.tensor(2.0)).item()) < 1e-5
    print("[OK] ex3_solution: DPO loss корректно ранжирует предпочтения.")


if __name__ == "__main__":
    main()
