"""Упражнение 06.3 — DPO loss на игрушечных лог-вероятностях (без LLM).

Опирается на §5 теории 06: DPO (Direct Preference Optimization) учит
модель напрямую на парах ответов «лучше (chosen) / хуже (rejected)»,
без отдельной reward-модели (модели вознаграждения — отдельной сети,
которая оценивает качество ответа) и RL (Reinforcement Learning,
обучение с подкреплением — классический подход RLHF, см. теорию).

Формула DPO loss для одной пары:
    loss = -log( sigmoid( beta * ( (lp_chosen  - lp_ref_chosen)
                                  -(lp_rejected - lp_ref_rejected) ) ) )
где lp_* — суммарные log-вероятности ответа под обучаемой (policy) и
опорной (ref) моделью. Минимизация loss повышает разрыв в пользу chosen.

Задача:
  1. dpo_loss: реализовать формулу на PyTorch.
  2. Проверить: если policy сильнее предпочитает chosen, чем ref, loss мал.

Что нужно знать:
    1. F.logsigmoid(x) — численно устойчивый log(sigmoid(x)) (эквивалент
        torch.log(torch.sigmoid(x)), но без потери точности при больших
        |x|), применяется поэлементно к тензору.
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
    """Средний DPO loss по батчу пар предпочтений.

    Args:
        lp_chosen: суммарные log-вероятности chosen-ответа под policy.
        lp_rejected: суммарные log-вероятности rejected-ответа под policy.
        ref_chosen: суммарные log-вероятности chosen-ответа под ref-моделью.
        ref_rejected: суммарные log-вероятности rejected-ответа под
            ref-моделью.
        beta: коэффициент масштабирования разницы логарифмов (регулирует
            силу штрафа за отклонение от ref-модели).

    Returns:
        Тензор-скаляр (0-мерный) со средним DPO loss по батчу (см. формулу
        в докстринге модуля).
    """
    # TODO
    raise NotImplementedError


def main() -> None:
    # torch.tensor(list) — создать тензор PyTorch из обычного Python-списка
    # (аналог np.array(list), но для PyTorch: поддерживает autograd,
    # работает на GPU и т.д.).
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
    # Случай C: policy = ref (нейтрально) → logits=0 → loss = -log(0.5).
    neutral = dpo_loss(
        lp_chosen=torch.tensor([-2.0]),
        lp_rejected=torch.tensor([-2.0]),
        ref_chosen=torch.tensor([-2.0]),
        ref_rejected=torch.tensor([-2.0]),
    )

    print(f"loss (policy предпочитает chosen): {good.item():.4f}")
    print(f"loss (policy предпочитает rejected): {bad.item():.4f}")
    print(f"loss (нейтрально): {neutral.item():.4f} ≈ -log(0.5)")

    assert good.item() < neutral.item() < bad.item()
    assert abs(neutral.item() - torch.log(torch.tensor(2.0)).item()) < 1e-5
    print("[OK] ex3_dpo_loss: loss ниже, когда модель предпочитает лучший ответ.")


if __name__ == "__main__":
    main()
