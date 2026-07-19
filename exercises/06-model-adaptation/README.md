# Практика 06 — Адаптация и сжатие моделей

Опирается на теорию: [06-model-adaptation](../../06-model-adaptation/README.md).
LM Studio **не нужен** — все задания на toy-тензорах (NumPy/PyTorch),
без скачивания реальных моделей и без GPU.

## Упражнения

### ex1_quantization_sim — симуляция квантизации

Квантизуете float32-матрицу в int8 вручную (симметричная схема §3),
измеряете сжатие (×4) и ошибку восстановления. Иллюстрирует «JPEG для
весов»: резко легче, качество почти не страдает.

```bash
uv run python 06-model-adaptation/ex1_quantization_sim/work.py
```

### ex2_lora_toy — игрушечная LoRA

Оборачиваете `nn.Linear` LoRA-адаптером (§2): база заморожена, обучаются
только низкоранговые `A` и `B` (ΔW = B @ A). Сравниваете число обучаемых
параметров full fine-tune vs LoRA — экономия на два порядка.

```bash
uv run python 06-model-adaptation/ex2_lora_toy/work.py
```

### ex3_dpo_loss — формула DPO

Реализуете DPO loss (§5) на игрушечных log-вероятностях пар
«chosen/rejected». Проверяете: loss ниже, когда модель предпочитает
лучший ответ сильнее опорной модели.

```bash
uv run python 06-model-adaptation/ex3_dpo_loss/work.py
```

> Как запускать упражнения (`setup_practice.py`, `work.py`, `solution.py`) —
> см. [общий README практики](../README.md#3-конвенция-файлов).
