"""Упражнение 06.1 — симуляция квантизации весов (без LLM).

Опирается на §3 теории 06: квантизация снижает точность чисел, которыми
записаны веса (обычно 16 бит → 8 или 4). Здесь мы вручную квантизуем
float32-матрицу в int8 (симметричная per-tensor схема), измеряем сжатие
и ошибку восстановления.

Задача:
  1. quantize: float32 → int8 + масштаб (scale).
  2. dequantize: int8 + scale → float32 (приближённо).
  3. В main: посчитать коэффициент сжатия и ошибку.
"""

from __future__ import annotations

import numpy as np


def quantize(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """float32 → (int8-массив, scale).

    Симметричная схема: scale = max(|w|) / 127; q = round(w / scale).
    """
    # TODO: посчитать scale, квантизовать в int8 (np.clip до [-127, 127])
    raise NotImplementedError


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """int8 + scale → приближённый float32."""
    # TODO: вернуть q * scale
    raise NotImplementedError


def main() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, size=(256, 256)).astype(np.float32)

    q, scale = quantize(w)
    w_hat = dequantize(q, scale)

    assert q.dtype == np.int8
    compression = w.nbytes / q.nbytes  # float32 (4 байта) → int8 (1 байт)
    max_err = float(np.max(np.abs(w - w_hat)))

    print(f"Сжатие: x{compression:.0f} (float32 → int8)")
    print(f"Макс. ошибка восстановления: {max_err:.4f}")
    assert abs(compression - 4.0) < 1e-6
    assert max_err <= scale / 2 + 1e-6  # ошибка округления ≤ полшага
    print("[OK] ex1_quantization_sim: int8 даёт x4 сжатие при малой ошибке.")


if __name__ == "__main__":
    main()
