"""Эталонное решение упражнения 06.1 — симуляция квантизации."""

from __future__ import annotations

import numpy as np


def quantize(weights: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(weights))) / 127.0
    if scale == 0.0:
        scale = 1.0
    q = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return (q.astype(np.float32) * scale).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, size=(256, 256)).astype(np.float32)

    q, scale = quantize(w)
    w_hat = dequantize(q, scale)

    assert q.dtype == np.int8
    compression = w.nbytes / q.nbytes
    max_err = float(np.max(np.abs(w - w_hat)))
    mean_err = float(np.mean(np.abs(w - w_hat)))

    print(f"Сжатие: x{compression:.0f} (float32 → int8)")
    print(f"Макс. ошибка: {max_err:.4f}, средняя: {mean_err:.4f}, scale={scale:.4f}")

    assert abs(compression - 4.0) < 1e-6
    # Ошибка округления не превышает половины шага квантизации.
    assert max_err <= scale / 2 + 1e-6
    print("[OK] ex1_solution: int8 → x4 сжатие, ошибка ≤ scale/2 (как JPEG).")


if __name__ == "__main__":
    main()
