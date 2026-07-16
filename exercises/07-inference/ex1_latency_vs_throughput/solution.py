"""Эталонное решение упражнения 07.1 — латентность vs throughput."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.lmstudio_client import (
    LMStudioUnavailableError,
    first_model_id,
    get_client,
)

PROMPT = "Перечисли 5 планет Солнечной системы одним предложением."


def timed_completion(prompt: str) -> tuple[str, float, int]:
    client = get_client()
    model = first_model_id(client)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    elapsed = time.perf_counter() - start
    text = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
    return text, elapsed, tokens


def sequential_throughput(n: int) -> float:
    total_tokens = 0
    start = time.perf_counter()
    for _ in range(n):
        _, _, toks = timed_completion(PROMPT)
        total_tokens += toks
    total_time = time.perf_counter() - start
    return total_tokens / total_time if total_time else 0.0


def parallel_throughput(n: int, workers: int = 4) -> float:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: timed_completion(PROMPT), range(n)))
    total_time = time.perf_counter() - start
    total_tokens = sum(toks for _, _, toks in results)
    return total_tokens / total_time if total_time else 0.0


def main() -> None:
    try:
        text, secs, toks = timed_completion(PROMPT)
        seq = sequential_throughput(4)
        par = parallel_throughput(4)
    except LMStudioUnavailableError as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Ответ: {text}")
    print(f"Один запрос: {secs:.2f} c, {toks} токенов, "
          f"{toks / secs:.1f} ток/сек (латентность)")
    print(f"Throughput последовательно: {seq:.1f} ток/сек")
    print(f"Throughput параллельно:     {par:.1f} ток/сек")
    print(
        "[OK] ex1_solution: параллельный throughput обычно ВЫШЕ "
        "последовательного — это и есть суть батчинга (§4)."
    )


if __name__ == "__main__":
    main()
