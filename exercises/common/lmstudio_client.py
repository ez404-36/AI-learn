"""Единая точка конфигурации доступа к локальной LLM через LM Studio.

LM Studio поднимает OpenAI-совместимый сервер на http://localhost:1234/v1,
поэтому мы работаем с ним через официальную библиотеку `openai`, просто
подменив `base_url`. API-ключ серверу не нужен — передаём заглушку.

Все LLM-упражнения импортируют помощники отсюда, чтобы:
  * не хардкодить URL/ключ в каждом файле;
  * при недоступном сервере падать с понятным сообщением, а не глухим
    traceback из глубины `httpx`.
"""

from __future__ import annotations

import sys

import openai

BASE_URL: str = "http://localhost:1234/v1"
# LM Studio не проверяет ключ, но клиент `openai` требует непустую строку.
API_KEY: str = "lm-studio"


class LMStudioUnavailableError(RuntimeError):
    """LM Studio не запущен или не отвечает на BASE_URL."""


def get_client() -> "openai.OpenAI":
    """Вернуть настроенный на LM Studio клиент OpenAI.

    Сам вызов не ходит в сеть — соединение проверяется при первом запросе.
    Для явной ранней диагностики используйте `ensure_server()`.
    """
    return openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)


def ensure_server(client: "openai.OpenAI | None" = None) -> "openai.OpenAI":
    """Проверить, что сервер поднят, иначе — понятная ошибка.

    Делает лёгкий запрос `models.list()`. При отказе соединения бросает
    LMStudioUnavailableError с инструкцией, что запустить.
    """
    checked = client if client is not None else get_client()
    try:
        checked.models.list()
    except openai.APIConnectionError as exc:  # сервер не поднят / не тот порт
        raise LMStudioUnavailableError(
            "Не удалось подключиться к LM Studio на "
            f"{BASE_URL}.\n"
            "Запустите LM Studio → вкладка 'Developer' / 'Local Server' → "
            "Start Server (порт 1234) и загрузите модель.\n"
            "Для эмбеддингов дополнительно загрузите embedding-модель "
            "(например nomic-embed-text)."
        ) from exc
    return checked


def first_model_id(client: "openai.OpenAI | None" = None) -> str:
    """Вернуть id первой загруженной в LM Studio модели.

    Удобно, когда не хочется хардкодить имя модели: LM Studio принимает
    любое имя, но так метка в ответе будет соответствовать реальной модели.
    """
    checked = ensure_server(client)
    models = checked.models.list().data
    if not models:
        raise LMStudioUnavailableError(
            "LM Studio запущен, но ни одна модель не загружена. "
            "Загрузите chat-модель на вкладке Local Server."
        )
    return models[0].id


if __name__ == "__main__":
    # Быстрая диагностика окружения: `uv run python common/lmstudio_client.py`
    try:
        model = first_model_id()
    except LMStudioUnavailableError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] LM Studio доступен на {BASE_URL}, активная модель: {model}")
