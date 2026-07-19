# Практика 02 — LLM

Опирается на теорию: [02-llm](../../02-llm/README.md).
**Требуется LM Studio** (chat-модель + embedding-модель). См.
[настройку окружения](../README.md#2-подготовка-lm-studio-для-llm-упражнений).

## Упражнения

### ex1_tokens_and_embeddings — эмбеддинги и поиск по смыслу

Получаете реальные эмбеддинги через `/v1/embeddings`, реализуете `cos_sim`
(§3 теории) и наивный «ближайший сосед». Демонстрирует главное свойство
эмбеддингов: близкий смысл → близкие векторы (фундамент RAG в разделе 04).

Нужна загруженная **embedding-модель** (например `nomic-embed-text`).

```bash
uv run python 02-llm/ex1_tokens_and_embeddings/work.py
```

### ex2_roles_and_temperature — роли и temperature

Собираете `messages` с ролями `system`/`user` (§8) и сравниваете разброс
ответов при `temperature=0` (детерминированно) и `temperature=1`
(вариативно) — параметр случайности из §5.

```bash
uv run python 02-llm/ex2_roles_and_temperature/work.py
```

> Как запускать упражнения (`setup_practice.py`, `work.py`, `solution.py`) —
> см. [общий README практики](../README.md#3-конвенция-файлов).
