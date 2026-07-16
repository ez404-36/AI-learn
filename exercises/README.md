# Практические занятия

Код-упражнения к разделам [01–08](../README.md) базы знаний. Каждый раздел
теории имеет здесь зеркальную папку `NN-.../` с заданиями на Python.

Формат: теория даёт понятие → упражнение заставляет его **запрограммировать**.
Часть заданий работает полностью локально (чистый Python/NumPy/PyTorch), часть —
через локальную LLM в **LM Studio**.

## 1. Подготовка окружения: uv и Python 3.13

Окружение управляется [`uv`](https://docs.astral.sh/uv/) — быстрый и
воспроизводимый менеджер пакетов и версий Python.

```bash
# 1. Установить uv (если ещё нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Установить нужную версию Python (uv скачает её сам)
uv python install 3.13

# 3. Из директории exercises/ создать .venv и поставить зависимости
cd exercises
uv sync

# 4. Создать рабочие копии упражнений (work.py в каждой папке)
uv run python setup_practice.py
```

`uv sync` читает `pyproject.toml` + `uv.lock`, создаёт `.venv/` и ставит
зафиксированные версии — на любой машине окружение получится одинаковым.

`setup_practice.py` копирует каждую заготовку `template.py` в `work.py` —
именно в `work.py` вы пишете решение. Он игнорируется git, поэтому ваши
правки не засоряют `git status` (подробнее — в [разделе 3](#3-конвенция-файлов)).

Запуск упражнения — без ручной активации окружения:

```bash
uv run python 01-foundations/ex1_neuron/work.py
```

> Альтернатива: `source .venv/bin/activate`, затем обычный `python ...`.

Добавить новую зависимость при доработке задания:

```bash
uv add <пакет>          # обновит pyproject.toml и uv.lock
```

## 2. Подготовка LM Studio (для LLM-упражнений)

[LM Studio](https://lmstudio.ai/) поднимает **OpenAI-совместимый** локальный
сервер — мы обращаемся к нему обычной библиотекой `openai`, подменив `base_url`.

1. Установите LM Studio и откройте вкладку **Local Server** (иконка `Developer`).
2. Загрузите **chat-модель** (подойдёт любая небольшая инструктивная модель:
   Qwen2.5-7B-Instruct, Llama-3.1-8B и т.п.). Для раздела 03 (tool calling)
   выбирайте модель с поддержкой function calling.
3. Загрузите **embedding-модель** — нужна для разделов 02 и 04
   (например `nomic-embed-text` или `text-embedding-bge`).
4. Нажмите **Start Server**. По умолчанию адрес — `http://localhost:1234/v1`.
5. Проверьте доступность:

   ```bash
   uv run python common/lmstudio_client.py
   ```

   Ожидаемый вывод: `[OK] LM Studio доступен ... активная модель: <id>`.

Все LLM-скрипты используют общий помощник
[`common/lmstudio_client.py`](common/lmstudio_client.py): он хранит `BASE_URL`,
отдаёт настроенный `get_client()` и при недоступном сервере бросает понятную
ошибку вместо «сырого» traceback. Ключи OpenAI нигде не нужны.

## 3. Конвенция файлов

Каждое упражнение — **папка** `NN-.../exN_<тема>/` с тремя файлами:

| Файл | Назначение | В git |
|------|------------|-------|
| `template.py` | Заготовка с `TODO` — исходный образец задания. | да |
| `solution.py` | Эталонное решение — сверяйтесь после попытки. | да |
| `work.py` | Ваша рабочая копия: здесь вы пишете решение. | нет (в `.gitignore`) |

`work.py` создаётся из `template.py` скриптом-генератором. Запустите его
один раз из директории `exercises/`:

```bash
uv run python setup_practice.py          # создать work.py для всех упражнений
uv run python setup_practice.py --list   # показать список без создания
uv run python setup_practice.py --force  # пересоздать заготовки (сотрёт ваш work.py!)
```

Так ваши правки живут в **незатрекованном** `work.py` и не засоряют
`git status`, не конфликтуют при `git pull`. Все три файла содержат блок
`if __name__ == "__main__":` с самопроверкой (`assert`/печать), поэтому их
можно запускать напрямую:

```bash
uv run python 01-foundations/ex1_neuron/work.py       # ваша работа
uv run python 01-foundations/ex1_neuron/solution.py   # эталон
```

> Примеры кода — учебные: показывают *механику* понятия, а не production-код.

## 4. Карта практики

| # | Практика | Опирается на | LM Studio |
|---|----------|--------------|-----------|
| 01 | [Основы](01-foundations/README.md) — нейрон, цикл обучения | [теория 01](../01-foundations/README.md) | не нужен |
| 02 | [LLM](02-llm/README.md) — эмбеддинги, роли и temperature | [теория 02](../02-llm/README.md) | нужен (chat + embed) |
| 03 | [Агенты](03-agents/README.md) — tool calling, ReAct-цикл | [теория 03](../03-agents/README.md) | нужен (tool calling) |
| 04 | [RAG](04-rag/README.md) — мини-RAG-пайплайн | [теория 04](../04-rag/README.md) | нужен (chat + embed) |
| 05 | [Пайплайны](05-pipelines-frameworks/README.md) — LangChain, LangGraph, PyTorch | [теория 05](../05-pipelines-frameworks/README.md) | частично |
| 06 | [Адаптация](06-model-adaptation/README.md) — квантизация, LoRA, DPO | [теория 06](../06-model-adaptation/README.md) | не нужен |
| 07 | [Инференс](07-inference/README.md) — latency/throughput, стриминг | [теория 07](../07-inference/README.md) | нужен |
| 08 | [Безопасность](08-safety/README.md) — guardrails, инъекции, evals | [теория 08](../08-safety/README.md) | нужен |

Раздел 09 (глоссарий) — справочник, практики не имеет.
