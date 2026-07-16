# Практика 03 — Агенты

Опирается на теорию: [03-agents](../../03-agents/README.md).
**Требуется LM Studio** с моделью, поддерживающей function calling.

## Упражнения

### ex1_tool_calling — вызов инструментов

Описываете инструмент `calculator` по OpenAI tool-схеме, отправляете
вопрос в модель и вручную выполняете запрошенный `tool_call`, возвращая
результат сообщением с ролью `tool` (§2 теории). Ключевая идея: **LLM не
выполняет код сама** — она лишь просит рантайм выполнить инструмент.

Калькулятор реализован через `ast` (без `eval`) — рантайм контролирует,
что именно исполняется.

```bash
uv run python 03-agents/ex1_tool_calling/work.py
```

### ex2_react_loop — цикл ReAct

Реализуете полный цикл **Thought → Action → Observation** (§3) поверх
LM Studio: модель текстом пишет действие, ваш рантайм парсит его,
выполняет инструмент (`lookup`/`calc`) и возвращает Observation в контекст,
пока не появится `Final:`.

```bash
uv run python 03-agents/ex2_react_loop/work.py
```

> Как запускать упражнения (`setup_practice.py`, `work.py`, `solution.py`) —
> см. [общий README практики](../README.md#3-конвенция-файлов).

> Оба `solution.py` умеют проверить инструменты (calculator/run_tool)
> оффлайн, если LM Studio недоступен.
