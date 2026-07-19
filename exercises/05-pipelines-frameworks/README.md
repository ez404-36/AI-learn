# Практика 05 — Пайплайны и фреймворки

Опирается на теорию: [05-pipelines-frameworks](../../05-pipelines-frameworks/README.md).
Частично требует LM Studio (ex1, ex2); ex3 работает локально.

## Упражнения

### ex1_langchain_chain — цепочка LangChain

Собираете двухшаговую цепочку через оператор `|` (§3): `ChatPromptTemplate
| ChatOpenAI | StrOutputParser`. `ChatOpenAI` направлен в LM Studio через
`base_url`. Требует LM Studio (chat).

```bash
uv run python 05-pipelines-frameworks/ex1_langchain_chain/work.py
```

### ex2_langgraph_agent — граф с условным ребром

Реализуете граф состояний LangGraph (§4): узлы `think`/`tool` и **условное
ребро** `think → tool → think`, повторяющее ReAct-цикл из раздела 03, пока
модель не даст `Final:`. Требует LM Studio (chat).

```bash
uv run python 05-pipelines-frameworks/ex2_langgraph_agent/work.py
```

### ex3_pytorch_training — реальный шаг обучения на PyTorch

Тот же цикл обучения, что в [01/ex2](../01-foundations/README.md), но на
настоящем PyTorch (§6): `loss.backward()` (autograd) + `optimizer.step()`.
LM Studio **не нужен**.

```bash
uv run python 05-pipelines-frameworks/ex3_pytorch_training/work.py
```

> Связь с 01: 01/ex2 считает градиенты руками (концептуально), здесь —
> autograd делает это за вас (инженерно).

> Как запускать упражнения (`setup_practice.py`, `work.py`, `solution.py`) —
> см. [общий README практики](../README.md#3-конвенция-файлов).
