# Диаграммы

Здесь лежат исходники всех диаграмм базы знаний в формате **draw.io** (`.drawio`). Файлы — это несжатый XML, его можно открыть и редактировать:

- онлайн на [app.diagrams.net](https://app.diagrams.net) (File → Open);
- в VS Code через расширение **Draw.io Integration** (Hediet);
- в десктопном приложении draw.io.

## Соглашение об именовании

```
<номер раздела>-<краткое имя>.drawio
```

Например, `04-rag-full.drawio` — диаграмма «полный RAG-пайплайн» из раздела 04.

## Как картинки попадают в тексты

В Markdown-файлах диаграммы подключаются как PNG:

```markdown
![Описание](../diagrams/04-rag-full.drawio.png)
```

Файлы `.drawio.png` нужно **экспортировать** из соответствующего `.drawio` (в draw.io: File → Export as → PNG, имя `<тот же>.drawio.png`). Сами PNG в репозиторий не закоммичены намеренно: они генерируются из исходников и легко устаревают. Откройте `.drawio` напрямую, чтобы увидеть актуальную версию диаграммы.

## Список диаграмм по разделам

### 01. Основы
- `01-ai-ml-dl.drawio` — вложенность AI ⊃ ML ⊃ Deep Learning
- `01-rules-vs-ml.drawio` — классическое программирование vs ML
- `01-neural-network.drawio` — структура нейросети по слоям
- `01-relu.drawio` — график функции активации ReLU (z → выход)
- `01-training-loop.drawio` — цикл обучения
- `01-training-vs-inference.drawio` — обучение vs использование

### 02. LLM
- `02-next-token.drawio` — предсказание следующего токена
- `02-tokenization.drawio` — токенизация текста
- `02-embeddings.drawio` — смысловое пространство embeddings
- `02-attention.drawio` — механизм внимания
- `02-generation-loop.drawio` — цикл авторегрессии
- `02-context-window.drawio` — контекстное окно
- `02-training-stages.drawio` — этапы обучения LLM
- `02-prompt-roles.drawio` — структура промпта по ролям

### 03. Агенты
- `03-llm-to-agent.drawio` — от LLM к агенту
- `03-tool-calling.drawio` — шаги tool calling
- `03-react-loop.drawio` — цикл ReAct
- `03-instruction-layers.drawio` — три слоя инструкций агента (system prompt / rules / запрос)
- `03-skill-disclosure.drawio` — три уровня загрузки Skill (progressive disclosure)
- `03-mcp-problem.drawio` — проблема интеграций без MCP
- `03-mcp-architecture.drawio` — архитектура MCP
- `03-acp-overview.drawio` — общение агентов через ACP
- `03-mcp-vs-acp.drawio` — сравнение MCP и ACP
- `03-multiagent-patterns.drawio` — схемы мультиагентных систем

### 04. RAG
- `04-rag-problem.drawio` — проблема, которую решает RAG
- `04-vector-search.drawio` — семантический поиск
- `04-indexing.drawio` — этап индексации
- `04-query-pipeline.drawio` — этап запроса
- `04-rag-full.drawio` — полный RAG-пайплайн
- `04-chunking.drawio` — стратегии чанкинга
- `04-rag-vs-finetune.drawio` — RAG vs fine-tuning vs контекст
- `04-rag-improvements.drawio` — re-ranking и hybrid search

### 05. Пайплайны и фреймворки
- `05-pipeline-concept.drawio` — понятие пайплайна
- `05-llm-pipeline.drawio` — LLM-пайплайн и цепочка
- `05-langchain.drawio` — компоненты LangChain
- `05-langgraph.drawio` — граф в LangGraph
- `05-pytorch-level.drawio` — уровень PyTorch
- `05-stack-layers.drawio` — карта слоёв AI-стека
- `05-tool-choice.drawio` — дерево выбора инструмента
