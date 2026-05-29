# Reranker (FastAPI · bge-reranker-v2-m3)

Cross-encoder реранкер для второй стадии гибридного RAG. Реализован на **FastAPI** +
**sentence-transformers**, инференс через PyTorch + CUDA. Заменил предыдущую
реализацию на TEI (Text Embeddings Inference) — возвращаем кастомный Python-сервис.

## Модель

**BAAI/bge-reranker-v2-m3** — encoder-only cross-encoder, принимает пары
(query, passage) и возвращает sigmoid-score релевантности.

| Параметр            | Значение                        |
| ------------------- | ------------------------------- |
| Параметров          | ~568M                           |
| Контекст            | 8192 токенов                    |
| VRAM в fp16         | ~1 ГБ                           |
| Целевая видеокарта  | NVIDIA Tesla A10 (24 ГБ, sm_86) |

## Запуск на VM

```bash
cd ~/reranker

# Первый запуск — собрать образ и скачать веса (~1 ГБ в ~/models-cache)
docker compose up -d --build

# Последующие запуски (веса из кэша, ~1 мин)
docker compose up -d

# Логи
docker compose logs -f
```

## Проверка с Mac

```bash
# Healthcheck
curl http://localhost:8081/health
# → {"status":"ok","model_loaded":true}

# Реранкинг двух документов
curl -sX POST http://localhost:8081/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "limit concurrent promises",
    "documents": [
      "function pLimit(concurrency) { /* ... */ }",
      "function unrelated() { return 42 }"
    ]
  }'
# → {"scores":[0.95, 0.03]}  — оба в [0,1], порядок совпадает с documents
```

## API

```
POST /rerank
  Body: {
    "query": "string",
    "documents": ["doc1", "doc2", ...],
    "normalize": true
  }
  → {
    "scores": [0.95, 0.03, ...]
  }

GET /health
  → {"status": "ok", "model_loaded": true|false}
```

**Важное отличие от TEI**: `scores` в порядке входных `documents`, без сортировки
и без поля `index`. Это упрощает `indexer/embedder.py` — просто `zip(results, scores)`.

## Структура

```
reranker/
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI-приложение, /rerank, /health, lifespan
│   ├── model.py      # ленивая загрузка CrossEncoder + .half()
│   └── schemas.py    # RerankRequest / RerankResponse (Pydantic)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Известные проблемы

- **`dtype` kwarg в XLM-RoBERTa** — обойдён загрузкой в дефолтной precision с
  последующим `_reranker.model.half()` (не через `torch_dtype` в конструкторе).
- **`@app.on_event` deprecated** — используем `lifespan` контекст-менеджер.
- **OOM при длинных документах** — если упадёт, снизить `max_length` в `model.py`
  с 8192 до 1024.
