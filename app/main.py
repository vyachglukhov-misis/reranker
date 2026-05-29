from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from app.model import get_reranker
from app.schemas import RerankRequest, RerankResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Стартап: прогреваем модель, чтобы первый запрос не ждал загрузки весов.
    get_reranker()
    yield
    # Шатдаун: PyTorch освобождает VRAM при завершении процесса.


app = FastAPI(
    title="Reranker",
    description="FastAPI-сервис реранкинга через BAAI/bge-reranker-v2-m3 (cross-encoder)",
    version="2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    from app import model as m
    return {"status": "ok", "model_loaded": m._reranker is not None}


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    reranker = get_reranker()

    pairs = [[req.query, doc] for doc in req.documents]

    raw_scores: np.ndarray = reranker.predict(
        pairs,
        batch_size=8,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if req.normalize:
        # Sigmoid: 1 / (1 + e^-x) → scores в [0, 1]
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        scores = raw_scores

    # Порядок scores совпадает с порядком documents — indexer не перекладывает индексы.
    return RerankResponse(scores=scores.tolist())
