from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from app.model import get_reranker
from app.schemas import RerankRequest, RerankResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_reranker()
    yield


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

    # CrossEncoder.predict() в ST 3.0.1 уже применяет sigmoid внутри
    # для моделей с num_labels=1 (bge-reranker-v2-m3). Не применяем sigmoid повторно.
    scores: np.ndarray = reranker.predict(
        pairs,
        batch_size=8,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return RerankResponse(scores=scores.tolist())
