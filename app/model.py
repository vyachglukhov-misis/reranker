import os
from sentence_transformers import CrossEncoder

CACHE_DIR = os.getenv("HF_HOME", "/data")
os.environ["HF_HOME"] = CACHE_DIR

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("Загружаем bge-reranker-v2-m3 на GPU (fp32)...")
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            max_length=512,
        )
        print("Реранкер готов")
    return _reranker
