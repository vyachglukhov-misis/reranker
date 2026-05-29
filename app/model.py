import os
from sentence_transformers import CrossEncoder

# Кэш HF — направляем на смонтированный том /data
CACHE_DIR = os.getenv("HF_HOME", "/data")
os.environ["HF_HOME"] = CACHE_DIR

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """
    Ленивая загрузка модели bge-reranker-v2-m3.

    CrossEncoder инстанцируется без torch_dtype — это обходит баг
    transformers с передачей dtype в __init__ XLM-RoBERTa (на нём горели
    с FlagEmbedding). Конвертируем в fp16 через .half() уже после загрузки.
    """
    global _reranker
    if _reranker is None:
        print("Загружаем bge-reranker-v2-m3 на GPU...")
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            max_length=8192,  # модель поддерживает до 8k токенов
        )
        # fp16 после загрузки — не через torch_dtype в конструкторе
        _reranker.model = _reranker.model.half()
        print("Реранкер готов")
    return _reranker
