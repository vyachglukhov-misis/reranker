from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос пользователя")
    documents: list[str] = Field(..., description="Список passage'ей для реранкинга")
    normalize: bool = Field(
        default=True,
        description="Применить sigmoid: scores в [0, 1]. False — сырые логиты.",
    )


class RerankResponse(BaseModel):
    scores: list[float] = Field(
        ...,
        description="Релевантность каждого документа. Порядок совпадает с documents.",
    )
