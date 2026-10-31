from typing import List, Optional
from pydantic import BaseModel, Field


class LabelScore(BaseModel):
    label: str = Field(..., description="Class label name", examples=["positive"])
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized probability score (0.0 to 1.0)", examples=[0.9452])
    percentage: float = Field(..., ge=0.0, le=100.0, description="Human-friendly percentage (0 to 100%)", examples=[94.52])


class ProbeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to be analyzed by the model",
        examples=["Отличный сервис, всё работает быстро и без сбоев!"]
    )
    model_id: Optional[str] = Field(
        None,
        description="Optional model ID override if not specified in URL path",
        examples=["rubert-sentiment-conversational"]
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Limit the number of ranked label scores returned",
        examples=[3]
    )


class ProbeResponse(BaseModel):
    model_id: str = Field(..., description="ID of the model that executed the inference", examples=["rubert-sentiment-conversational"])
    text: str = Field(..., description="Analyzed input text")
    primary_label: str = Field(..., description="Top predicted class label", examples=["positive"])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability score of the primary label", examples=[0.9452])
    scores: List[LabelScore] = Field(..., description="Full or ranked list of label probabilities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds", examples=[14.8])
    tokens_count: int = Field(..., description="Estimated or actual token count of input", examples=[9])
    device: str = Field("cpu", description="Compute device used for inference (cpu / cuda)", examples=["cpu"])
    is_mock: bool = Field(False, description="Flag indicating if inference was simulated via mock engine", examples=[False])


class ProbeBatchItem(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=10000)


class ProbeBatchRequest(BaseModel):
    items: List[ProbeBatchItem] = Field(..., min_length=1, max_length=50, description="List of sample texts for batch testing")
    top_k: Optional[int] = Field(None, ge=1, le=50)


class ProbeBatchResponse(BaseModel):
    model_id: str
    results: List[ProbeResponse]
    total_latency_ms: float
