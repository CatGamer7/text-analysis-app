from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ModelTaskType(str, Enum):
    SENTIMENT = "sentiment"
    EMOTION = "emotion"
    TOXICITY = "toxicity"
    TOPIC = "topic_classification"


class ModelStatus(str, Enum):
    READY = "ready"
    DOWNLOADING = "downloading"
    NOT_INSTALLED = "not_installed"
    ERROR = "error"


class ModelMetadata(BaseModel):
    id: str = Field(..., description="Unique model identifier (slug)", examples=["rubert-sentiment-conversational"])
    name: str = Field(..., description="Human-readable model name", examples=["RuBERT Sentiment Conversational"])
    description: str = Field(..., description="Brief model description and intended use cases")
    task_type: ModelTaskType = Field(..., description="Primary NLP task performed by the model")
    languages: List[str] = Field(..., description="Supported languages (e.g. ['ru'], ['en'])", examples=[["ru"]])
    labels: List[str] = Field(..., description="Output class labels", examples=[["negative", "neutral", "positive"]])
    hf_repo_id: Optional[str] = Field(None, description="HuggingFace model repository ID", examples=["sberbank-ai/sbert_large_nlu_ru"])
    max_sequence_length: int = Field(512, description="Maximum input sequence length in tokens", examples=[512])
    default_sample_text: Optional[str] = Field(None, description="Default sample text for testing in UI")
    status: ModelStatus = Field(ModelStatus.READY, description="Current availability status of the model")


class ModelListResponse(BaseModel):
    items: List[ModelMetadata] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of registered models")
