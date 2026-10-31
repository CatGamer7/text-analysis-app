from typing import Dict, List, Optional
from app.schemas.model import ModelMetadata, ModelStatus, ModelTaskType


class ModelRegistry:
    """Registry maintaining catalog of available NLP models for social media analysis."""

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
        self._initialize_default_catalog()

    def _initialize_default_catalog(self) -> None:
        defaults = [
            ModelMetadata(
                id="rubert-sentiment-conversational",
                name="RuBERT Conversational Sentiment (Russian)",
                description="Анализ тональности русскоязычных текстов социальных сетей на базе RuBERT conversational.",
                task_type=ModelTaskType.SENTIMENT,
                languages=["ru"],
                labels=["negative", "neutral", "positive"],
                hf_repo_id="cointegrated/rubert-tiny-sentiment-balanced",
                max_sequence_length=512,
                default_sample_text="Проект получился отличный, интерфейс понятный и отзывчивый!",
                status=ModelStatus.READY
            ),
            ModelMetadata(
                id="roberta-sentiment-en",
                name="Twitter-RoBERTa Sentiment (English)",
                description="Анализ тональности англоязычных твитов и постов в соцсетях на базе RoBERTa.",
                task_type=ModelTaskType.SENTIMENT,
                languages=["en"],
                labels=["negative", "neutral", "positive"],
                hf_repo_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
                max_sequence_length=512,
                default_sample_text="This new update is fantastic! Really enjoying the user experience.",
                status=ModelStatus.READY
            ),
            ModelMetadata(
                id="rubert-toxicity-ru",
                name="RuBERT Toxicity & Hate Speech (Russian)",
                description="Детекция токсичности, оскорблений и агрессии в русскоязычных комментариях и сообществах.",
                task_type=ModelTaskType.TOXICITY,
                languages=["ru"],
                labels=["neutral", "toxic", "insult", "threat", "obscenity"],
                hf_repo_id="cointegrated/rubert-toxic-editor",
                max_sequence_length=512,
                default_sample_text="Какая потрясающая дискуссия, спасибо всем участникам за вежливость.",
                status=ModelStatus.READY
            ),
            ModelMetadata(
                id="rubert-emotion-ru",
                name="RuBERT Emotion Classifier (Russian)",
                description="Классификация базовых эмоций в русскоязычном пользовательском контенте.",
                task_type=ModelTaskType.EMOTION,
                languages=["ru"],
                labels=["neutral", "joy", "sadness", "anger", "fear", "surprise"],
                hf_repo_id="AnReal/rubert-tiny2-russian-emotion-detection",
                max_sequence_length=512,
                default_sample_text="Я так рад, что наконец-то удалось завершить это исследование!",
                status=ModelStatus.READY
            ),
            ModelMetadata(
                id="distilbert-emotion-en",
                name="DistilBERT Emotion Classifier (English)",
                description="Быстрая классификация эмоций на английском языке (joy, sadness, anger, fear, love, surprise).",
                task_type=ModelTaskType.EMOTION,
                languages=["en"],
                labels=["sadness", "joy", "love", "anger", "fear", "surprise"],
                hf_repo_id="bhadresh-psavani/distilbert-base-uncased-emotion",
                max_sequence_length=512,
                default_sample_text="I am amazed by the efficiency of this system!",
                status=ModelStatus.READY
            ),
        ]
        for m in defaults:
            self._models[m.id] = m

    def list_models(self) -> List[ModelMetadata]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self._models.get(model_id)

    def register_model(self, model: ModelMetadata) -> None:
        self._models[model.id] = model

    def model_exists(self, model_id: str) -> bool:
        return model_id in self._models


model_registry = ModelRegistry()
