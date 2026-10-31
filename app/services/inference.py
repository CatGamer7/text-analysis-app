import time
import math
import hashlib
from typing import List, Optional
from app.core.config import settings
from app.schemas.model import ModelMetadata, ModelTaskType
from app.schemas.probe import LabelScore, ProbeResponse


class InferenceService:
    """Service responsible for model inference execution in probing and evaluation."""

    def __init__(self):
        self._loaded_pipelines = {}

    def _mock_predict(self, model: ModelMetadata, text: str) -> List[LabelScore]:
        """
        Deterministic heuristic inference engine for testing and prototyping without torch weights.
        Produces realistic probabilities based on input sentiment cues and stable hashing.
        """
        text_lower = text.lower()
        labels = model.labels
        scores = {}

        if model.task_type == ModelTaskType.SENTIMENT:
            pos_words = ["отличн", "хорош", "супер", "класс", "любл", "прекрасн", "good", "great", "love", "awesome", "fantastic", "best"]
            neg_words = ["плох", "ужасн", "отвратительн", "ненавиж", "сломал", "bad", "terrible", "awful", "hate", "worst", "fail"]
            
            pos_score = sum(1 for w in pos_words if w in text_lower)
            neg_score = sum(1 for w in neg_words if w in text_lower)
            
            if pos_score > neg_score:
                weights = {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
            elif neg_score > pos_score:
                weights = {"negative": 0.85, "neutral": 0.10, "positive": 0.05}
            else:
                weights = {"neutral": 0.70, "positive": 0.15, "negative": 0.15}
            
            scores = {label: weights.get(label, 0.33) for label in labels}

        elif model.task_type == ModelTaskType.TOXICITY:
            toxic_words = ["дурак", "идиот", "мразь", "урод", "убью", "trash", "idiot", "hate", "kill"]
            has_toxic = any(w in text_lower for w in toxic_words)
            if has_toxic:
                scores = {"toxic": 0.75, "insult": 0.65, "threat": 0.15, "obscenity": 0.40, "neutral": 0.05}
            else:
                scores = {"neutral": 0.96, "toxic": 0.02, "insult": 0.01, "threat": 0.005, "obscenity": 0.005}

        elif model.task_type == ModelTaskType.EMOTION:
            if any(w in text_lower for w in ["рад", "счастлив", "ура", "joy", "happy", "great"]):
                primary = "joy"
            elif any(w in text_lower for w in ["грустн", "печал", "тоск", "sad", "unhappy"]):
                primary = "sadness"
            elif any(w in text_lower for w in ["зол", "бесит", "ярост", "angry", "furious"]):
                primary = "anger"
            elif any(w in text_lower for w in ["боюсь", "страшн", "ужас", "fear", "afraid"]):
                primary = "fear"
            elif any(w in text_lower for w in ["вау", "ух ты", "шок", "surprise", "amazed"]):
                primary = "surprise"
            else:
                primary = "neutral" if "neutral" in labels else labels[0]

            scores = {l: 0.05 for l in labels}
            scores[primary] = 0.75

        else:
            # Deterministic pseudorandom distribution based on MD5 of text
            h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            primary_idx = h % len(labels)
            scores = {label: 0.1 for label in labels}
            scores[labels[primary_idx]] = 0.8

        # Normalize so sum = 1.0
        total = sum(scores.values()) or 1.0
        result = []
        for label in labels:
            val = round(scores.get(label, 0.0) / total, 4)
            result.append(
                LabelScore(
                    label=label,
                    score=val,
                    percentage=round(val * 100, 2)
                )
            )

        # Sort descending by score
        result.sort(key=lambda x: x.score, reverse=True)
        return result

    def _real_predict(self, model: ModelMetadata, text: str) -> List[LabelScore]:
        """Execution using HuggingFace Transformers pipeline when available."""
        try:
            from transformers import pipeline
            if model.id not in self._loaded_pipelines:
                repo_id = model.hf_repo_id or model.id
                self._loaded_pipelines[model.id] = pipeline(
                    "text-classification",
                    model=repo_id,
                    return_all_scores=True
                )
            pipe = self._loaded_pipelines[model.id]
            raw_output = pipe(text[:model.max_sequence_length])[0]
            
            result = []
            for item in raw_output:
                score = round(float(item["score"]), 4)
                result.append(
                    LabelScore(
                        label=item["label"],
                        score=score,
                        percentage=round(score * 100, 2)
                    )
                )
            result.sort(key=lambda x: x.score, reverse=True)
            return result
        except Exception:
            # Fall back to mock if torch/transformers isn't installed
            return self._mock_predict(model, text)

    def probe(self, model: ModelMetadata, text: str, top_k: Optional[int] = None) -> ProbeResponse:
        start_time = time.perf_counter()
        
        # Decide between real inference and mock mode
        use_mock = settings.use_mock_inference
        if not use_mock:
            try:
                import transformers  # noqa: F401
                scores = self._real_predict(model, text)
                is_mock = False
            except ImportError:
                scores = self._mock_predict(model, text)
                is_mock = True
        else:
            scores = self._mock_predict(model, text)
            is_mock = True

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        # Approximate tokens as whitespace word splits + punctuation heuristic
        tokens_count = len(text.split())

        if top_k and top_k > 0:
            filtered_scores = scores[:top_k]
        else:
            filtered_scores = scores

        primary = scores[0]

        return ProbeResponse(
            model_id=model.id,
            text=text,
            primary_label=primary.label,
            confidence=primary.score,
            scores=filtered_scores,
            latency_ms=elapsed_ms,
            tokens_count=tokens_count,
            device="cpu",
            is_mock=is_mock
        )


inference_service = InferenceService()
