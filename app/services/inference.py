import sys
import time
import math
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

from app.core.config import settings
from app.schemas.model import ModelMetadata, ModelTaskType
from app.schemas.probe import LabelScore, ProbeResponse


def get_asset_path(rel_path: str) -> Path:
    """Resolve asset paths in both development and PyInstaller bundled mode."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent.parent
    p = base / rel_path
    if p.exists():
        return p
    cwd_path = Path.cwd() / rel_path
    if cwd_path.exists():
        return cwd_path
    return p


class InferenceService:
    """Service responsible for model inference execution in probing and evaluation."""

    def __init__(self):
        self._loaded_pipelines = {}
        self._onnx_sessions: Dict[str, Tuple[object, object, Dict[int, str]]] = {}
        self._init_onnx_models()

    def _init_onnx_models(self) -> None:
        """Pre-load local ONNX models for high-performance CPU inference."""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_dir = get_asset_path("models/rubert_tiny_sentiment")
            model_path = model_dir / "onnx" / "model_quantized.onnx"
            if not model_path.exists():
                model_path = model_dir / "onnx" / "model.onnx"
            tokenizer_path = model_dir / "tokenizer.json"

            if model_path.exists() and tokenizer_path.exists():
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                tokenizer.enable_truncation(max_length=512)

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )

                # id2label for rubert-tiny-sentiment-balanced
                id2label = {0: "negative", 1: "neutral", 2: "positive"}
                self._onnx_sessions["rubert-sentiment-conversational"] = (session, tokenizer, id2label)
        except Exception:
            # Fall back gracefully if ONNX libraries or weights are absent
            pass

    def _onnx_predict(self, model_id: str, text: str) -> Optional[List[LabelScore]]:
        """Execute inference using the local ONNX runtime."""
        if model_id not in self._onnx_sessions:
            return None

        session, tokenizer, id2label = self._onnx_sessions[model_id]

        enc = tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        input_names = [inp.name for inp in session.get_inputs()]
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = token_type_ids

        outputs = session.run(None, feeds)[0]
        logits = outputs[0]

        # Numerically stable Softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        result = []
        for idx, prob in enumerate(probs):
            label_name = id2label.get(idx, f"class_{idx}")
            val = round(float(prob), 4)
            result.append(
                LabelScore(
                    label=label_name,
                    score=val,
                    percentage=round(val * 100, 2)
                )
            )

        result.sort(key=lambda x: x.score, reverse=True)
        return result

    def _mock_predict(self, model: ModelMetadata, text: str) -> List[LabelScore]:
        """
        Deterministic heuristic inference engine for testing and prototyping without torch weights.
        Produces dynamic probabilities based on input sentiment cues and stable hashing.
        """
        text_lower = text.lower()
        labels = model.labels
        scores = {}

        if model.task_type == ModelTaskType.SENTIMENT:
            pos_words = [
                "отличн", "хорош", "супер", "класс", "любл", "прекрасн", "замечательн", "восторг", 
                "лучш", "кайф", "топ", "огонь", "good", "great", "love", "awesome", "fantastic", "best"
            ]
            neg_words = [
                "плох", "ужасн", "отвратительн", "ненавиж", "сломал", "говно", "говн", "жоп", 
                "какашк", "хрен", "дрянь", "мусор", "фигн", "дерьм", "отстой", "бесит", "туп", 
                "bad", "terrible", "awful", "hate", "worst", "fail", "crap", "shit"
            ]

            pos_score = sum(1 for w in pos_words if w in text_lower)
            neg_score = sum(1 for w in neg_words if w in text_lower)

            # Generate pseudo-random deterministic jitter between 0.01 and 0.08
            h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:6], 16)
            jitter = (h % 70) / 1000.0

            if pos_score > neg_score:
                top_p = round(0.85 + jitter, 4)
                rem = round((1.0 - top_p) / 2.0, 4)
                weights = {"positive": top_p, "neutral": rem, "negative": round(1.0 - top_p - rem, 4)}
            elif neg_score > pos_score:
                top_p = round(0.88 + jitter, 4)
                rem = round((1.0 - top_p) / 2.0, 4)
                weights = {"negative": top_p, "neutral": rem, "positive": round(1.0 - top_p - rem, 4)}
            else:
                base_neu = round(0.55 + ((h % 20) / 100.0), 4)
                rem = round((1.0 - base_neu) / 2.0, 4)
                weights = {"neutral": base_neu, "positive": rem, "negative": round(1.0 - base_neu - rem, 4)}

            scores = {label: weights.get(label, 0.33) for label in labels}

        elif model.task_type == ModelTaskType.TOXICITY:
            toxic_words = ["дурак", "идиот", "мразь", "урод", "убью", "жоп", "дерьм", "trash", "idiot", "hate", "kill"]
            has_toxic = any(w in text_lower for w in toxic_words)
            if has_toxic:
                scores = {"toxic": 0.82, "insult": 0.68, "threat": 0.15, "obscenity": 0.45, "neutral": 0.05}
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
            h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            primary_idx = h % len(labels)
            scores = {label: 0.1 for label in labels}
            scores[labels[primary_idx]] = 0.8

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
            return self._mock_predict(model, text)

    def probe(self, model: ModelMetadata, text: str, top_k: Optional[int] = None) -> ProbeResponse:
        start_time = time.perf_counter()

        is_mock = True
        device = "cpu"
        scores = None

        # 1. High-speed local ONNX inference
        if model.id in self._onnx_sessions:
            try:
                scores = self._onnx_predict(model.id, text)
                if scores:
                    is_mock = False
                    device = "cpu (onnx)"
            except Exception:
                scores = None

        # 2. HuggingFace Transformers pipeline (if explicitly configured)
        if scores is None and not settings.use_mock_inference:
            try:
                scores = self._real_predict(model, text)
                is_mock = False
                device = "cpu (pytorch)"
            except Exception:
                scores = None

        # 3. Deterministic mock fallback
        if scores is None:
            scores = self._mock_predict(model, text)
            is_mock = True
            device = "cpu (heuristic)"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
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
            device=device,
            is_mock=is_mock
        )


inference_service = InferenceService()
