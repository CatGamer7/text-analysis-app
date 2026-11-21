"""Прототип инференса NLP-моделей.

Загружает модель, принимает текст, возвращает метки с оценками (или сущности для NER).
Если GPU недоступен или на нём произошла ошибка — автоматически переходит на CPU (fallback).

Запуск:
    python model_runner.py                       # демо на трёх примерах
    python model_runner.py --list                # список моделей
    python model_runner.py --task sentiment --language ru --text "Отличный сервис!"
    python model_runner.py --task ner --language en --text "Elon Musk founded SpaceX" --model ner-wikineural-multi
    python model_runner.py --request request.json
"""
import argparse
import json
import logging
import time
import warnings
from typing import Callable, Dict, List, Optional

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

from models import DEFAULTS, MODELS  # noqa: E402

logger = logging.getLogger("nlp")

TASKS = ("ner", "sentiment", "emotion")
LANGUAGES = ("ru", "en")
DEVICES = ("auto", "cpu", "cuda", "mps")
MAX_LENGTH = 512


class NLPError(Exception):
    """Ошибка с кодом из schema.json: INVALID_INPUT, MODEL_NOT_FOUND, MODEL_LOAD_FAILED, INFERENCE_FAILED."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ----------------------------------------------------------------------------- устройства

def available_devices() -> List[str]:
    """Доступные устройства в порядке приоритета: cuda → mps → cpu."""
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def is_device_error(exc: Exception) -> bool:
    """Ошибка видеокарты (нехватка памяти, сбой CUDA/MPS) — такую лечим переходом на CPU."""
    msg = str(exc).lower()
    return any(m in msg for m in ("out of memory", "cuda", "cudnn", "cublas", "mps", "device-side", "no kernel image"))


# ----------------------------------------------------------------------------- загрузка модели

def _to_bio(label: str) -> str:
    """BIOLU → BIO. Модель Gherman/bert-base-NER-Russian размечена в BIOLU (U-CITY, L-CITY),
    а стандартная склейка сущностей в transformers понимает только B-/I-."""
    if len(label) > 2 and label[1] == "-":
        if label[0] in "US":
            return "B-" + label[2:]
        if label[0] in "LE":
            return "I-" + label[2:]
    return label


def load_tokenizer(hf_name: str):
    """AutoTokenizer с обходом несовместимости: токенизаторы, сохранённые transformers 5.x,
    хранят extra_special_tokens списком, а transformers 4.x ждёт словарь
    ('list' object has no attribute 'keys')."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(hf_name)
    except AttributeError as e:
        if "keys" not in str(e):
            raise
        return AutoTokenizer.from_pretrained(hf_name, extra_special_tokens={})


def load_pipeline(model_name: str, device: str):
    """Загружает модель с Hugging Face (или из кэша) на указанное устройство."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline

    spec = MODELS[model_name]
    tokenizer = load_tokenizer(spec["hf"])
    if not tokenizer.model_max_length or tokenizer.model_max_length > MAX_LENGTH:
        tokenizer.model_max_length = MAX_LENGTH

    if spec["task"] == "ner":
        model = AutoModelForTokenClassification.from_pretrained(spec["hf"])
        id2label = {int(i): _to_bio(lbl) for i, lbl in model.config.id2label.items()}
        model.config.id2label = id2label
        model.config.label2id = {lbl: i for i, lbl in id2label.items()}
        return pipeline("token-classification", model=model, tokenizer=tokenizer,
                        aggregation_strategy="simple", device=torch.device(device))

    model = AutoModelForSequenceClassification.from_pretrained(spec["hf"])
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=torch.device(device))


def _twitter_preprocess(text: str) -> str:
    """Как в карточке cardiffnlp: @упоминания → @user, ссылки → http."""
    return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w
                    for w in text.split(" "))


# ----------------------------------------------------------------------------- прототип

class NLPPrototype:
    def __init__(self, loader: Callable = load_pipeline, devices_fn: Callable[[], List[str]] = available_devices):
        self._loader = loader          # в тестах подменяется на поддельный
        self._devices_fn = devices_fn
        self._cache: Dict[tuple, dict] = {}   # (модель, устройство из запроса) → {pipe, device, fallback}

    def process(self, request: dict) -> dict:
        """Запрос по schema.json → ответ по schema.json. Исключений наружу не бросает."""
        try:
            return self._process(request)
        except NLPError as e:
            return {"error": {"code": e.code, "message": e.message}}
        except Exception as e:  # noqa: BLE001
            logger.exception("Непредвиденная ошибка")
            return {"error": {"code": "INFERENCE_FAILED", "message": f"{type(e).__name__}: {e}"}}

    # --- шаги обработки

    def _process(self, request: dict) -> dict:
        task, language, text, device, model_name = self._validate(request)
        entry = self._get_model(model_name, device)

        start = time.perf_counter()
        try:
            results = self._predict(entry["pipe"], model_name, text)
        except Exception as e:  # noqa: BLE001
            if entry["device"] != "cpu" and is_device_error(e):
                # Ошибка видеокарты во время работы → переносим модель на CPU и повторяем
                logger.warning("Ошибка на %s (%s) → повтор на CPU", entry["device"], e)
                entry = self._load(model_name, device, ["cpu"], fallback=True)
                start = time.perf_counter()
                try:
                    results = self._predict(entry["pipe"], model_name, text)
                except Exception as e2:  # noqa: BLE001
                    raise NLPError("INFERENCE_FAILED", f"Ошибка инференса: {e2}")
            else:
                raise NLPError("INFERENCE_FAILED", f"Ошибка инференса: {e}")
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "task": task,
            "language": language,
            "model": model_name,
            "device": entry["device"],
            "fallback": entry["fallback"],
            "inference_time_ms": round(elapsed_ms, 2),
            "results": results,
        }

    def _validate(self, request: dict):
        if not isinstance(request, dict):
            raise NLPError("INVALID_INPUT", "Запрос должен быть JSON-объектом")
        task, language, text = request.get("task"), request.get("language"), request.get("text")
        device = request.get("device", "auto")
        if task not in TASKS:
            raise NLPError("INVALID_INPUT", f"Поле task должно быть одним из {TASKS}")
        if language not in LANGUAGES:
            raise NLPError("INVALID_INPUT", f"Поле language должно быть одним из {LANGUAGES}")
        if not isinstance(text, str) or not text.strip():
            raise NLPError("INVALID_INPUT", "Поле text пустое")
        if device not in DEVICES:
            raise NLPError("INVALID_INPUT", f"Поле device должно быть одним из {DEVICES}")

        model_name = request.get("model") or DEFAULTS[(task, language)]
        spec = MODELS.get(model_name)
        if spec is None:
            raise NLPError("MODEL_NOT_FOUND", f"Модель '{model_name}' не найдена. Список: python model_runner.py --list")
        if spec["task"] != task:
            raise NLPError("INVALID_INPUT", f"Модель '{model_name}' решает задачу '{spec['task']}', а не '{task}'")
        if language not in spec["languages"]:
            raise NLPError("INVALID_INPUT", f"Модель '{model_name}' не поддерживает язык '{language}'")
        return task, language, text, device, model_name

    def _get_model(self, model_name: str, device: str) -> dict:
        key = (model_name, device)
        if key in self._cache:
            return self._cache[key]

        available = self._devices_fn()
        if device == "auto":
            chain, fallback = available, False
        elif device == "cpu":
            chain, fallback = ["cpu"], False
        elif device in available:
            chain, fallback = [device, "cpu"], False
        else:
            logger.warning("Устройство %s недоступно → CPU", device)
            chain, fallback = ["cpu"], True
        return self._load(model_name, device, chain, fallback)

    def _load(self, model_name: str, requested: str, chain: List[str], fallback: bool) -> dict:
        """Пробует устройства по очереди; CPU — последний вариант."""
        last_error: Optional[Exception] = None
        for dev in chain:
            try:
                pipe = self._loader(model_name, dev)
                self._predict(pipe, model_name, "Проверка модели")   # пробный прогон ловит сбои GPU сразу
                entry = {"pipe": pipe, "device": dev, "fallback": fallback}
                self._cache[(model_name, requested)] = entry
                logger.info("Модель %s загружена на %s", model_name, dev)
                return entry
            except OSError as e:   # не скачалась / нет файлов — другое устройство не поможет
                raise NLPError("MODEL_LOAD_FAILED", f"Не удалось скачать или прочитать модель {MODELS[model_name]['hf']}: {e}")
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning("Не удалось запустить %s на %s: %s", model_name, dev, e)
                fallback = True
        raise NLPError("MODEL_LOAD_FAILED", f"Модель {model_name} не запускается: {last_error}")

    def _predict(self, pipe, model_name: str, text: str) -> list:
        spec = MODELS[model_name]
        if spec["task"] == "ner":
            kwargs = {"stride": MAX_LENGTH // 4} if getattr(getattr(pipe, "tokenizer", None), "is_fast", False) else {}
            raw = pipe(text, **kwargs)
            return [
                {"entity": r["entity_group"], "word": text[r["start"]:r["end"]],
                 "start": int(r["start"]), "end": int(r["end"]), "score": round(float(r["score"]), 4)}
                for r in raw
            ]

        if spec["hf"].startswith("cardiffnlp/"):
            text = _twitter_preprocess(text)
        raw = pipe(text, top_k=None, truncation=True, max_length=MAX_LENGTH,
                   function_to_apply="sigmoid" if spec.get("multi_label") else "softmax")
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        results = [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in raw]
        return sorted(results, key=lambda r: r["score"], reverse=True)


# ----------------------------------------------------------------------------- CLI

DEMO = [
    {"task": "ner", "language": "ru", "text": "Иван Петров живёт в Москве"},
    {"task": "sentiment", "language": "ru", "text": "Отличный сервис, очень доволен!"},
    {"task": "emotion", "language": "en", "text": "I am so happy today!", "device": "cuda"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Прототип инференса: NER, тональность, эмоции")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--language", choices=LANGUAGES)
    parser.add_argument("--text")
    parser.add_argument("--model", help="название из models.py (необязательно)")
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--request", help="JSON-файл с запросом")
    parser.add_argument("--list", action="store_true", help="показать список моделей")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог (видно fallback)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    if args.list:
        for name, spec in MODELS.items():
            print(f"{name:30s} {spec['task']:10s} {','.join(spec['languages']):6s} {spec['hf']}")
        return

    runner = NLPPrototype()
    if args.request:
        with open(args.request, encoding="utf-8") as f:
            requests = [json.load(f)]
    elif args.task:
        req = {"task": args.task, "language": args.language, "text": args.text, "device": args.device}
        if args.model:
            req["model"] = args.model
        requests = [req]
    else:
        requests = DEMO

    for req in requests:
        print("Запрос:", json.dumps(req, ensure_ascii=False))
        print(json.dumps(runner.process(req), ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
