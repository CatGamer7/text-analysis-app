"""Тесты прототипа без настоящих моделей и без GPU: модель подменяется поддельной.

Запуск: pytest -v
"""
import json
from pathlib import Path

import pytest

from model_runner import NLPPrototype

SCHEMA = json.loads((Path(__file__).parent.parent / "schema.json").read_text(encoding="utf-8"))


class FakePipe:
    """Ведёт себя как pipeline из transformers. fail(device, text) может бросить исключение."""

    def __init__(self, model_name, device, fail=None):
        self.model_name, self.device, self.fail = model_name, device, fail

    def __call__(self, text, **kwargs):
        if self.fail:
            self.fail(self.device, text)
        if self.model_name.startswith("ner"):
            i = text.find("Москв")
            return [] if i < 0 else [{"entity_group": "CITY", "start": i, "end": i + 6, "score": 0.99}]
        return [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.1}]


def make_runner(devices=("cuda", "cpu"), fail=None, load_fail=None):
    def loader(model_name, device):
        if load_fail:
            load_fail(device)
        return FakePipe(model_name, device, fail)
    return NLPPrototype(loader=loader, devices_fn=lambda: list(devices))


def check_schema(resp):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(resp, SCHEMA["response"])


# ---------------------------------------------------------------- обычная работа

def test_sentiment_default_model():
    resp = make_runner().process({"task": "sentiment", "language": "ru", "text": "Отлично"})
    check_schema(resp)
    assert resp["model"] == "sentiment-rubert-tiny-ru"
    assert resp["device"] == "cuda" and resp["fallback"] is False
    assert resp["results"][0] == {"label": "positive", "score": 0.9}


def test_ner_returns_entities_with_positions():
    text = "Я живу в Москве"
    resp = make_runner().process({"task": "ner", "language": "ru", "text": text})
    check_schema(resp)
    ent = resp["results"][0]
    assert ent["entity"] == "CITY" and text[ent["start"]:ent["end"]] == ent["word"] == "Москве"


def test_explicit_model():
    resp = make_runner().process({"task": "emotion", "language": "ru", "text": "Ура", "model": "emotion-xlmr-multi"})
    assert resp["model"] == "emotion-xlmr-multi"


def test_model_is_cached():
    calls = []
    runner = NLPPrototype(loader=lambda m, d: calls.append(d) or FakePipe(m, d), devices_fn=lambda: ["cpu"])
    for _ in range(3):
        runner.process({"task": "sentiment", "language": "ru", "text": "Отлично"})
    assert len(calls) == 1


# ---------------------------------------------------------------- fallback GPU → CPU

def test_cuda_requested_but_absent():
    resp = make_runner(devices=("cpu",)).process({"task": "sentiment", "language": "ru", "text": "x", "device": "cuda"})
    check_schema(resp)
    assert resp["device"] == "cpu" and resp["fallback"] is True


def test_auto_on_cpu_only_machine_is_not_fallback():
    resp = make_runner(devices=("cpu",)).process({"task": "sentiment", "language": "ru", "text": "x"})
    assert resp["device"] == "cpu" and resp["fallback"] is False


def test_gpu_fails_on_load():
    def load_fail(device):
        if device == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available")
    resp = make_runner(load_fail=load_fail).process({"task": "sentiment", "language": "ru", "text": "x"})
    check_schema(resp)
    assert resp["device"] == "cpu" and resp["fallback"] is True


def test_gpu_out_of_memory_during_inference():
    def fail(device, text):
        if device == "cuda" and text != "Проверка модели":
            raise RuntimeError("CUDA out of memory")
    resp = make_runner(fail=fail).process({"task": "sentiment", "language": "ru", "text": "длинный текст"})
    check_schema(resp)
    assert resp["device"] == "cpu" and resp["fallback"] is True and resp["results"]


# ---------------------------------------------------------------- ошибки

@pytest.mark.parametrize("request_, code", [
    ({"task": "summary", "language": "ru", "text": "x"}, "INVALID_INPUT"),
    ({"task": "ner", "language": "de", "text": "x"}, "INVALID_INPUT"),
    ({"task": "ner", "language": "ru", "text": "   "}, "INVALID_INPUT"),
    ({"task": "ner", "language": "ru", "text": "x", "device": "tpu"}, "INVALID_INPUT"),
    ({"task": "ner", "language": "ru", "text": "x", "model": "abc"}, "MODEL_NOT_FOUND"),
    ({"task": "ner", "language": "ru", "text": "x", "model": "sentiment-rubert-tiny-ru"}, "INVALID_INPUT"),
    ({"task": "sentiment", "language": "ru", "text": "x", "model": "sentiment-twitter-roberta-en"}, "INVALID_INPUT"),
    ("не объект", "INVALID_INPUT"),
])
def test_invalid_requests(request_, code):
    resp = make_runner().process(request_)
    check_schema(resp)
    assert resp["error"]["code"] == code


def test_download_error():
    def load_fail(device):
        raise OSError("Can't load model: no internet")
    resp = make_runner(load_fail=load_fail).process({"task": "sentiment", "language": "ru", "text": "x"})
    check_schema(resp)
    assert resp["error"]["code"] == "MODEL_LOAD_FAILED"


def test_model_broken_even_on_cpu():
    def load_fail(device):
        raise RuntimeError("weights are corrupted")
    resp = make_runner(load_fail=load_fail).process({"task": "sentiment", "language": "ru", "text": "x"})
    assert resp["error"]["code"] == "MODEL_LOAD_FAILED"


def test_inference_error_not_from_gpu():
    def fail(device, text):
        if text == "плохой":
            raise ValueError("tokenizer exploded")
    resp = make_runner(fail=fail).process({"task": "sentiment", "language": "ru", "text": "плохой"})
    check_schema(resp)
    assert resp["error"]["code"] == "INFERENCE_FAILED"


# ---------------------------------------------------------------- schema.json

def test_schema_examples_are_valid():
    jsonschema = pytest.importorskip("jsonschema")
    for name, example in SCHEMA["examples"].items():
        jsonschema.validate(example, SCHEMA["request" if name.endswith("_request") else "response"])
