import pytest
from app.services.inference import inference_service
from app.services.model_registry import model_registry


def test_onnx_inference_real_predictions():
    model = model_registry.get_model("rubert-sentiment-conversational")
    assert model is not None

    # Test negative sentence
    neg_resp = inference_service.probe(model, "какашки жопа говницо всё сломалось и ужасно")
    assert neg_resp.primary_label == "negative"
    assert neg_resp.confidence > 0.80
    assert neg_resp.device == "cpu (onnx)"
    assert neg_resp.is_mock is False

    # Test positive sentence
    pos_resp = inference_service.probe(model, "Проект получился великолепный, всё работает просто идеально!")
    assert pos_resp.primary_label == "positive"
    assert pos_resp.confidence > 0.80
    assert pos_resp.device == "cpu (onnx)"
    assert pos_resp.is_mock is False

    # Test neutral sentence
    neu_resp = inference_service.probe(model, "Сегодня 14 ноября, на улице плюс пять градусов.")
    assert neu_resp.primary_label == "neutral"
    assert neu_resp.confidence > 0.70
    assert neu_resp.device == "cpu (onnx)"
