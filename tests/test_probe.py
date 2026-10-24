from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_probe_model_path_endpoint():
    payload = {
        "text": "Мне очень понравился этот исследовательский проект!",
        "top_k": 3
    }
    response = client.post("/api/v1/models/rubert-sentiment-conversational/probe", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "rubert-sentiment-conversational"
    assert data["text"] == payload["text"]
    assert "primary_label" in data
    assert "confidence" in data
    assert len(data["scores"]) <= 3
    assert data["latency_ms"] >= 0


def test_probe_generic_endpoint():
    payload = {
        "model_id": "roberta-sentiment-en",
        "text": "The results of this analysis are outstanding and very promising."
    }
    response = client.post("/api/v1/probe", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "roberta-sentiment-en"
    assert data["primary_label"] in ["positive", "neutral", "negative"]


def test_probe_nonexistent_model():
    payload = {"text": "Тестовый текст"}
    response = client.post("/api/v1/models/unknown-model/probe", json=payload)
    assert response.status_code == 404


def test_probe_batch_endpoint():
    payload = {
        "items": [
            {"id": "1", "text": "Текст с положительной окраской."},
            {"id": "2", "text": "Текст с негативной окраской и критикой."}
        ]
    }
    response = client.post("/api/v1/models/rubert-sentiment-conversational/probe/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "rubert-sentiment-conversational"
    assert len(data["results"]) == 2
    assert "total_latency_ms" in data
