from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_list_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["total"] >= 1
    assert any(m["id"] == "rubert-sentiment-conversational" for m in data["items"])


def test_filter_models_by_task_and_language():
    response = client.get("/api/v1/models?task_type=sentiment&language=ru")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    for item in data["items"]:
        assert item["task_type"] == "sentiment"
        assert "ru" in [lang.lower() for lang in item["languages"]]


def test_get_single_model():
    response = client.get("/api/v1/models/rubert-sentiment-conversational")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "rubert-sentiment-conversational"
    assert "negative" in data["labels"]
    assert "positive" in data["labels"]


def test_get_nonexistent_model():
    response = client.get("/api/v1/models/non-existent-model-xyz")
    assert response.status_code == 404
