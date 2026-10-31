import time
import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_upload_and_process_csv():
    csv_content = "text\nОтличная новость, всё работает прекрасно!\nУжасное качество обслуживания.\nОбычный нейтральный комментарий."
    files = {
        "file": ("comments.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    }
    data = {
        "model_id": "rubert-sentiment-conversational"
    }

    response = client.post("/api/v1/files/process", files=files, data=data)
    assert response.status_code == 202
    res_data = response.json()
    assert "task_id" in res_data
    assert res_data["status"] == "pending"
    assert res_data["total_rows"] == 3
    assert res_data["message"] == "Задача принята, вот её id"

    task_id = res_data["task_id"]

    # Wait for async processing to complete
    max_retries = 20
    completed = False
    for _ in range(max_retries):
        status_resp = client.get(f"/api/v1/files/tasks/{task_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data["status"] == "completed":
            assert status_data["progress_percentage"] == 100.0
            assert status_data["processed_rows"] == 3
            assert status_data["download_url"] is not None
            completed = True
            break
        time.sleep(0.05)

    assert completed, "Task did not complete in time"

    # Test downloading via /progress endpoint (returns file when completed)
    progress_resp = client.get(f"/api/v1/files/tasks/{task_id}/progress")
    assert progress_resp.status_code == 200
    assert "text/csv" in progress_resp.headers["content-type"]
    file_text = progress_resp.text
    assert "predicted_label" in file_text
    assert "confidence" in file_text
    assert "probabilities_json" in file_text

    # Test downloading via /download endpoint
    download_resp = client.get(f"/api/v1/files/tasks/{task_id}/download")
    assert download_resp.status_code == 200
    assert "predicted_label" in download_resp.text


def test_upload_nonexistent_model():
    files = {"file": ("data.csv", io.BytesIO(b"text\nHello"), "text/csv")}
    response = client.post("/api/v1/files/process", files=files, data={"model_id": "invalid-model"})
    assert response.status_code == 404


def test_upload_invalid_file_extension():
    files = {"file": ("data.txt", io.BytesIO(b"text\nHello"), "text/plain")}
    response = client.post("/api/v1/files/process", files=files, data={"model_id": "rubert-sentiment-conversational"})
    assert response.status_code == 400


def test_upload_empty_csv():
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/api/v1/files/process", files=files, data={"model_id": "rubert-sentiment-conversational"})
    assert response.status_code == 422


def test_get_nonexistent_task():
    response = client.get("/api/v1/files/tasks/nonexistent-id-123")
    assert response.status_code == 404
