import asyncio
import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.schemas.task import TaskCreatedResponse, TaskProgressResponse, TaskStatus
from app.services.inference import inference_service
from app.services.model_registry import model_registry


class FileTaskService:
    """Service managing asynchronous CSV file processing with ML models."""

    def __init__(self, storage_dir: str = "data/tasks"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, dict] = {}

    def parse_csv_content(self, file_bytes: bytes) -> tuple[Optional[str], List[str]]:
        """
        Parses raw bytes from uploaded CSV file (supports UTF-8, UTF-8-BOM, CP1251).
        Expects 1 column format. Returns tuple of (optional header, list of text rows).
        """
        text_content = None
        for encoding in ["utf-8-sig", "utf-8", "windows-1251", "cp1251", "latin-1"]:
            try:
                text_content = file_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text_content is None:
            raise ValueError("Не удалось распознать кодировку файла (поддерживаются UTF-8, CP1251).")

        reader = csv.reader(io.StringIO(text_content))
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]

        if not rows:
            raise ValueError("CSV файл пуст.")

        # Extract text from the first column
        all_items = [r[0].strip() for r in rows if r[0].strip()]
        if not all_items:
            raise ValueError("Файл не содержит текстовых строк в первой колонке.")

        first_item = all_items[0].lower()
        has_header = first_item in ["text", "текст", "content", "комментарий", "comment", "message", "сообщение"]
        
        if has_header and len(all_items) > 1:
            header = all_items[0]
            texts = all_items[1:]
        else:
            header = "text"
            texts = all_items

        return header, texts

    def create_task(self, filename: str, model_id: str, header: str, texts: List[str]) -> TaskCreatedResponse:
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        total_rows = len(texts)

        task_record = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "filename": filename,
            "model_id": model_id,
            "header": header,
            "texts": texts,
            "total_rows": total_rows,
            "processed_rows": 0,
            "progress_percentage": 0.0,
            "output_file": None,
            "error": None,
            "created_at": now,
            "completed_at": None,
        }
        self._tasks[task_id] = task_record

        return TaskCreatedResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Задача принята, вот её id",
            model_id=model_id,
            filename=filename,
            total_rows=total_rows,
            created_at=now,
        )

    async def execute_task_async(self, task_id: str) -> None:
        """Asynchronous worker that processes texts row-by-row and updates progress."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task["status"] = TaskStatus.PROCESSING
        model_id = task["model_id"]
        model = model_registry.get_model(model_id)
        if not model:
            task["status"] = TaskStatus.FAILED
            task["error"] = f"Модель '{model_id}' не найдена в реестре."
            return

        output_path = self.storage_dir / f"result_{task_id}.csv"
        header = task["header"]
        texts = task["texts"]
        total = task["total_rows"]

        try:
            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                # Write enhanced CSV header
                writer.writerow([
                    header,
                    "predicted_label",
                    "confidence",
                    "confidence_percentage",
                    "probabilities_json"
                ])

                for idx, text in enumerate(texts):
                    # Perform inference
                    probe = inference_service.probe(model=model, text=text)
                    scores_dict = {s.label: s.percentage for s in probe.scores}
                    
                    writer.writerow([
                        text,
                        probe.primary_label,
                        round(probe.confidence, 4),
                        f"{round(probe.confidence * 100, 2)}%",
                        json.dumps(scores_dict, ensure_ascii=False)
                    ])

                    processed = idx + 1
                    task["processed_rows"] = processed
                    task["progress_percentage"] = round((processed / total) * 100, 2)

                    # Yield control to event loop so other requests are handled smoothly
                    if idx % 5 == 0:
                        await asyncio.sleep(0.001)

            task["status"] = TaskStatus.COMPLETED
            task["output_file"] = str(output_path)
            task["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)

    def get_task(self, task_id: str) -> Optional[TaskProgressResponse]:
        task = self._tasks.get(task_id)
        if not task:
            return None

        download_url = None
        if task["status"] == TaskStatus.COMPLETED:
            download_url = f"/api/v1/files/tasks/{task_id}/download"

        return TaskProgressResponse(
            task_id=task["task_id"],
            status=task["status"],
            total_rows=task["total_rows"],
            processed_rows=task["processed_rows"],
            progress_percentage=task["progress_percentage"],
            download_url=download_url,
            error=task["error"],
            created_at=task["created_at"],
            completed_at=task["completed_at"]
        )

    def get_output_file_path(self, task_id: str) -> Optional[Path]:
        task = self._tasks.get(task_id)
        if not task or not task.get("output_file"):
            return None
        path = Path(task["output_file"])
        return path if path.exists() else None


file_task_service = FileTaskService()
