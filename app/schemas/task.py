from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskCreatedResponse(BaseModel):
    task_id: str = Field(..., description="Уникальный идентификатор задачи обработки", examples=["b7a48d8c-0f9e-4e89-8d75-5c1a7b489d12"])
    status: TaskStatus = Field(TaskStatus.PENDING, description="Начальный статус задачи", examples=[TaskStatus.PENDING])
    message: str = Field("Задача принята в обработку", description="Информационное сообщение", examples=["Задача принята, вот её id"])
    model_id: str = Field(..., description="ID примененной модели", examples=["rubert-sentiment-conversational"])
    filename: str = Field(..., description="Имя загруженного исходного файла", examples=["comments.csv"])
    total_rows: int = Field(..., description="Количество текстовых строк для обработки", examples=[120])
    created_at: str = Field(..., description="Время создания задачи (ISO 8601)")


class TaskProgressResponse(BaseModel):
    task_id: str = Field(..., description="Уникальный идентификатор задачи")
    status: TaskStatus = Field(..., description="Текущий статус обработки")
    total_rows: int = Field(..., description="Всего строк в файле", examples=[100])
    processed_rows: int = Field(..., description="Количество обработанных строк", examples=[45])
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Прогресс обработки в процентах (0.0 - 100.0)", examples=[45.0])
    download_url: Optional[str] = Field(None, description="Ссылка для скачивания результирующего CSV (доступна при статусе completed)")
    error: Optional[str] = Field(None, description="Сообщение об ошибке, если статус FAILED")
    created_at: str = Field(..., description="Время постановки задачи")
    completed_at: Optional[str] = Field(None, description="Время завершения обработки")
