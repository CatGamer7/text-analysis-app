import asyncio
from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile, status
from fastapi.responses import FileResponse
from app.schemas.task import TaskCreatedResponse, TaskProgressResponse, TaskStatus
from app.services.model_registry import model_registry
from app.services.task_service import file_task_service

router = APIRouter()


@router.post(
    "/files/process",
    response_model=TaskCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Загрузка и запуск асинхронной обработки CSV файла",
    description=(
        "Принимает CSV файл строгого формата (1 колонка с текстами) и идентификатор модели. "
        "Возвращает 'задача принята, вот её id' и запускает асинхронную обработку в фоновом режиме."
    )
)
async def upload_and_process_file(
    file: UploadFile = File(..., description="CSV файл с 1 столбцом текстов"),
    model_id: str = Form(..., description="ID модели для классификации", examples=["rubert-sentiment-conversational"])
) -> TaskCreatedResponse:
    # 1. Проверка существования модели
    if not model_registry.model_exists(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID '{model_id}' не найдена в каталоге."
        )

    # 2. Проверка расширения файла
    filename = file.filename or "data.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поддерживаются только файлы формата .csv."
        )

    # 3. Чтение и парсинг CSV
    try:
        content = await file.read()
        header, texts = file_task_service.parse_csv_content(content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    # 4. Регистрация задачи
    task_response = file_task_service.create_task(
        filename=filename,
        model_id=model_id,
        header=header,
        texts=texts
    )

    # 5. Запуск асинхронной обработки
    asyncio.create_task(file_task_service.execute_task_async(task_response.task_id))

    return task_response


@router.get(
    "/files/tasks/{task_id}/progress",
    summary="Получение прогресса / выгрузка готового файла",
    description=(
        "Возвращает прогресс задачи (в % обработанных строк / всего строк). "
        "Если обработка завершена — возвращает готовый CSV файл с результатами."
    )
)
def get_task_progress_or_file(
    task_id: str = Path(..., description="ID задачи обработки")
):
    task = file_task_service.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача с ID '{task_id}' не найдена."
        )

    # Если задача завершена — возвращаем результирующий файл
    if task.status == TaskStatus.COMPLETED:
        file_path = file_task_service.get_output_file_path(task_id)
        if file_path and file_path.exists():
            return FileResponse(
                path=str(file_path),
                media_type="text/csv",
                filename=f"result_{task_id}.csv"
            )

    # Если задача в процессе, ожидает или завершилась с ошибкой — возвращаем JSON с прогрессом
    return task


@router.get(
    "/files/tasks/{task_id}",
    response_model=TaskProgressResponse,
    summary="Статус задачи и прогресс в формате JSON",
    description="Всегда возвращает JSON со статусом, % выполнения и количеством строк."
)
def get_task_status_json(
    task_id: str = Path(..., description="ID задачи обработки")
) -> TaskProgressResponse:
    task = file_task_service.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача с ID '{task_id}' не найдена."
        )
    return task


@router.get(
    "/files/tasks/{task_id}/download",
    summary="Скачать обработанный CSV файл",
    description="Скачивает обработанный CSV файл после завершения задачи."
)
def download_task_result(
    task_id: str = Path(..., description="ID задачи обработки")
):
    task = file_task_service.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача с ID '{task_id}' не найдена."
        )

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Задача ещё не завершена (текущий статус: {task.status.value})."
        )

    file_path = file_task_service.get_output_file_path(task_id)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Результирующий файл не найден на диске."
        )

    return FileResponse(
        path=str(file_path),
        media_type="text/csv",
        filename=f"result_{task_id}.csv"
    )
