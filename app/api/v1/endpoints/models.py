from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.model import ModelListResponse, ModelMetadata, ModelTaskType
from app.services.model_registry import model_registry

router = APIRouter()


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Retrieve all supported NLP models for social media analysis with filtering by task type and language."
)
def list_models(
    task_type: Optional[ModelTaskType] = Query(None, description="Filter by NLP task type"),
    language: Optional[str] = Query(None, description="Filter by supported language code (e.g. 'ru', 'en')")
) -> ModelListResponse:
    models = model_registry.list_models()
    
    if task_type:
        models = [m for m in models if m.task_type == task_type]
    if language:
        models = [m for m in models if language.lower() in [l.lower() for l in m.languages]]
        
    return ModelListResponse(items=models, total=len(models))


@router.get(
    "/models/{model_id}",
    response_model=ModelMetadata,
    summary="Get model details",
    description="Retrieve detailed specifications, metadata, and default sample text for a specific model."
)
def get_model(model_id: str) -> ModelMetadata:
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in registry."
        )
    return model
