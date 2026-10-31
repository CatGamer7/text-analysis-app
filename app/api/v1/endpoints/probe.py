import time
from fastapi import APIRouter, HTTPException, Path, status
from app.schemas.probe import (
    ProbeBatchRequest,
    ProbeBatchResponse,
    ProbeRequest,
    ProbeResponse,
)
from app.services.inference import inference_service
from app.services.model_registry import model_registry

router = APIRouter()


@router.post(
    "/models/{model_id}/probe",
    response_model=ProbeResponse,
    summary="Probe model on single text",
    description="Run single-sample evaluation/testing for an individual text against the specified model."
)
def probe_model(
    model_id: str = Path(..., description="ID of the model to test"),
    payload: ProbeRequest = ...
) -> ProbeResponse:
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in registry."
        )

    return inference_service.probe(
        model=model,
        text=payload.text,
        top_k=payload.top_k
    )


@router.post(
    "/probe",
    response_model=ProbeResponse,
    summary="Generic probe endpoint",
    description="Run model evaluation where model_id is specified directly in request body."
)
def probe_generic(
    payload: ProbeRequest
) -> ProbeResponse:
    if not payload.model_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'model_id' must be specified in request body when calling /probe directly."
        )

    model = model_registry.get_model(payload.model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{payload.model_id}' not found in registry."
        )

    return inference_service.probe(
        model=model,
        text=payload.text,
        top_k=payload.top_k
    )


@router.post(
    "/models/{model_id}/probe/batch",
    response_model=ProbeBatchResponse,
    summary="Batch probe model",
    description="Evaluate a small batch of sample texts against the specified model (up to 50 items)."
)
def probe_model_batch(
    model_id: str = Path(..., description="ID of the model to test"),
    payload: ProbeBatchRequest = ...
) -> ProbeBatchResponse:
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in registry."
        )

    start_time = time.perf_counter()
    results = [
        inference_service.probe(model=model, text=item.text, top_k=payload.top_k)
        for item in payload.items
    ]
    total_elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return ProbeBatchResponse(
        model_id=model.id,
        results=results,
        total_latency_ms=total_elapsed_ms
    )
