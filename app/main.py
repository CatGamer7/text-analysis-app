from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.api.v1.router import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Бэкенд-сервис для настольного приложения анализа текстовых данных социальных сетей "
            "с помощью открытых языковых моделей (русский и английский языки).\n\n"
            "Включает точки управления каталогом моделей, апробации (проверки работы) моделей "
            "на единичных примерах и тестовых выборках."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS for Electron / web UI
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    application.include_router(api_router, prefix=settings.api_v1_prefix)

    @application.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    return application


app = create_application()
