from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Text Analysis Backend API"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # CORS configuration for desktop frontend (Electron / web)
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]

    # Models inference mode
    use_mock_inference: bool = True
    models_cache_dir: str = "./models_cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore"
    )


settings = Settings()
