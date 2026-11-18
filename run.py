import multiprocessing
import sys
import uvicorn
from app.core.config import settings
from app.main import app

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Ensure standard streams exist when running without a console on Windows
    if sys.stdout is None:
        import os
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        import os
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    uvicorn.run(
        "app.main:app" if settings.debug else app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
