"""Main FastAPI application entry point (Embedded Desktop Sidecar)."""
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.config_routes import router as config_router
from app.api.data_routes import router as data_router
from app.api.task_routes import router as task_router
from app.atlas_plugin_system import get_tool_catalog
from app.api.framework_routes import router as framework_router
from app.core.config import settings
from app.core.database import init_db

logger = logging.getLogger(__name__)

# Initialize database on startup - FATAL on failure
init_db()
logger.info("SQLite database initialized successfully")


app = FastAPI(
    title="Atlas Framework API",
    description=(
        "Embedded desktop sidecar: offline-first research operating system powered "
        "by a single local orchestrator, hybrid RAG substrate, and optional plugins."
    ),
    version="2.0.0-sidecar",
)
logger.info("FastAPI app created")


# CORS Configuration - Locked to Tauri internal origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "https://tauri.localhost",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return proper error response."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "type": type(exc).__name__,
        },
    )


app.include_router(config_router)
app.include_router(framework_router)
app.include_router(data_router)
app.include_router(task_router)
logger.info("Routes included")


@app.on_event("startup")
async def startup_event():
    """Application startup for the Atlas Framework backend."""
    logger.info("Atlas Framework sidecar starting up")
    app.state.startup_complete = False

    async def _background_startup():
        """Warm the tool catalog without blocking the event loop."""
        try:
            catalog = get_tool_catalog()
            catalog.refresh()
            logger.info(
                "Atlas Framework tool catalog ready with %d core tool(s) and %d plugin(s)",
                len(catalog.list_core_tools()),
                len(catalog.list_plugins()),
            )
            app.state.startup_complete = True
            logger.info("Atlas Framework backend ready")
        except Exception as exc:
            logger.error("Background startup failed: %s", exc, exc_info=True)
            logger.warning("Framework services will be initialized lazily on first request")
            app.state.startup_complete = True

    asyncio.create_task(_background_startup())


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Atlas Sidecar shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
