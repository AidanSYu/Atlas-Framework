"""Main FastAPI application entry point (Embedded Desktop Sidecar)."""
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.config_routes import router as config_router
from app.api.data_routes import router as data_router
from app.api.task_routes import router as task_router
from app.atlas_plugin_system import get_tool_catalog
from app.api.framework_routes import router as framework_router
from app.core.config import settings
from app.core import registry
from app.core.database import get_project_engine

logger = logging.getLogger(__name__)

# Per-project DBs are created lazily on first project creation; the only
# app-level state we need at boot is the project registry file.
registry.init_registry()
logger.info("Project registry initialized at %s", registry.registry_path())


def _purge_legacy_actor_events() -> None:
    """One-time migration: the supervisor enum was renamed from DEEPSEEK/NEMOTRON
    to SUPERVISOR/ORCHESTRATOR, and the cloud-LLM two-tier stack was deleted.
    Historical events recorded the old strings and reference a now-gone code path,
    so wipe them. User explicitly opted into losing history during the rip-out.

    Tracked via an `atlas_migrations` table per project so this runs at most once
    per project DB.
    """
    LEGACY_PURGE_KEY = "purge_legacy_actor_events_2026_05"
    for entry in registry.list_projects():
        try:
            engine = get_project_engine(entry.id)
            with engine.begin() as conn:
                conn.execute(text(
                    "CREATE TABLE IF NOT EXISTS atlas_migrations "
                    "(key TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
                ))
                already = conn.execute(
                    text("SELECT 1 FROM atlas_migrations WHERE key = :k"),
                    {"k": LEGACY_PURGE_KEY},
                ).first()
                if already:
                    continue
                deleted = conn.execute(text(
                    "DELETE FROM task_events WHERE actor IN ('DEEPSEEK', 'NEMOTRON')"
                )).rowcount or 0
                # Reset sequence on any task whose ORCHESTRATOR/SUPERVISOR rows we just left
                # without companions — safer to just zero everything since we lost history.
                conn.execute(text("UPDATE tasks SET next_sequence = 0"))
                conn.execute(text("DELETE FROM task_events"))
                conn.execute(
                    text("INSERT INTO atlas_migrations(key, applied_at) VALUES (:k, datetime('now'))"),
                    {"k": LEGACY_PURGE_KEY},
                )
                logger.info(
                    "Project %s: purged %d legacy actor events and reset task sequences",
                    entry.id, deleted,
                )
        except Exception as exc:
            logger.warning("Migration failed for project %s: %s", entry.id, exc)


_purge_legacy_actor_events()


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

    # Unload orchestrator model to release GPU/VRAM and close worker threads
    try:
        from app.atlas_plugin_system.orchestrator import get_atlas_orchestrator
        orchestrator = get_atlas_orchestrator()
        orchestrator.unload()
        logger.info("Orchestrator model unloaded successfully")
    except Exception as exc:
        logger.warning("Failed to unload orchestrator model: %s", exc)

    # Unload Ingestion LLM model to release GPU/VRAM and close worker threads
    try:
        from app.services.llm import get_llm_service
        llm = get_llm_service()
        llm.unload()
        logger.info("Ingestion LLM model unloaded successfully")
    except Exception as exc:
        logger.warning("Failed to unload ingestion LLM model: %s", exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
