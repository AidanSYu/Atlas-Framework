"""
Atlas Framework backend server entry point.

This launches the FastAPI sidecar used by both local development and the
bundled desktop application.
"""
import os
# Force disable hugging face symlinks to prevent WinError 1314 before any imports happen
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import logging
from pathlib import Path

# Configure logging before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    
    return base_path / relative_path


def setup_environment():
    """Set up environment variables for bundled app.

    Per-project state (SQLite, Qdrant) lives inside each project's folder
    under ATLAS_WORKSPACES_DIR and is not configured here. Only the shared
    models dir + workspaces root are set up at app startup.
    """
    if hasattr(sys, '_MEIPASS'):
        if sys.platform == 'win32':
            app_data = Path(os.environ.get('LOCALAPPDATA', '')) / 'Atlas'
        elif sys.platform == 'darwin':
            app_data = Path.home() / 'Library' / 'Application Support' / 'Atlas'
        else:
            app_data = Path.home() / '.atlas'

        app_data.mkdir(parents=True, exist_ok=True)
        models_dir = app_data / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('MODELS_DIR', str(models_dir))
        logger.info(f"Using models from: {models_dir}")

        workspaces_dir = app_data / 'workspaces'
        workspaces_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('ATLAS_WORKSPACES_DIR', str(workspaces_dir))
        logger.info(f"Using workspaces dir: {workspaces_dir}")
    else:
        # Dev: models live in the repo-root models/ directory.
        dev_models = Path(__file__).resolve().parent.parent.parent / "models"
        os.environ.setdefault("MODELS_DIR", str(dev_models))
        logger.info(f"Using dev models from: {dev_models}")


def main():
    """Start the Atlas backend server."""
    import uvicorn

    setup_environment()

    from app.core.config import settings
    from app.core import registry
    registry.init_registry()
    logger.info(f"Project registry: {registry.registry_path()}")

    host = settings.API_HOST
    port = settings.API_PORT

    logger.info("Starting Atlas Backend Server...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Binding to http://{host}:{port}")
    logger.info(f"Framework health endpoint: http://{host}:{port}/api/framework/health")
    logger.info(f"Framework tools endpoint: http://{host}:{port}/api/framework/tools")

    # Import the app after environment setup
    from app.main import app

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
    except OSError as e:
        winerr = getattr(e, "winerror", None)
        msg = str(e).lower()
        if winerr == 10048 or "10048" in msg or "address already in use" in msg or "only one usage" in msg:
            logger.error(
                "Port %s is already in use. Either stop the other process using it "
                "(e.g. another backend instance or Tauri app), or set API_PORT to a different port "
                "(e.g. in src/backend/.env: API_PORT=8001).",
                port,
            )
        raise


if __name__ == "__main__":
    main()
