"""Config routes for the Atlas Framework.

The framework is offline-first; the cloud-LLM provider stack
(DeepSeek/MiniMax/OpenAI/Anthropic) was removed. These endpoints now only
expose whether legacy API keys are still present in `config/.env` so the
UI can prompt the user to clear them; they no longer make outbound API
calls to verify keys.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/config")
logger = logging.getLogger(__name__)


class ConfigKeysStatus(BaseModel):
    has_openai: bool
    has_anthropic: bool
    has_deepseek: bool
    has_minimax: bool
    # Surfaces to the UI that these are legacy values not used by the runtime.
    note: str = (
        "Cloud LLM providers were removed in the single-orchestrator transition. "
        "These keys, if set, are inert."
    )


@router.get("/keys", response_model=ConfigKeysStatus)
async def get_keys_status() -> ConfigKeysStatus:
    """Report which legacy API keys exist (for the UI's clear-them prompt)."""
    return ConfigKeysStatus(
        has_openai=bool(getattr(settings, "OPENAI_API_KEY", "")),
        has_anthropic=bool(getattr(settings, "ANTHROPIC_API_KEY", "")),
        has_deepseek=bool(getattr(settings, "DEEPSEEK_API_KEY", "")),
        has_minimax=bool(getattr(settings, "MINIMAX_API_KEY", "")),
    )
