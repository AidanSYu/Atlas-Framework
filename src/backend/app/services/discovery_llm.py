"""Discovery OS LLM Service — Isolated from global model selector.

This service provides API-based LLM access exclusively for Discovery OS agents
(Coordinator, Executor). It is completely independent from the main LLMService
used for chat/retrieval, ensuring that global model changes do not affect
active discovery sessions.

Architecture:
- DeepSeek V3 (deepseek-chat): Orchestration, planning, structured JSON output.
  NOTE: deepseek-reasoner (R1) does NOT reliably return content — it puts
  output in reasoning_content and leaves content empty. Use deepseek-chat.
- MiniMax (MiniMax-M2.5): Fast tool calling and constrained generation.
- LiteLLM: Unified API gateway for provider abstraction.
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


class DiscoveryLLMService:
    """Isolated LLM service for Discovery OS agents.

    This service is completely independent from the global LLMService.
    It uses API models exclusively and maintains its own configuration.
    """

    def __init__(self, config: Settings):
        self.config = config
        self._deepseek_client: Optional[AsyncOpenAI] = None
        self._minimax_client_ready = False

        # Discovery-specific configuration (does NOT read from global state)
        # Load from config — these are isolated from global model selector
        self._orchestration_provider = config.DISCOVERY_ORCHESTRATION_PROVIDER
        self._orchestration_model = config.DISCOVERY_ORCHESTRATION_MODEL
        self._tool_provider = config.DISCOVERY_TOOL_PROVIDER
        self._tool_model = config.DISCOVERY_TOOL_MODEL

        # Validate API keys at initialization
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate that required API keys are configured."""
        if self._orchestration_provider == "deepseek" and not self.config.DEEPSEEK_API_KEY:
            logger.warning(
                "DeepSeek API key not configured. Discovery OS orchestration will fail. "
                "Set DEEPSEEK_API_KEY in .env file."
            )
        if self._tool_provider == "minimax" and not self.config.MINIMAX_API_KEY:
            logger.warning(
                "MiniMax API key not configured. Discovery OS tool calling will fail. "
                "Set MINIMAX_API_KEY in .env file."
            )

    def _get_deepseek_client(self) -> AsyncOpenAI:
        """Lazy-load DeepSeek client."""
        if self._deepseek_client is None:
            if not self.config.DEEPSEEK_API_KEY:
                raise ValueError(
                    "DeepSeek API key not configured. Cannot initialize DiscoveryLLMService. "
                    "Set DEEPSEEK_API_KEY in .env file."
                )
            self._deepseek_client = AsyncOpenAI(
                api_key=self.config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com",
            )
            logger.info("DeepSeek client initialized for Discovery OS orchestration")
        return self._deepseek_client

    def _ensure_minimax_ready(self):
        """Ensure MiniMax is configured via LiteLLM."""
        if not self._minimax_client_ready:
            if not self.config.MINIMAX_API_KEY:
                raise ValueError(
                    "MiniMax API key not configured. Cannot use tool calling. "
                    "Set MINIMAX_API_KEY in .env file."
                )
            try:
                import litellm as _litellm
            except ImportError:
                raise RuntimeError(
                    "litellm is not installed. Run: pip install litellm"
                )
            # LiteLLM uses environment variables or can be set at runtime
            _litellm.api_key = self.config.MINIMAX_API_KEY
            self._minimax_client_ready = True
            logger.info("MiniMax configured via LiteLLM for Discovery OS tool calling")

    # ========================================================================
    # Orchestration API (High-level planning and reasoning)
    # ========================================================================

    async def orchestrate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Generate text using the orchestration model (DeepSeek).

        Use this for:
        - High-level planning and task decomposition
        - Reasoning about research strategy
        - Analyzing corpus context and generating questions

        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        if self._orchestration_provider == "deepseek":
            return await self._orchestrate_deepseek(
                prompt, system_prompt, temperature, max_tokens, **kwargs
            )
        else:
            raise ValueError(f"Unsupported orchestration provider: {self._orchestration_provider}")

    def _is_reasoner_model(self) -> bool:
        return "reasoner" in self._orchestration_model.lower() or "r1" in self._orchestration_model.lower()

    async def _orchestrate_deepseek(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> str:
        """Generate text using DeepSeek API.

        Handles both deepseek-chat (V3) and deepseek-reasoner (R1):
        - V3: supports system role, temperature, and returns content normally.
        - R1: does NOT support temperature, may return empty content (output
          is in reasoning_content). We fall back to reasoning_content when
          content is empty.
        """
        client = self._get_deepseek_client()
        is_reasoner = self._is_reasoner_model()

        messages = []
        if system_prompt:
            if is_reasoner:
                # R1 doesn't support system role — prepend to user message
                prompt = f"{system_prompt}\n\n---\n\n{prompt}"
            else:
                messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: Dict[str, Any] = {
            "model": self._orchestration_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not is_reasoner:
            call_kwargs["temperature"] = temperature
        call_kwargs.update(kwargs)

        try:
            response = await client.chat.completions.create(**call_kwargs)
            msg = response.choices[0].message
            content = msg.content or ""

            # R1 puts output in reasoning_content when content is empty
            if not content.strip() and hasattr(msg, "reasoning_content"):
                reasoning = getattr(msg, "reasoning_content", None) or ""
                if reasoning.strip():
                    logger.info(
                        "DeepSeek reasoner returned empty content; "
                        "extracting from reasoning_content (%d chars)",
                        len(reasoning),
                    )
                    content = reasoning

            if not content.strip():
                logger.warning(
                    "DeepSeek %s returned empty content AND empty reasoning_content",
                    self._orchestration_model,
                )

            return content
        except Exception as e:
            logger.error("DeepSeek orchestration failed (%s): %s", self._orchestration_model, e)
            raise RuntimeError(f"DeepSeek API error ({self._orchestration_model}): {e}") from e

    # ========================================================================
    # Orchestration-level constrained generation (DeepSeek — for coordinator)
    # ========================================================================

    async def orchestrate_constrained(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Generate structured JSON using the orchestration model (DeepSeek).

        Use this for coordinator-level reasoning that must produce structured output.
        DeepSeek is far more reliable at complex JSON than MiniMax because it is a
        reasoning model — MiniMax is optimised for fast tool calling, not analysis.

        Args:
            prompt: The user prompt
            schema: JSON Schema for output validation (used to build enforcement + defaults)
            system_prompt: Optional system instructions (JSON enforcement is prepended)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON object with all required schema fields populated.
        """
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        # Build a readable schema description to embed in the system prompt
        field_lines = []
        for k in required_fields:
            prop = properties.get(k, {})
            desc = prop.get("description", "required")
            field_lines.append(f'  "{k}": {desc}')
        schema_desc = "\n".join(field_lines)

        json_enforcement = (
            "RESPONSE FORMAT: Respond with ONLY a single valid JSON object. "
            "No markdown fences, no prose, no explanation outside the JSON. "
            "Your response must start with { and end with }.\n"
            f"Required fields:\n{schema_desc}"
        )
        effective_system = (
            f"{json_enforcement}\n\n{system_prompt}" if system_prompt else json_enforcement
        )
        parsed: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None
        last_raw = ""
        retries = 3

        for attempt in range(1, retries + 1):
            raw = await self.orchestrate(
                prompt=prompt,
                system_prompt=effective_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = raw or ""
            if not last_raw.strip():
                last_error = RuntimeError("DeepSeek returned an empty response body")
                logger.warning(
                    "DeepSeek returned empty response on attempt %d/%d",
                    attempt,
                    retries,
                )
                if attempt < retries:
                    await asyncio.sleep(0.4 * attempt)
                    continue
                break

            json_str = self._extract_json_from_content(last_raw)
            try:
                loaded = json.loads(json_str)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "DeepSeek JSON decode failed on attempt %d/%d: %s | raw: %s",
                    attempt,
                    retries,
                    e,
                    last_raw[:220],
                )
                if attempt < retries:
                    await asyncio.sleep(0.4 * attempt)
                    continue
                break

            if isinstance(loaded, list):
                if loaded and isinstance(loaded[0], dict):
                    logger.warning("DeepSeek returned a JSON array; unwrapping first element")
                    loaded = loaded[0]
                else:
                    last_error = RuntimeError(
                        f"DeepSeek returned a JSON array with no dict element: {json_str[:200]}"
                    )
                    if attempt < retries:
                        await asyncio.sleep(0.4 * attempt)
                        continue
                    break

            if not isinstance(loaded, dict):
                last_error = RuntimeError("DeepSeek response parsed to a non-object JSON type")
                if attempt < retries:
                    await asyncio.sleep(0.4 * attempt)
                    continue
                break

            parsed = loaded
            break

        if parsed is None:
            raw_preview = (last_raw or "").strip()[:260]
            raise RuntimeError(
                "DeepSeek JSON decode error after retries. "
                f"Last error: {last_error}. Raw preview: {raw_preview or '<empty>'}"
            ) from last_error

        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                logger.warning("DeepSeek returned a JSON array; unwrapping first element")
                parsed = parsed[0]
            else:
                raise RuntimeError(
                    f"DeepSeek returned a JSON array with no dict element: {json_str[:200]}"
                )

        # Fill missing required keys with safe type-appropriate defaults
        for key in required_fields:
            if key not in parsed or parsed[key] is None:
                prop = properties.get(key, {})
                t = prop.get("type", "string")
                if t == "array":
                    parsed[key] = []
                elif t == "boolean":
                    parsed[key] = False
                elif t == "object":
                    parsed[key] = {}
                else:
                    parsed[key] = ""

        return parsed

    # ========================================================================
    # Tool Calling API (Constrained generation for structured outputs — MiniMax)
    # ========================================================================

    async def generate_constrained(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured output using the tool model (MiniMax).

        Use this for:
        - Constrained JSON generation (questions, goals, scripts)
        - Tool call parameter extraction
        - Any structured output that must match a schema

        Args:
            prompt: The user prompt
            schema: JSON Schema for output validation
            system_prompt: Optional system instructions
            temperature: Sampling temperature (very low for strict schemas)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Parsed JSON object matching the schema

        Raises:
            RuntimeError: If MiniMax API key is not configured or the call fails.
                          Discovery OS requires both DeepSeek AND MiniMax to be fully
                          configured. Set MINIMAX_API_KEY in your .env file.
        """
        if self._tool_provider == "minimax":
            if not self.config.MINIMAX_API_KEY:
                raise RuntimeError(
                    "Discovery OS requires a working MiniMax API key for constrained generation. "
                    "Set MINIMAX_API_KEY in your .env file to proceed."
                )
            return await self._generate_minimax(
                prompt, schema, system_prompt, temperature, max_tokens, **kwargs
            )
        else:
            raise ValueError(f"Unsupported tool provider: {self._tool_provider}")

    def _extract_json_from_content(self, content: str) -> str:
        """Extract JSON from content that may contain markdown or extra text.

        Handles:
        - Markdown code blocks (```json ... ``` or ``` ... ```)
        - Mixed content (text before/after JSON)
        - Plain JSON

        Returns:
            Extracted JSON string ready for parsing
        """
        # Log raw content for debugging
        logger.debug(f"MiniMax raw content (first 500 chars): {content[:500]}")

        # Try to extract from markdown code block first
        if "```" in content:
            # Match ```json...``` or ```...```
            pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Extracted from markdown: {extracted[:200]}")
                return extracted

        # Try to find JSON object boundaries { ... }
        start = content.find("{")
        if start != -1:
            # Find matching closing brace
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        extracted = content[start:i+1]
                        logger.debug(f"Extracted JSON object: {extracted[:200]}")
                        return extracted

        # Try to find JSON array boundaries [ ... ]
        start = content.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "[":
                    depth += 1
                elif content[i] == "]":
                    depth -= 1
                    if depth == 0:
                        extracted = content[start:i+1]
                        logger.debug(f"Extracted JSON array: {extracted[:200]}")
                        return extracted

        # Fallback: return stripped content and hope for the best
        return content.strip()

    def _build_schema_fallback(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build a safe empty dict that satisfies the schema's required fields."""
        fallback: Dict[str, Any] = {}
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            prop = properties.get(key, {})
            t = prop.get("type", "string")
            if t == "array":
                fallback[key] = []
            elif t == "boolean":
                fallback[key] = False
            elif t == "object":
                fallback[key] = {}
            else:
                fallback[key] = ""
        return fallback

    async def _generate_minimax(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate constrained output using MiniMax via LiteLLM."""
        self._ensure_minimax_ready()

        # Prepend a hard JSON-object enforcement line to every system message.
        # MiniMax occasionally returns a JSON array when not explicitly told otherwise.
        json_enforcement = (
            "CRITICAL FORMAT RULE: Your ENTIRE response must be a single JSON object "
            "that starts with { and ends with }. Never return a JSON array []. "
            "Never add text outside the JSON object."
        )
        effective_system = (
            f"{json_enforcement}\n\n{system_prompt}" if system_prompt else json_enforcement
        )

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]

        try:
            import litellm  # noqa: PLC0415 — lazy import (optional dependency)

            response = await asyncio.to_thread(
                litellm.completion,
                model=f"minimax/{self._tool_model}",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                api_key=self.config.MINIMAX_API_KEY,
                **kwargs,
            )

            content = response.choices[0].message.content or "{}"

            # Extract JSON from potentially mixed content
            json_str = self._extract_json_from_content(content)

            # Parse JSON
            parsed = json.loads(json_str)

            # MiniMax sometimes returns an array despite json_object mode.
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    logger.warning("MiniMax returned a JSON array; unwrapping first element")
                    parsed = parsed[0]
                else:
                    # List of scalars — one retry with an explicit correction message.
                    logger.warning(
                        "MiniMax returned a JSON array of scalars (%s); retrying",
                        json_str[:100],
                    )
                    retry_messages = list(messages) + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "That response was a JSON array. I need a JSON object (curly braces {}). "
                                "Please respond with ONLY a valid JSON object."
                            ),
                        },
                    ]
                    retry_response = await asyncio.to_thread(
                        litellm.completion,
                        model=f"minimax/{self._tool_model}",
                        messages=retry_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                        api_key=self.config.MINIMAX_API_KEY,
                        **kwargs,
                    )
                    retry_content = retry_response.choices[0].message.content or "{}"
                    retry_json_str = self._extract_json_from_content(retry_content)
                    parsed = json.loads(retry_json_str)
                    if isinstance(parsed, list):
                        # Both attempts returned arrays — build a safe fallback dict
                        # so the caller can still proceed rather than crashing.
                        logger.error(
                            "MiniMax returned a JSON array after retry; using schema fallback. "
                            "Raw: %s", retry_json_str[:200]
                        )
                        parsed = self._build_schema_fallback(schema)

            # Fill any missing required keys with safe defaults
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            for key in required:
                if key not in parsed or parsed[key] is None:
                    prop = properties.get(key, {})
                    t = prop.get("type", "string")
                    if t == "array":
                        parsed[key] = []
                    elif t == "boolean":
                        parsed[key] = False
                    elif t == "object":
                        parsed[key] = {}
                    else:
                        parsed[key] = ""

            return parsed

        except json.JSONDecodeError as e:
            logger.error("MiniMax returned invalid JSON: %s", e)
            logger.error("Raw content: %s", locals().get("content", "N/A"))
            logger.warning(
                "Using schema fallback due to JSON decode error — "
                "downstream agents will receive empty/default values for: %s",
                list(schema.get("required", [])),
            )
            return self._build_schema_fallback(schema)
        except RuntimeError:
            raise  # Already formatted — let it propagate
        except Exception as e:
            logger.error("MiniMax constrained generation failed: %s", e)
            raise RuntimeError(f"MiniMax API error: {e}") from e

    # ========================================================================
    # Configuration Management
    # ========================================================================

    def get_orchestration_model(self) -> str:
        """Get the current orchestration model identifier."""
        return f"{self._orchestration_provider}/{self._orchestration_model}"

    def get_tool_model(self) -> str:
        """Get the current tool calling model identifier."""
        return f"{self._tool_provider}/{self._tool_model}"

    def set_orchestration_model(self, provider: str, model: str):
        """Change the orchestration model (e.g., switch to local DeepSeek).

        Args:
            provider: Provider name (e.g., "deepseek", "local")
            model: Model identifier (e.g., "deepseek-reasoner", "deepseek-r1-70b.gguf")
        """
        logger.info(f"Switching orchestration model: {provider}/{model}")
        self._orchestration_provider = provider
        self._orchestration_model = model
        # Clear cached client if switching providers
        if provider != "deepseek":
            self._deepseek_client = None
        self._validate_configuration()

    def set_tool_model(self, provider: str, model: str):
        """Change the tool calling model (e.g., switch to local MiniMax).

        Args:
            provider: Provider name (e.g., "minimax", "local")
            model: Model identifier (e.g., "MiniMax-M2.5", "minimax-text-01.gguf")
        """
        logger.info(f"Switching tool model: {provider}/{model}")
        self._tool_provider = provider
        self._tool_model = model
        # Clear cached client state
        if provider != "minimax":
            self._minimax_client_ready = False
        self._validate_configuration()


# ============================================================================
# Singleton Management
# ============================================================================

_discovery_llm_service: Optional[DiscoveryLLMService] = None


def get_discovery_llm_service() -> DiscoveryLLMService:
    """Get or create the singleton DiscoveryLLMService.

    This service is completely independent from the global LLMService.
    It maintains its own configuration and API clients.
    """
    global _discovery_llm_service
    if _discovery_llm_service is None:
        from app.core.config import settings
        _discovery_llm_service = DiscoveryLLMService(settings)
        logger.info(
            f"DiscoveryLLMService initialized — "
            f"Orchestration: {_discovery_llm_service.get_orchestration_model()}, "
            f"Tools: {_discovery_llm_service.get_tool_model()}"
        )
    return _discovery_llm_service
