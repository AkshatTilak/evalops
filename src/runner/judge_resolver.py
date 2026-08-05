"""EvalOps Judge Model Resolver with GuardRoute Registry Sync and Local Fallback (sub_09_01)."""

import logging
from common.config.settings import get_settings
from common.models.registry import get_active_model

logger = logging.getLogger("evalops.runner.judge_resolver")


async def resolve_judge_model(role: str = "completion") -> str:
    """Resolve evaluation judge model string, syncing with GuardRoute registry and falling back to local model if remote API keys are missing."""
    settings = get_settings()
    try:
        model_spec = await get_active_model(role)
        provider = model_spec.provider.lower()
        if model_spec.mode == "local":
            return model_spec.model_id

        # Verify provider API key presence
        has_key = False
        if provider in ("gemini", "google") and getattr(settings, "GOOGLE_API_KEY", None):
            has_key = True
        elif provider == "openrouter" and getattr(settings, "OPENROUTER_API_KEY", None):
            has_key = True
        elif provider == "groq" and getattr(settings, "GROQ_API_KEY", None):
            has_key = True
        elif provider == "openai" and getattr(settings, "OPENAI_API_KEY", None):
            has_key = True

        if has_key:
            return model_spec.model_id

        logger.warning(
            "Judge model '%s' provider '%s' key missing. Falling back to local judge model 'harrier-0.6b'.",
            model_spec.model_id, provider
        )
        return "harrier-0.6b"
    except Exception as e:
        logger.warning("Error resolving judge model from registry: %s. Using default local judge.", e)
        return "harrier-0.6b"
