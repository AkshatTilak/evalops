"""ActionNodeMockRegistry for intercepting side-effects during evaluation test runs.
S5-10d: Prevents external webhook mutations or live database calls when eval_mode=True.
"""

import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("evalops.action_mock")


class ActionNodeMockRegistry:
    """Registry for mocking ActionNode external webhooks and side-effects during evaluation runs."""

    _instance: Optional["ActionNodeMockRegistry"] = None

    def __init__(self):
        self.mock_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.invoked_payloads: list = []

    @classmethod
    def get_instance(cls) -> "ActionNodeMockRegistry":
        if cls._instance is None:
            cls._instance = ActionNodeMockRegistry()
        return cls._instance

    def register_mock(self, action_type: str, mock_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Registers a custom mock handler for a specific action type (e.g. 'http_webhook', 'send_email')."""
        self.mock_handlers[action_type] = mock_func

    def intercept(self, action_type: str, payload: Dict[str, Any], eval_mode: bool = False) -> Optional[Dict[str, Any]]:
        """Intercepts execution if eval_mode is True.
        
        Returns mock result if intercepted, or None if live execution should proceed.
        """
        if not eval_mode:
            return None

        self.invoked_payloads.append({
            "action_type": action_type,
            "payload": payload
        })
        logger.info(f"[EVAL MOCK INTERCEPT] Intercepted ActionNode '{action_type}' with payload: {payload}")

        if action_type in self.mock_handlers:
            return self.mock_handlers[action_type](payload)

        # Default mock response for HTTP webhooks / side effects
        return {
            "status": 200,
            "mocked": True,
            "action_type": action_type,
            "response": {"message": "Simulated ActionNode execution success", "payload_received": payload}
        }

    def clear(self):
        """Clears registered mocks and invoked log history."""
        self.mock_handlers.clear()
        self.invoked_payloads.clear()
