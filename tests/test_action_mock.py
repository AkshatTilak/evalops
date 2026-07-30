"""Unit test for S5-10d: Action Node Mocking Framework."""

from projects.evalops.src.runner.action_mock import ActionNodeMockRegistry


def test_action_node_mock_registry():
    """Verify ActionNodeMockRegistry intercepts action nodes in eval mode."""
    registry = ActionNodeMockRegistry()
    registry.clear()

    payload = {"url": "https://api.thirdparty.com/webhook", "method": "POST", "body": {"user_id": "123"}}

    # 1. Live mode (eval_mode=False) -> returns None (proceed with real call)
    live_result = registry.intercept("http_webhook", payload, eval_mode=False)
    assert live_result is None
    assert len(registry.invoked_payloads) == 0

    # 2. Eval mode default mock (eval_mode=True)
    eval_result = registry.intercept("http_webhook", payload, eval_mode=True)
    assert eval_result is not None
    assert eval_result["status"] == 200
    assert eval_result["mocked"] is True
    assert len(registry.invoked_payloads) == 1

    # 3. Custom registered mock
    registry.register_mock("custom_action", lambda p: {"status": 201, "custom_key": p.get("data")})
    custom_result = registry.intercept("custom_action", {"data": "hello"}, eval_mode=True)
    assert custom_result["status"] == 201
    assert custom_result["custom_key"] == "hello"
    assert len(registry.invoked_payloads) == 2
