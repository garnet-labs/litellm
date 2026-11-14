from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.fallback_utils import completion_with_fallbacks


def test_completion_with_fallbacks_all_attempts_fail(monkeypatch):
    completion = AsyncMock(side_effect=Exception("Provider unavailable"))
    monkeypatch.setattr(litellm, "acompletion", completion)

    with pytest.raises(Exception, match="All fallback attempts failed"):
        completion_with_fallbacks(
            model="gpt-4o-mini",
            api_key="sk-invalid",
            messages=[{"role": "user", "content": "hi"}],
            kwargs={"fallbacks": ["anthropic/claude-3-5-haiku-latest"]},
        )

    assert [call.kwargs["model"] for call in completion.call_args_list] == [
        "gpt-4o-mini",
        "anthropic/claude-3-5-haiku-latest",
    ]
    assert completion.await_count == 2


def test_completion_with_fallbacks_returns_successful_response(monkeypatch):
    expected_response = MagicMock()
    expected_response.choices[0].message.content = "ok"
    completion = AsyncMock(
        side_effect=[Exception("Provider unavailable"), expected_response]
    )
    monkeypatch.setattr(litellm, "acompletion", completion)

    response = completion_with_fallbacks(
        model="gpt-4o-mini",
        api_key="sk-invalid",
        messages=[{"role": "user", "content": "hi"}],
        kwargs={"fallbacks": ["anthropic/claude-3-5-haiku-latest"]},
    )

    assert response.choices[0].message.content == "ok"
    assert [call.kwargs["model"] for call in completion.call_args_list] == [
        "gpt-4o-mini",
        "anthropic/claude-3-5-haiku-latest",
    ]
    assert completion.await_count == 2
