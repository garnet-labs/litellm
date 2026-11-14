import pytest

from litellm.litellm_core_utils.fallback_utils import completion_with_fallbacks


def test_completion_with_fallbacks_all_attempts_fail():
    with pytest.raises(Exception, match="All fallback attempts failed"):
        completion_with_fallbacks(
            model="gpt-4o-mini",
            api_key="sk-invalid",
            messages=[{"role": "user", "content": "hi"}],
            kwargs={"fallbacks": ["anthropic/claude-3-5-haiku-latest"]},
        )


def test_completion_with_fallbacks_returns_successful_response():
    response = completion_with_fallbacks(
        model="gpt-4o-mini",
        api_key="sk-invalid",
        messages=[{"role": "user", "content": "hi"}],
        kwargs={"fallbacks": [{"model": "gpt-4o-mini", "mock_response": "ok"}]},
    )

    assert response.choices[0].message.content == "ok"
