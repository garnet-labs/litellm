from unittest.mock import call, patch

import pytest
import tiktoken
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

import litellm
from litellm import (
    create_pretrained_tokenizer,
    create_tokenizer,
    decode,
    encode,
    token_counter,
)
from litellm.utils import _select_tokenizer_helper, claude_json_str


@pytest.fixture(autouse=True)
def isolate_tokenizer_selection(monkeypatch):
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", False)
    monkeypatch.setattr(litellm, "disable_token_counter", False)
    monkeypatch.setattr(
        litellm, "anthropic_models", [*litellm.anthropic_models, "claude-sonnet-4-5"]
    )
    _select_tokenizer_helper.cache_clear()
    yield
    _select_tokenizer_helper.cache_clear()


@pytest.fixture
def hf_tokenizer():
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "Hello": 1, "world": 2, "café": 3, "世界": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


@pytest.mark.parametrize("text", ["", "Hello world", "café 世界"])
def test_openai_tokenizer_helpers(text):
    expected_ids = tiktoken.get_encoding("cl100k_base").encode(text)

    tokens = encode(model="gpt-3.5-turbo", text=text)

    assert isinstance(tokens, list)
    assert tokens == expected_ids
    assert decode(model="gpt-3.5-turbo", tokens=tokens) == text
    assert token_counter(model="gpt-3.5-turbo", text=text) == len(expected_ids)


@pytest.mark.parametrize("text", ["", "Hello world", "café 世界"])
def test_anthropic_tokenizer_helpers(text):
    expected_ids = Tokenizer.from_str(claude_json_str).encode(text).ids

    tokens = encode(model="claude-sonnet-4-5", text=text)

    assert isinstance(tokens, list)
    assert tokens == expected_ids
    assert decode(model="claude-sonnet-4-5", tokens=tokens) == text
    assert token_counter(model="claude-sonnet-4-5", text=text) == len(expected_ids)
    assert (
        _select_tokenizer_helper(model="claude-sonnet-4-5")["type"]
        == "huggingface_tokenizer"
    )


@pytest.mark.parametrize(
    "model,identifier",
    [
        ("meta-llama/Llama-2-7b-chat", "hf-internal-testing/llama-tokenizer"),
        ("meta-llama/llama-3-70b-instruct", "Xenova/llama-3-tokenizer"),
    ],
)
def test_llama_tokenizer_helpers(model, identifier, hf_tokenizer):
    with patch(
        "litellm.utils.Tokenizer.from_pretrained", return_value=hf_tokenizer
    ) as from_pretrained:
        tokens = encode(model=model, text="Hello world")

        assert isinstance(tokens, list)
        assert tokens == [1, 2]
        assert decode(model=model, tokens=tokens) == "Hello world"
        assert token_counter(model=model, text="Hello world") == 2
        assert _select_tokenizer_helper(model=model)["tokenizer"] is hf_tokenizer
        from_pretrained.assert_called_once_with(identifier)


@pytest.mark.parametrize(
    "text,expected_ids", [("", []), ("Hello world", [1, 2]), ("café 世界", [3, 4])]
)
def test_custom_json_tokenizer_helpers(hf_tokenizer, text, expected_ids):
    custom_tokenizer = create_tokenizer(hf_tokenizer.to_str())

    assert custom_tokenizer["type"] == "huggingface_tokenizer"
    assert isinstance(custom_tokenizer["tokenizer"], Tokenizer)
    with patch(
        "litellm.utils._select_tokenizer",
        side_effect=AssertionError("Model tokenizer must not be selected"),
    ):
        tokens = encode(
            model="gpt-3.5-turbo", text=text, custom_tokenizer=custom_tokenizer
        )

        assert isinstance(tokens, list)
        assert tokens == expected_ids
        assert (
            decode(
                model="gpt-3.5-turbo", tokens=tokens, custom_tokenizer=custom_tokenizer
            )
            == text
        )
        assert token_counter(
            model="gpt-3.5-turbo", text=text, custom_tokenizer=custom_tokenizer
        ) == len(expected_ids)


@pytest.mark.parametrize("revision", ["main", "test-revision"])
def test_create_pretrained_tokenizer(hf_tokenizer, revision):
    with patch(
        "litellm.utils.Tokenizer.from_pretrained", return_value=hf_tokenizer
    ) as from_pretrained:
        custom_tokenizer = create_pretrained_tokenizer(
            "test/custom-tokenizer", revision=revision
        )

    from_pretrained.assert_called_once_with(
        "test/custom-tokenizer", revision=revision, auth_token=None
    )
    assert custom_tokenizer == {
        "type": "huggingface_tokenizer",
        "tokenizer": hf_tokenizer,
    }
    tokens = encode(text="Hello world", custom_tokenizer=custom_tokenizer)
    assert tokens == [1, 2]
    assert decode(tokens=tokens, custom_tokenizer=custom_tokenizer) == "Hello world"
    assert token_counter(text="Hello world", custom_tokenizer=custom_tokenizer) == 2


def test_create_pretrained_tokenizer_retries_without_auth_token(hf_tokenizer):
    with patch(
        "litellm.utils.Tokenizer.from_pretrained",
        side_effect=[TypeError("unsupported auth_token argument"), hf_tokenizer],
    ) as from_pretrained:
        custom_tokenizer = create_pretrained_tokenizer("test/custom-tokenizer")

    assert from_pretrained.call_args_list == [
        call("test/custom-tokenizer", revision="main", auth_token=None),
        call("test/custom-tokenizer", revision="main"),
    ]
    assert custom_tokenizer["tokenizer"] is hf_tokenizer


def test_create_tokenizer_rejects_invalid_json():
    with pytest.raises(Exception, match="expected"):
        create_tokenizer("not valid JSON")


def test_token_counter_rejects_text_and_messages():
    with pytest.raises(ValueError, match="text and messages cannot both be set"):
        token_counter(
            model="gpt-3.5-turbo",
            text="Hello world",
            messages=[{"role": "user", "content": "Hello world"}],
        )
