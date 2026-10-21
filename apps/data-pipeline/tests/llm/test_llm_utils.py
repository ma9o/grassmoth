"""Tests for llm/traces.py pure utility functions.

Covers the OpenRouter client.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, cast

import pytest

from tests.support.helpers import run_async as _run

pytestmark = pytest.mark.contract


def _require_mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


class _FakeChatCompletions:
    def __init__(self, response: Mapping[str, object], seen: dict[str, object]):
        self._response = response
        self._seen = seen

    async def create(self, **kwargs):
        self._seen["kwargs"] = kwargs
        return self._response


class _FakeOpenRouterClient:
    def __init__(self, response: Mapping[str, object], seen: dict[str, object]):
        self.chat = type(
            "_FakeChatNamespace",
            (),
            {"completions": _FakeChatCompletions(response, seen)},
        )()


class TestOpenRouterClient:
    def test_sdk_response_preserves_reasoning_extensions_and_usage(self):
        from openai.types.chat import ChatCompletion

        from nof1_causal_service.llm import openrouter as openrouter_client

        response = ChatCompletion.model_validate(
            {
                "id": "completion-1",
                "created": 0,
                "model": "test-model",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning": "explanation",
                            "reasoning_details": [{"type": "reasoning.text", "text": "detail"}],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            }
        )
        parsed = openrouter_client._OpenRouterResponse.model_validate(response)
        message = openrouter_client._assistant_message(parsed.choices[0].message)

        assert message["content"] == "answer"
        assert message["reasoning"] == "explanation"
        assert message["reasoning_details"] == [{"type": "reasoning.text", "text": "detail"}]
        assert openrouter_client._usage_from_response(parsed) == {
            "input_tokens": 11,
            "output_tokens": 7,
            "reasoning_tokens": 3,
        }

    def test_call_model_enforces_local_timeout(self, monkeypatch):
        from nof1_causal_service.llm import openrouter as openrouter_client

        seen: dict[str, object] = {}

        class _SlowChatCompletions:
            async def create(self, **kwargs):
                seen["kwargs"] = kwargs
                await asyncio.sleep(0.05)
                return {"choices": []}

        slow_client = type(
            "_SlowOpenRouterClient",
            (),
            {
                "chat": type(
                    "_SlowChatNamespace",
                    (),
                    {"completions": _SlowChatCompletions()},
                )()
            },
        )()

        client = cast("Any", slow_client)

        timeout_seconds = cast("Any", 0.01)
        with pytest.raises(TimeoutError):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    client=client,
                    config=openrouter_client.GenerateConfig(timeout=timeout_seconds),
                )
            )

        assert _require_mapping(seen["kwargs"])["timeout"] == 0.01

    def test_call_model_logs_completion_tool_calls_and_reasoning(self, monkeypatch, caplog):
        from nof1_causal_service.llm import openrouter as openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "final answer",
                        "reasoning": "thought process",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "validate_extractions",
                                    "arguments": {"ok": True},
                                },
                            }
                        ],
                    },
                }
            ],
        }

        client = cast("Any", _FakeOpenRouterClient(response, seen))

        with caplog.at_level(logging.INFO):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    client=client,
                    config=openrouter_client.GenerateConfig(),
                    log_label="extraction chunk=1",
                )
            )

        assert _require_mapping(seen["kwargs"])["model"] == "test-model"
        assert "call_model completion:\nfinal answer" in caplog.text
        assert '"name": "validate_extractions"' in caplog.text
        assert "call_model reasoning:\nthought process" in caplog.text

    def test_call_model_logs_completion_without_label(self, monkeypatch, caplog):
        from nof1_causal_service.llm import openrouter as openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "unlabeled completion",
                    },
                }
            ],
        }

        client = cast("Any", _FakeOpenRouterClient(response, seen))

        with caplog.at_level(logging.INFO):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    client=client,
                    config=openrouter_client.GenerateConfig(),
                )
            )

        assert _require_mapping(seen["kwargs"])["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert "call_model completion:\nunlabeled completion" in caplog.text

    def test_call_model_strips_repo_openrouter_prefix(self, monkeypatch):
        from nof1_causal_service.llm import openrouter as openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "anthropic/claude-sonnet-4",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        client = cast("Any", _FakeOpenRouterClient(response, seen))

        _run(
            openrouter_client.call_model(
                "openrouter/anthropic/claude-sonnet-4",
                [{"role": "user", "content": "hello"}],
                client=client,
                config=openrouter_client.GenerateConfig(),
            )
        )

        assert _require_mapping(seen["kwargs"])["model"] == "anthropic/claude-sonnet-4"

    def test_call_model_uses_ambient_env_openrouter_key(self, monkeypatch):
        from nof1_causal_service.llm import openrouter as openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        def fake_async_openai(*, base_url: str, api_key: str):
            del base_url
            seen["api_key"] = api_key
            return _FakeOpenRouterClient(response, seen)

        monkeypatch.setattr(openrouter_client, "AsyncOpenAI", fake_async_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

        _run(
            openrouter_client.call_model(
                "test-model",
                [{"role": "user", "content": "hello"}],
                client=openrouter_client.create_openrouter_client(),
                config=openrouter_client.GenerateConfig(),
            )
        )

        assert seen["api_key"] == "env-key"
        assert _require_mapping(seen["kwargs"])["model"] == "test-model"

    def test_call_model_uses_reasoning_config_in_extra_body(self, monkeypatch):
        from nof1_causal_service.llm import openrouter as openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        client = cast("Any", _FakeOpenRouterClient(response, seen))

        _run(
            openrouter_client.call_model(
                "test-model",
                [{"role": "user", "content": "hello"}],
                client=client,
                config=openrouter_client.GenerateConfig(reasoning_effort="high"),
            )
        )

        assert _require_mapping(seen["kwargs"])["extra_body"] == {
            "provider": {"sort": "throughput"},
            "reasoning": {"effort": "high"},
        }
