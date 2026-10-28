import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relagent.scientist.usage import TokenBudgetExhausted, track_model_usage


def test_raw_client_metadata_survives_conversion_and_is_persisted(tmp_path):
    raw = SimpleNamespace(
        id="request-1",
        model="model",
        usage={"total_tokens": 12, "prompt_tokens": 10, "completion_tokens": 2},
        _hidden_params={"response_cost": 0.02},
    )
    model = SimpleNamespace(client=Mock(return_value=raw))
    model.run = lambda: SimpleNamespace(usage=model.client().usage)
    track_model_usage(model, tmp_path, 10)
    model.run()
    with pytest.raises(TokenBudgetExhausted):
        model.run()
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["tokens"] == 12
    assert usage["api_cost_usd"] == 0.02
    assert usage["cost_source"] == "litellm_estimate"
    assert usage["cost_available"]
    record = json.loads((tmp_path / "usage.jsonl").read_text())
    assert record["response_id"] == "request-1"
    assert record["prompt_tokens"] == 10
    assert model.client.__wrapped__.call_count == 1


def test_provider_zero_cost_and_token_fallback_are_valid(tmp_path):
    model = SimpleNamespace(
        run=lambda: {"usage": {"prompt_tokens": 4, "completion_tokens": 2, "cost": 0}}
    )
    track_model_usage(model, tmp_path, 10)
    model.run()
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["tokens"] == 6
    assert usage["api_cost_usd"] == 0
    assert usage["cost_source"] == "provider_reported"


def test_missing_cost_never_becomes_a_known_zero(tmp_path):
    model = SimpleNamespace(run=lambda: {"usage": {"total_tokens": 4}})
    track_model_usage(model, tmp_path, 10)
    model.run()
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert not usage["cost_available"]
    assert usage["cost_source"] == "incomplete"


def test_failure_is_recorded_without_claiming_full_accounting(tmp_path):
    model = SimpleNamespace(run=Mock(side_effect=TimeoutError("timeout")))
    track_model_usage(model, tmp_path, 10)
    with pytest.raises(TimeoutError):
        model.run()
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["failed_requests"] == 1
    assert not usage["usage_available"]
    assert not usage["cost_available"]
    with pytest.raises(RuntimeError, match="omitted token usage"):
        model.run()


def test_request_records_accumulate_cache_tokens_and_cost_sources(tmp_path):
    responses = [
        {
            "usage": {
                "total_tokens": 10,
                "cost": 0.1,
                "prompt_tokens_details": {"cached_tokens": 3},
            }
        },
        {
            "usage": {"total_tokens": 20, "cache_creation_input_tokens": 5},
            "_hidden_params": {"response_cost": 0.2},
        },
    ]
    model = SimpleNamespace(run=Mock(side_effect=responses))
    track_model_usage(model, tmp_path, 100)
    model.run()
    model.run()
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["tokens"] == 30
    assert usage["cache_read_tokens"] == 3
    assert usage["cache_write_tokens"] == 5
    assert usage["api_cost_usd"] == pytest.approx(0.3)
    assert usage["cost_source"] == "mixed_estimate"
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2


def test_anthropic_cache_creation_tokens_count_toward_the_budget(tmp_path):
    response = {
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 118,
            "total_tokens": 121,
            "cache_creation_input_tokens": 5375,
        },
        "_hidden_params": {"response_cost": 0.021},
    }
    model = SimpleNamespace(
        client=Mock(return_value=response), model_type="anthropic/claude-sonnet-4-6"
    )
    track_model_usage(model, tmp_path, 5000)
    model.client(model=model.model_type)
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["tokens"] == 5496
    assert usage["prompt_tokens"] == 5378
    assert usage["token_accounting"] == "includes_cache_creation"
    with pytest.raises(TokenBudgetExhausted):
        model.client(model=model.model_type)
    record = json.loads((tmp_path / "usage.jsonl").read_text())
    assert record["reported_total_tokens"] == 121
    assert record["total_tokens"] == 5496
