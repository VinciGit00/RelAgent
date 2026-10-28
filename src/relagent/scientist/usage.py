"""Persist request-level usage and explicitly sourced costs before CAMEL conversion."""

import json
import logging
import math
import time
from datetime import datetime, timezone
from functools import lru_cache, wraps
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenBudgetExhausted(RuntimeError):
    pass


def _field(value, name, default=None):
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def response_cost(response, model_name=None):
    """Keep provider charges distinct from LiteLLM's calculated estimates."""
    reported = _field(response, "usage")
    for key in ("cost", "cost_usd"):
        value = _field(reported, key)
        if _number(value):
            return float(value), "provider_reported"
    hidden = _field(response, "_hidden_params", {}) or {}
    value = _field(hidden, "response_cost")
    if _number(value):
        return float(value), "litellm_estimate"
    if model_name:
        try:
            from litellm import completion_cost

            value = completion_cost(completion_response=response, model=str(model_name))
            if _number(value):
                return float(value), "litellm_estimate"
        except Exception:
            pass
    return None, "unavailable"


@lru_cache(maxsize=1)
def anthropic_cache_writes_excluded():
    """Probe the installed converter instead of assuming all LiteLLM versions agree."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    probe = AnthropicConfig().calculate_usage(
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 0,
        },
        None,
    )
    if probe.total_tokens not in (0, 7):
        raise RuntimeError("Unsupported Anthropic token accounting semantics")
    return probe.total_tokens == 0


def track_model_usage(model, directory: Path, token_budget=None):
    # CAMEL converts LiteLLM responses to OpenAI objects and drops _hidden_params.
    # Wrap its client when available so request metadata is captured beforehand.
    target = "client" if callable(getattr(model, "client", None)) else "run"
    original = getattr(model, target)
    directory.mkdir(parents=True, exist_ok=True)
    usage = {
        "requests": 0,
        "completed_requests": 0,
        "failed_requests": 0,
        "tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "token_budget": token_budget,
        "usage_available": True,
        "token_accounting": "includes_cache_creation",
        "api_cost_usd": 0.0,
        "cost_available": True,
        "cost_source": "no_requests",
    }
    sources = set()

    def persist(record=None):
        if record is not None:
            with (directory / "usage.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
        path = directory / "usage.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(usage, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    persist()

    @wraps(original)
    def run(*args, **kwargs):
        if token_budget is not None and not usage["usage_available"]:
            raise RuntimeError(
                "Provider omitted token usage; cannot enforce cumulative budget"
            )
        if token_budget is not None and usage["tokens"] >= token_budget:
            raise TokenBudgetExhausted("Cumulative token budget reached")
        usage["requests"] += 1
        started = time.monotonic()
        record = {
            "request_number": usage["requests"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        persist()
        logger.info("Model request %s started", usage["requests"])
        try:
            response = original(*args, **kwargs)
        except Exception as error:
            usage["failed_requests"] += 1
            # A failed call may have reached the provider; never assume it was free.
            usage["usage_available"] = False
            usage["cost_available"] = False
            usage["cost_source"] = "incomplete"
            record.update(
                status="error",
                error_type=type(error).__name__,
                duration_seconds=time.monotonic() - started,
            )
            persist(record)
            raise
        reported = _field(response, "usage")
        total = _field(reported, "total_tokens")
        prompt = _field(reported, "prompt_tokens")
        completion = _field(reported, "completion_tokens")
        if total is None and _number(prompt) and _number(completion):
            total = prompt + completion
        reported_total = total
        reported_prompt = prompt
        model_name = kwargs.get("model") or getattr(model, "model_type", None)
        cache_write = _field(reported, "cache_creation_input_tokens")
        if (
            str(model_name).startswith("anthropic/")
            and _number(cache_write)
            and cache_write
            and anthropic_cache_writes_excluded()
        ):
            if _number(total):
                total += cache_write
            if _number(prompt):
                prompt += cache_write
        known = _number(total)
        usage["usage_available"] = usage["usage_available"] and known
        if known:
            usage["tokens"] += int(total)
        details = _field(reported, "prompt_tokens_details")
        fields = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cache_read_tokens": _field(
                reported, "cache_read_input_tokens", _field(details, "cached_tokens")
            ),
            "cache_write_tokens": _field(
                reported,
                "cache_creation_input_tokens",
                _field(details, "cache_write_tokens"),
            ),
        }
        for key, value in fields.items():
            if _number(value):
                usage[key] += int(value)
        cost, source = response_cost(
            response, kwargs.get("model") or getattr(model, "model_type", None)
        )
        usage["cost_available"] = usage["cost_available"] and cost is not None
        if cost is not None:
            usage["api_cost_usd"] += cost
            sources.add(source)
        usage["cost_source"] = (
            (next(iter(sources)) if len(sources) == 1 else "mixed_estimate")
            if usage["cost_available"]
            else "incomplete"
        )
        usage["completed_requests"] += 1
        record.update(
            status="completed",
            response_id=_field(response, "id"),
            model=_field(response, "model"),
            total_tokens=total if known else None,
            reported_total_tokens=reported_total,
            reported_prompt_tokens=reported_prompt,
            **fields,
            api_cost_usd=cost,
            cost_source=source,
            cumulative_tokens=usage["tokens"],
            cumulative_api_cost_usd=usage["api_cost_usd"],
            duration_seconds=time.monotonic() - started,
        )
        persist(record)
        if token_budget is not None and not usage["usage_available"]:
            raise RuntimeError(
                "Provider omitted token usage; cannot enforce cumulative budget"
            )
        logger.info(
            "Model request %s finished; tokens=%s; cost_usd=%s (%s)",
            usage["requests"],
            usage["tokens"],
            usage["api_cost_usd"] if usage["cost_available"] else "unknown",
            usage["cost_source"],
        )
        return response

    setattr(model, target, run)
