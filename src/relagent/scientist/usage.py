"""Persist provider-reported usage after every completed model request."""

import json
import logging
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenBudgetExhausted(RuntimeError):
    pass


def track_model_usage(model, directory: Path, token_budget=None):
    original = model.run
    usage = {
        "requests": 0,
        "tokens": 0,
        "token_budget": token_budget,
        "usage_available": True,
    }

    @wraps(original)
    def run(*args, **kwargs):
        if token_budget is not None and usage["tokens"] >= token_budget:
            raise TokenBudgetExhausted("Cumulative token budget reached")
        logger.info("Model request %s started", usage["requests"] + 1)
        response = original(*args, **kwargs)
        reported = getattr(response, "usage", None)
        total = getattr(reported, "total_tokens", None)
        if total is not None:
            usage["tokens"] += int(total)
        usage["requests"] += 1
        usage["usage_available"] = usage["usage_available"] and total is not None
        path = directory / "usage.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(usage), encoding="utf-8")
        temporary.replace(path)
        if token_budget is not None and not usage["usage_available"]:
            raise RuntimeError(
                "Provider omitted token usage; cannot enforce cumulative budget"
            )
        logger.info(
            "Model request %s finished; cumulative reported tokens=%s",
            usage["requests"],
            usage["tokens"],
        )
        return response

    model.run = run
