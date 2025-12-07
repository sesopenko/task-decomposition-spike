from __future__ import annotations

from typing import Protocol


class RunUsageLike(Protocol):
    """Protocol to describe the minimal interface we need from RunUsage."""

    input_tokens: int
    output_tokens: int
    # cached_input_tokens is not currently present on RunUsage, but we
    # include it defensively and default to 0 if missing.
    # details: dict | None  # not needed for cost calculation


# Pricing constants for gpt-5.1 (USD per 1M tokens)
INPUT_PRICE_PER_MILLION = 1.25
OUTPUT_PRICE_PER_MILLION = 10.00
CACHED_INPUT_PRICE_PER_MILLION = 0.125


def calculate_cost(usage: RunUsageLike) -> str:
    """
    Calculate the approximate cost of a run based on token usage.

    Args:
        usage: An object with at least `input_tokens` and `output_tokens`
               attributes, and optionally `cached_input_tokens`.

    Returns:
        A string formatted as: "Cost: $0.XXXXXXX"
    """
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cached_input_tokens = getattr(usage, "cached_input_tokens", 0) or 0

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    cached_input_cost = (cached_input_tokens / 1_000_000) * CACHED_INPUT_PRICE_PER_MILLION

    total_cost = input_cost + output_cost + cached_input_cost

    # Format to 7 decimal places as requested
    return f"Cost: ${total_cost:.7f}"
