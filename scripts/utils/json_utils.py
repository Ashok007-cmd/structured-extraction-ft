"""Shared JSON extraction utilities used by both the serving layer and evaluation scripts."""

import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from raw model output.

    Handles markdown fences (```json ... ```) and truncated trailing braces.
    """
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1)

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    elif start != -1:
        last_brace = text.rfind("}")
        text = text[start : last_brace + 1] if last_brace != -1 else text[start:]

    try:
        import json

        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
