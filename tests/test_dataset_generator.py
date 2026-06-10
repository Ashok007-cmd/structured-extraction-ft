import json
import random
import pytest
from data.generate_dataset import (
    TEMPLATES,
    PERSONS,
    ORGANIZATIONS,
    LOCATIONS,
    PRODUCTS,
    pick,
    sample_amount,
    sample_date_pair,
    generate_sft_example,
    corrupt_json,
    generate_dpo_example
)

def test_pick():
    # Test pick basic functionality
    items = [1, 2, 3]
    val = pick(items)
    assert val in items

    # Test avoiding used items
    used = {1, 2}
    val = pick(items, used=used)
    assert val == 3

    # Fallback to choosing from full list if all are used
    used_all = {1, 2, 3}
    val = pick(items, used=used_all)
    assert val in items

def test_sample_amount():
    fmt_str, val, currency = sample_amount()
    assert isinstance(fmt_str, str)
    assert isinstance(val, (int, float))
    assert isinstance(currency, str)
    assert len(currency) == 1
    assert currency in ["$", "€", "£", "¥"]

def test_sample_date_pair():
    raw, norm = sample_date_pair()
    assert isinstance(raw, str)
    assert isinstance(norm, str)
    # Check normalization format (YYYY-MM-DD)
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", norm)

def test_generate_sft_example():
    random.seed(42)
    used_names = set()
    for template in TEMPLATES:
        ex = generate_sft_example(template, used_names)
        assert ex is not None
        assert "text" in ex
        assert "structured_json" in ex
        assert "json_string" in ex
        assert "scenario" in ex
        assert "num_entities" in ex
        assert ex["scenario"] == template["scenario"]

        # Verify json_string is minified (no newlines/indentation)
        assert "\n" not in ex["json_string"]
        
        # Verify schema
        struct = ex["structured_json"]
        assert "event_type" in struct
        assert "entities" in struct
        assert isinstance(struct["entities"], list)
        for entity in struct["entities"]:
            assert "type" in entity
            assert "name" in entity

def test_corrupt_json():
    # Base example
    original = {
        "event_type": "acquisition",
        "acquirer": "NexGen Dynamics",
        "target": "Pinnacle Systems",
        "entities": [
            {"type": "organization", "name": "NexGen Dynamics"},
            {"type": "organization", "name": "Pinnacle Systems"},
            {"type": "person", "name": "Sarah Chen"}
        ],
        "financials": [{"type": "acquisition_value", "amount": 500000000, "currency": "$"}],
        "dates": [{"raw": "March 22, 2024", "normalized": "2024-03-22", "context": "announcement"}],
        "relationships": [
            {"type": "acquired", "subject": "NexGen Dynamics", "object": "Pinnacle Systems"}
        ],
        "metrics": []
    }

    # Test that corrupt_json successfully changes the original JSON
    random.seed(42)
    # Run multiple times to cover different corruption paths
    for _ in range(50):
        corrupted = corrupt_json(original)
        assert corrupted != original or corrupted.get("_note") == "This might have errors"
        # Validate structure is still valid python dict
        assert isinstance(corrupted, dict)

def test_generate_dpo_example():
    sft_example = {
        "text": "NexGen Dynamics acquired Pinnacle Systems.",
        "output_json": {
            "event_type": "acquisition",
            "entities": [{"type": "organization", "name": "NexGen Dynamics"}]
        }
    }
    
    random.seed(42)
    dpo_ex = generate_dpo_example(sft_example)
    assert dpo_ex is not None
    assert "prompt" in dpo_ex
    assert "chosen" in dpo_ex
    assert "rejected" in dpo_ex

    assert isinstance(dpo_ex["prompt"], list)
    assert len(dpo_ex["prompt"]) == 2
    assert dpo_ex["prompt"][0]["role"] == "system"
    assert dpo_ex["prompt"][1]["role"] == "user"

    assert isinstance(dpo_ex["chosen"], list)
    assert len(dpo_ex["chosen"]) == 1
    assert dpo_ex["chosen"][0]["role"] == "assistant"

    assert isinstance(dpo_ex["rejected"], list)
    assert len(dpo_ex["rejected"]) == 1
    assert dpo_ex["rejected"][0]["role"] == "assistant"
