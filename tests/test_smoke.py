"""
Lightweight smoke tests that run on CPU without loading real model weights.
Verifies pipeline logic: dataset formatting, JSON extraction, metrics computation,
and DPO format function — all without OOM risk.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from scripts.evaluate import compute_metrics
from scripts.utils.json_utils import extract_json

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_clean():
    raw = '{"event_type": "acquisition", "entities": []}'
    result = extract_json(raw)
    assert result == {"event_type": "acquisition", "entities": []}


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"event_type": "merger"}\n```'
    result = extract_json(raw)
    assert result == {"event_type": "merger"}


def test_extract_json_with_surrounding_text():
    raw = 'Here is the result:\n{"event_type": "ipo"}\nDone.'
    result = extract_json(raw)
    assert result is not None
    assert result["event_type"] == "ipo"


def test_extract_json_invalid_returns_none():
    assert extract_json("not json at all") is None
    assert extract_json("") is None


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

PREDICTED = {
    "event_type": "acquisition",
    "entities": [
        {"type": "organization", "name": "Acme Corp"},
        {"type": "person", "name": "Jane Doe"},
    ],
    "dates": [{"raw": "Jan 1 2024", "normalized": "2024-01-01"}],
    "financials": [{"type": "deal_value", "amount": "1000000"}],
}

GROUND_TRUTH = {
    "event_type": "acquisition",
    "entities": [
        {"type": "organization", "name": "Acme Corp"},
        {"type": "person", "name": "Jane Doe"},
        {"type": "person", "name": "John Smith"},
    ],
    "dates": [{"raw": "Jan 1 2024", "normalized": "2024-01-01"}],
    "financials": [{"type": "deal_value", "amount": "1000000"}],
}


def test_entity_recall_partial():
    m = compute_metrics(PREDICTED, GROUND_TRUTH)
    # 2 of 3 ground truth entities present → recall = 2/3
    assert abs(m["entity_recall"] - 2 / 3) < 1e-6


def test_entity_precision_full():
    m = compute_metrics(PREDICTED, GROUND_TRUTH)
    # all predicted entities are real → precision = 1.0
    assert m["entity_precision"] == 1.0


def test_date_normalization_exact_match():
    m = compute_metrics(PREDICTED, GROUND_TRUTH)
    assert m["date_normalization_recall"] == 1.0
    assert m["date_normalization_precision"] == 1.0


def test_no_hallucinations():
    m = compute_metrics(PREDICTED, GROUND_TRUTH)
    assert m["hallucination_count"] == 0


def test_hallucination_detected():
    pred_with_hallucination = dict(PREDICTED)
    pred_with_hallucination["entities"] = [
        {"type": "person", "name": "Ghost Entity"},
    ]
    m = compute_metrics(pred_with_hallucination, GROUND_TRUTH)
    assert m["hallucination_count"] == 1


def test_structural_fidelity_correct():
    m = compute_metrics(PREDICTED, GROUND_TRUTH)
    assert m["structural_fidelity"] == 1.0


def test_structural_fidelity_flat_strings():
    flat_pred = {"entities": ["Acme Corp", "Jane Doe"]}
    m = compute_metrics(flat_pred, GROUND_TRUTH)
    assert m["structural_fidelity"] == 0.0


# ---------------------------------------------------------------------------
# DPO dataset format
# ---------------------------------------------------------------------------

def test_dpo_dataset_format():
    """Verify DPO train.jsonl has expected prompt/chosen/rejected structure."""
    dpo_path = Path("data/dpo_dataset/train.jsonl")
    if not dpo_path.exists():
        return  # skip if dataset not generated yet
    with open(dpo_path) as f:
        first = json.loads(f.readline())
    assert "prompt" in first, "DPO example must have 'prompt'"
    assert "chosen" in first, "DPO example must have 'chosen'"
    assert "rejected" in first, "DPO example must have 'rejected'"
    # prompt is a list of messages
    assert isinstance(first["prompt"], list)
    # chosen/rejected are single-element lists with assistant messages
    assert isinstance(first["chosen"], list)
    assert first["chosen"][0]["role"] == "assistant"
    assert isinstance(first["rejected"], list)
    assert first["rejected"][0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# SFT dataset format
# ---------------------------------------------------------------------------

def test_sft_dataset_format():
    """Verify SFT train.jsonl has expected messages structure."""
    sft_path = Path("data/sft_dataset/train.jsonl")
    if not sft_path.exists():
        return
    with open(sft_path) as f:
        first = json.loads(f.readline())
    assert "messages" in first
    roles = [m["role"] for m in first["messages"]]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    # assistant content must be valid JSON
    for msg in first["messages"]:
        if msg["role"] == "assistant":
            parsed = extract_json(msg["content"])
            assert parsed is not None, "SFT assistant output must be valid JSON"
