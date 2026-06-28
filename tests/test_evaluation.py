from scripts.evaluate import compute_metrics
from scripts.utils.json_utils import extract_json


def test_extract_json():
    # 1. Perfect JSON
    text1 = '{"event_type": "funding", "company": "Apex"}'
    assert extract_json(text1) == {"event_type": "funding", "company": "Apex"}

    # 2. Markdown fenced JSON
    text2 = '```json\n{"event_type": "funding", "company": "Apex"}\n```'
    assert extract_json(text2) == {"event_type": "funding", "company": "Apex"}

    # 3. Surrounding text
    text3 = 'Sure, here is the data:\n{"event_type": "funding", "company": "Apex"}\nhope this helps!'
    assert extract_json(text3) == {"event_type": "funding", "company": "Apex"}

    # 4. Malformed JSON
    text4 = '{"event_type": "funding", "company": "Apex"'
    assert extract_json(text4) is None

def test_compute_metrics_exact_match():
    gt = {
        "event_type": "acquisition",
        "entities": [
            {"type": "organization", "name": "NexGen Dynamics"},
            {"type": "person", "name": "Sarah Chen"}
        ],
        "financials": [{"type": "acquisition_value", "amount": 500000000, "currency": "$"}],
        "dates": [{"raw": "March 22, 2024", "normalized": "2024-03-22", "context": "announcement"}],
        "relationships": [
            {"type": "employment", "subject": "Sarah Chen", "object": "NexGen Dynamics", "role": "CEO"}
        ]
    }

    # Evaluate identical dict
    metrics = compute_metrics(gt, gt)
    assert metrics["schema_compliance"] == 1.0
    assert metrics["section_recall"] == 1.0
    assert metrics["section_precision"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_f1"] == 1.0
    assert metrics["date_normalization_recall"] == 1.0
    assert metrics["date_normalization_precision"] == 1.0
    assert metrics["financial_recall"] == 1.0
    assert metrics["financial_precision"] == 1.0
    assert metrics["structural_fidelity"] == 1.0
    assert metrics["hallucination_count"] == 0

def test_compute_metrics_mismatch():
    gt = {
        "event_type": "funding",
        "entities": [
            {"type": "organization", "name": "NexGen"},
            {"type": "person", "name": "Sarah Chen"}
        ],
        "dates": [{"raw": "March 22, 2024", "normalized": "2024-03-22"}]
    }

    # Prediction drops one entity, adds a hallucination, and wrong date norm
    pred = {
        "event_type": "funding",
        "entities": [
            {"type": "organization", "name": "NexGen"},
            {"type": "person", "name": "Fake Person"}
        ],
        "dates": [{"raw": "March 22, 2024", "normalized": "March 22, 2024"}]  # Not normalized
    }

    metrics = compute_metrics(pred, gt)

    # Check that schema complies (still valid types and keys)
    assert metrics["schema_compliance"] == 1.0

    # Ground truth entities = {"nexgen", "sarah chen"}
    # Predicted entities = {"nexgen", "fake person"}
    # Shared = {"nexgen"}
    # Recall = 1/2 = 0.5, Precision = 1/2 = 0.5, F1 = 0.5
    assert metrics["entity_recall"] == 0.5
    assert metrics["entity_precision"] == 0.5
    assert metrics["entity_f1"] == 0.5

    # Dates: gt_dates = {("March 22, 2024", "2024-03-22")}
    # pred_dates = {("March 22, 2024", "March 22, 2024")}
    # Shared = empty
    assert metrics["date_normalization_recall"] == 0.0
    assert metrics["date_normalization_precision"] == 0.0

    # Hallucinations: {"fake person"} -> count = 1
    assert metrics["hallucination_count"] == 1

def test_compute_metrics_flat_structure():
    gt = {
        "event_type": "funding",
        "entities": [
            {"type": "organization", "name": "NexGen"}
        ]
    }

    pred = {
        "event_type": "funding",
        "entities": ["NexGen"]  # flat list of strings instead of list of dicts
    }

    metrics = compute_metrics(pred, gt)
    assert metrics["schema_compliance"] == 0.0
    assert metrics["structural_fidelity"] == 0.0
