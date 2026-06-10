import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Enforce project path import for serving/scripts
sys.path.append(str(Path(__file__).parent.parent.resolve()))


@pytest.fixture
def client():
    """Build a TestClient with model loading mocked out."""
    with patch("serving.inference.ModelLoader.load_quantized_model_and_tokenizer") as mock_load:
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "PROMPT"
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 1

        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 5)
        mock_tok.return_value = {"input_ids": mock_inputs}

        mock_model = MagicMock()
        mock_model.device = "cpu"

        mock_output_ids = MagicMock()
        mock_model.generate.return_value = [mock_output_ids]

        mock_load.return_value = (mock_model, mock_tok)

        from serving.api import app

        with TestClient(app) as test_client:
            # Override the decode call to return a canned JSON extraction.
            test_client.app.state.model.tokenizer.decode.return_value = (
                '{"event_type": "funding_round", "entities": '
                '[{"type": "organization", "name": "Acme Corp"}]}'
            )
            yield test_client


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_when_loaded(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_extract_returns_parsed_json(client):
    resp = client.post("/v1/extract", json={"text": "Acme Corp raised $10M."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid_json"] is True
    assert body["result"]["event_type"] == "funding_round"
    assert body["result"]["entities"][0]["name"] == "Acme Corp"


def test_extract_rejects_oversized_input(client):
    settings = client.app.state.settings
    too_long = "x" * (settings.max_request_chars + 1)
    resp = client.post("/v1/extract", json={"text": too_long})
    assert resp.status_code == 413


def test_extract_rejects_empty_text(client):
    resp = client.post("/v1/extract", json={"text": ""})
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"extract_requests_total" in resp.content


def test_readyz_when_not_loaded():
    with patch("serving.inference.ModelLoader.load_quantized_model_and_tokenizer") as mock_load:
        mock_load.side_effect = RuntimeError("boom")

        from serving.api import app

        with TestClient(app) as test_client:
            resp = test_client.get("/readyz")
            assert resp.status_code == 503
