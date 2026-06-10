# Architecture

## Overview

This project has two halves that share a common model-loading layer:

1. **Training pipeline** (`scripts/`) — produces LoRA adapters via QLoRA SFT
   and DPO preference tuning.
2. **Serving layer** (`serving/`) — a FastAPI application that loads the base
   model plus a LoRA adapter and exposes a real-time extraction endpoint.

Both halves use [`scripts/utils/model_loader.py`](../scripts/utils/model_loader.py)
(`ModelLoader.load_quantized_model_and_tokenizer`) so that the quantization
config, attention implementation, and tokenizer setup are identical between
training, evaluation, and serving — eliminating train/serve skew.

## Components

### `scripts/utils/config.py`

Pydantic models (`SFTConfigData`, `DPOConfigData`) that validate
`configs/*.yaml`. Shared base class (`BaseConfigData`) holds quantization,
LoRA, and training hyperparameters.

### `scripts/run_sft.py` / `scripts/run_dpo.py`

Phase 1/2 training entry points. Each:
1. Loads and validates its YAML config.
2. Loads the quantized base model + tokenizer via `ModelLoader`.
3. Configures a `LoraConfig` and trains with TRL's `SFTTrainer` /
   `DPOTrainer`.
4. Saves the adapter under `outputs/<phase>/adapter`.

### `scripts/evaluate.py`

Loads base/SFT/DPO models in isolated subprocesses (to avoid GPU memory
fragmentation), runs greedy generation against the eval split, and computes
JSON-validity, schema-compliance, entity F1, date/financial accuracy, and
hallucination metrics against `data/schemas/extraction_schema.json`.

### `serving/inference.py`

`ExtractionModel` wraps a loaded model + tokenizer:
- `load()` — loads the base model and (if present) the adapter at
  `settings.adapter_path` via `ModelLoader`.
- `extract(text)` — builds the extraction prompt via the tokenizer's chat
  template, runs greedy generation, extracts/repairs JSON from the output
  (`extract_json`), and validates it against the extraction schema.

### `serving/api.py`

FastAPI app with:
- `POST /v1/extract` — runs `ExtractionModel.extract` and returns a
  structured response (`ExtractResponse`), including whether the output was
  valid JSON and schema-valid.
- `GET /healthz` — liveness probe (process up).
- `GET /readyz` — readiness probe (model loaded); returns 503 until the
  model finishes loading.
- `GET /metrics` — Prometheus metrics (`extract_requests_total`,
  `extract_request_latency_seconds`).

The model is loaded once at process startup via FastAPI's `lifespan` context
manager and held as a singleton on `app.state.model`.

### `serving/settings.py`

`pydantic-settings`-based configuration. All values are overridable via
`EXTRACT_*` environment variables (e.g. `EXTRACT_MODEL_NAME_OR_PATH`,
`EXTRACT_ADAPTER_PATH`, `EXTRACT_MAX_NEW_TOKENS`).

## Data Flow (Inference)

```
client ──POST /v1/extract──▶ FastAPI
                                │
                                ▼
                        ExtractionModel.extract
                                │
                  tokenizer.apply_chat_template
                                │
                          model.generate (greedy)
                                │
                       extract_json + jsonschema
                                │
                                ▼
                      ExtractResponse (JSON)
```

## Memory Safety

Training scripts and tests load multi-gigabyte models. Two patterns matter
for keeping memory bounded:

- `scripts/evaluate.py` evaluates each model variant (base/SFT/DPO) in a
  **separate subprocess** so CUDA memory is fully released between models.
- Tests that mock `main()` entry points must mock the *config-loading
  function*, not `builtins.open` — see the note in
  [CONTRIBUTING.md](../CONTRIBUTING.md#memory-safety-note-for-tests).
