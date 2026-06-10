# Structured Extraction Fine-Tuning & Serving

[![CI](https://github.com/your-org/structured-extraction-ft/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/structured-extraction-ft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

An open-source, end-to-end pipeline for fine-tuning a small language model
(**Qwen2.5-0.5B-Instruct**) on **structured JSON extraction** — converting
unstructured business text into normalized entity-relationship JSON — and
serving it as a **real-time inference API**.

The pipeline uses **LoRA/QLoRA SFT** followed by **DPO preference tuning**,
and ships with a production-style **FastAPI** service, Docker images, CI,
and OSS contribution scaffolding so others can build on top of it.

## Why This Project?

Structured extraction from unstructured text is a classic case where
**prompting alone is insufficient**. Even powerful LLMs produce inconsistent
schemas, hallucinated entities, and malformed outputs. Fine-tuning bakes the
exact output schema into model weights — and a thin serving layer turns that
into something you can actually call from an application.

## Architecture

```
                ┌──────────────────────┐
                │   Training Pipeline   │
                │ run_sft.py → run_dpo.py │
                │   (QLoRA + DPO, TRL)   │
                └──────────┬────────────┘
                           │ adapters
                           ▼
                ┌──────────────────────┐
  HTTP request  │   FastAPI Service     │  /healthz, /readyz, /metrics
 ─────────────▶ │   serving/api.py       │ ◀── Prometheus scrape
                │   (ModelLoader +       │
                │    QLoRA adapter)      │
                └──────────┬────────────┘
                           │ JSON
                           ▼
                  Structured extraction
                  (event_type, entities,
                   dates, financials, ...)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details and
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for running the API in Docker.

## Project Structure

```
├── configs/
│   ├── sft_config.yaml        # SFT hyperparameters
│   └── dpo_config.yaml        # DPO hyperparameters
├── data/
│   ├── generate_dataset.py    # Synthetic dataset generator
│   ├── schemas/                # JSON schema for extraction output
│   ├── sft_dataset/            # 5,000 SFT examples (7 scenarios)
│   └── dpo_dataset/             # 2,000 DPO preference pairs
├── scripts/
│   ├── run_sft.py              # Phase 1: QLoRA supervised fine-tuning
│   ├── run_dpo.py               # Phase 2: DPO preference tuning
│   ├── evaluate.py              # Phase 3: Multi-metric evaluation
│   ├── full_pipeline.py         # End-to-end: SFT → DPO → Eval
│   └── utils/                    # Config + model loading utilities
├── serving/
│   ├── api.py                    # FastAPI app (real-time inference)
│   ├── inference.py              # Model loading + extraction logic
│   ├── schemas.py                # Pydantic request/response models
│   └── settings.py               # Environment-based configuration
├── tests/                          # pytest suite (training + serving)
├── outputs/                         # Trained adapters + logs
├── docs/                             # Architecture, deployment, report
├── Dockerfile                        # Training image
├── Dockerfile.serve                  # Serving image
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Quick Start — Training

```bash
pip install -e ".[dev]"
python data/generate_dataset.py --sft-size 5000 --dpo-size 2000
HF_HUB_OFFLINE=1 python scripts/full_pipeline.py
```

## Quick Start — Real-Time Inference API

```bash
pip install -e ".[serving]"

# Point at your fine-tuned adapter (defaults to ./outputs/dpo/adapter)
export EXTRACT_ADAPTER_PATH=./outputs/dpo/adapter

make serve
# or: uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

Call the API:

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Acme Corp announced a $50M Series B led by Example Ventures on March 3, 2026."}'
```

```json
{
  "result": {
    "event_type": "funding_round",
    "entities": [
      {"type": "organization", "name": "Acme Corp"},
      {"type": "organization", "name": "Example Ventures"}
    ],
    "dates": [{"raw": "March 3, 2026", "normalized": "2026-03-03", "context": "announcement"}],
    "financials": [{"type": "raise", "amount": 50000000, "currency": "USD"}],
    "relationships": [],
    "metrics": []
  },
  "raw_output": "{...}",
  "valid_json": true,
  "schema_valid": true,
  "latency_ms": 842.3
}
```

Run with Docker Compose:

```bash
docker compose up --build
```

## Results

| Metric | Base Model | SFT | SFT+DPO | Δ |
|--------|-----------|-----|---------|---|
| Valid JSON Rate | 12% | 96% | 98% | **+86 pp** |
| Entity F1 | 0.082 | 0.921 | 0.947 | **+0.865** |
| Hallucinations/output | 3.2 | 0.3 | 0.1 | **-3.1** |

See [docs/report.md](docs/report.md) for the full technical writeup.

## Hardware

NVIDIA GTX 1650 (4GB VRAM) — feasible with QLoRA 4-bit + paged_adamw_8bit.

## Key Challenge Solved

The entire pipeline runs on 4GB VRAM by using:
- 4-bit NF4 quantization (250MB for 0.5B model)
- Batch size 1 with gradient accumulation
- 8-bit paged AdamW optimizer
- HF_HUB_OFFLINE for cached model loading

## Contributing

This is an open-source project — issues, discussions, and pull requests are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, the
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines, and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

Licensed under the [Apache License 2.0](LICENSE).
