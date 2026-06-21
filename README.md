<div align="center">

# 🧠 Structured Extraction Fine-Tuning & Serving

**Production-grade QLoRA SFT + DPO pipeline that turns unstructured text into schema-validated JSON — runs on a 4 GB laptop GPU.**

[![CI](https://github.com/Ashok007-cmd/structured-extraction-ft/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashok007-cmd/structured-extraction-ft/actions/workflows/ci.yml)
[![Docker](https://github.com/Ashok007-cmd/structured-extraction-ft/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Ashok007-cmd/structured-extraction-ft/actions/workflows/docker-publish.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GHCR](https://img.shields.io/badge/Docker-ghcr.io-blue?logo=docker)](https://ghcr.io/ashok007-cmd/structured-extraction-ft)
[![PyPI](https://img.shields.io/badge/PyPI-structured--extraction--ft-orange?logo=pypi)](https://pypi.org/p/structured-extraction-ft)

</div>

---

## What This Project Does

Fine-tuning solves what prompting cannot: **consistent, schema-compliant, zero-hallucination JSON extraction** from unstructured business text. This project provides:

- **A complete training pipeline** — synthetic dataset generation → QLoRA SFT → DPO preference tuning → multi-metric evaluation
- **A production FastAPI inference server** — with Prometheus metrics, readiness probes, and concurrency control  
- **Docker images** published to GitHub Container Registry (GHCR)
- **34 CPU-only tests** that run in ~5s with no GPU required

All of this runs on a **GTX 1650 (4 GB VRAM)** through careful memory engineering.

---

## Results

Evaluated on 30 held-out examples across base model / SFT / DPO:

| Metric | Base Model | After SFT | After DPO |
|---|---|---|---|
| **Entity F1** | 0.737 | 0.990 | **1.000** |
| **Entity Recall** | 0.674 | 0.982 | **1.000** |
| **Entity Precision** | 0.892 | 1.000 | **1.000** |
| **Date Normalization** | 0% | 100% | **100%** |
| **Financial Recall** | 12.5% | 100% | **94.1%** |
| **Hallucinations / output** | 0.75 | 0.0 | **0.0** |
| **Structural Fidelity** | 1.000 | 1.000 | **1.000** |

> **Key improvements:** Entity F1 +0.263 · Date normalization 0% → 100% · Hallucinations completely eliminated · Financial recall 12.5% → 100%

See [docs/report.md](docs/report.md) for full training curves and per-step logs.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Training Pipeline                  │
│                                                     │
│  data/generate_dataset.py  →  5 000 SFT examples   │
│                            →  2 000 DPO pairs       │
│                                                     │
│  scripts/run_sft.py   (QLoRA SFT, 4-bit NF4)       │
│         ↓  outputs/sft/adapter                      │
│  scripts/run_dpo.py   (DPO preference tuning)       │
│         ↓  outputs/dpo/adapter                      │
│  scripts/evaluate.py  (entity F1, dates, financials)│
└────────────────────────┬────────────────────────────┘
                         │  LoRA adapter
                         ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Inference Server               │
│                                                     │
│  POST /v1/extract   →  structured JSON output       │
│  GET  /healthz      →  liveness probe               │
│  GET  /readyz       →  readiness probe              │
│  GET  /metrics      →  Prometheus metrics           │
└─────────────────────────────────────────────────────┘
```

**Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) fine-tuned with QLoRA (4-bit NF4) on synthetic business-text extraction examples.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full component breakdown and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker deployment.

---

## Quick Start

### 1. Training

```bash
git clone https://github.com/Ashok007-cmd/structured-extraction-ft.git
cd structured-extraction-ft
pip install -e ".[dev]"

# Generate synthetic datasets (5 000 SFT + 2 000 DPO examples)
python data/generate_dataset.py --sft-size 5000 --dpo-size 2000

# Run the full SFT → DPO → Evaluation pipeline
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
python scripts/full_pipeline.py
```

Run stages individually:

```bash
python scripts/run_sft.py configs/sft_config.yaml   # Phase 1: QLoRA SFT
python scripts/run_dpo.py configs/dpo_config.yaml   # Phase 2: DPO tuning
python scripts/evaluate.py                           # Phase 3: Evaluation
python scripts/generate_report.py                    # Phase 4: Report
```

### 2. Inference API

**Local:**
```bash
pip install -e ".[serving]"
export EXTRACT_ADAPTER_PATH=./outputs/dpo/adapter
make serve
```

**Docker (GHCR):**
```bash
docker pull ghcr.io/ashok007-cmd/structured-extraction-ft:latest
docker compose up
curl http://localhost:8000/healthz
```

### 3. Example Request

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
    "dates": [{"raw": "March 3, 2026", "normalized": "2026-03-03"}],
    "financials": [{"type": "funding_raised", "amount": 50000000, "currency": "$"}],
    "relationships": [],
    "metrics": []
  },
  "valid_json": true,
  "schema_valid": true,
  "latency_ms": 864.2
}
```

---

## Project Structure

```
structured-extraction-ft/
├── configs/
│   ├── sft_config.yaml          # QLoRA SFT hyperparameters (GTX 1650 tuned)
│   └── dpo_config.yaml          # DPO hyperparameters
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator (7 scenarios)
│   └── schemas/extraction_schema.json
├── scripts/
│   ├── run_sft.py               # Phase 1: QLoRA supervised fine-tuning
│   ├── run_dpo.py               # Phase 2: DPO preference tuning
│   ├── evaluate.py              # Phase 3: multi-metric evaluation
│   ├── full_pipeline.py         # End-to-end orchestration
│   └── utils/
│       ├── config.py            # Pydantic config classes
│       └── model_loader.py      # Quantized model + adapter loader
├── serving/
│   ├── api.py                   # FastAPI app (lifespan, semaphore, metrics)
│   ├── inference.py             # ExtractionModel: load / unload / extract
│   ├── schemas.py               # Pydantic request/response models
│   └── settings.py              # Pydantic-settings (EXTRACT_* env vars)
├── tests/                       # 34 pytest tests — all CPU-only, ~5s
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── report.md                # Auto-generated training + eval report
├── .github/workflows/
│   ├── ci.yml                   # Lint + test on every push/PR
│   ├── publish.yml              # Publish to PyPI on release
│   └── docker-publish.yml       # Build & push Docker images to GHCR
├── Dockerfile                   # Training image
├── Dockerfile.serve             # Serving image
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## Configuration

Environment variables for the inference server (prefix `EXTRACT_`):

| Variable | Default | Description |
|---|---|---|
| `EXTRACT_MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-0.5B-Instruct` | Base model |
| `EXTRACT_ADAPTER_PATH` | `./outputs/dpo/adapter` | LoRA adapter directory |
| `EXTRACT_USE_4BIT` | `true` | 4-bit NF4 quantization |
| `EXTRACT_MAX_NEW_TOKENS` | `512` | Max generation tokens |
| `EXTRACT_MAX_REQUEST_CHARS` | `8000` | Input size cap |
| `EXTRACT_MAX_CONCURRENCY` | `1` | Parallel inference slots |

---

## Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU VRAM | 4 GB | GTX 1650 (4 GB) |
| System RAM | 8 GB | 12 GB |
| Disk | 5 GB | — |
| CUDA | 12.x | 13.0 |

**Memory-efficiency techniques used:**
- 4-bit NF4 quantization — 0.5B model fits in ~250 MB VRAM
- `paged_adamw_8bit` — offloads optimizer states to CPU
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces allocator fragmentation
- `dataloader_pin_memory=False` — prevents pinned pages from exhausting system RAM
- `precompute_ref_log_probs=True` (DPO) — reference pass runs once, eliminating dual-forward OOM

---

## Development

```bash
make install-dev   # pip install -e ".[dev,serving,tracking]" + pre-commit hooks
make lint          # ruff check
make format        # ruff --fix + ruff format
make type          # mypy
make test          # pytest (34 tests, CPU-only, ~5s)
make test-mem-capped  # same under a 2 GB systemd memory cap
```

---

## Known Challenges & Solutions

| Challenge | Solution |
|---|---|
| OOM on 4 GB VRAM with 1.5B model | Switched to 0.5B + 4-bit NF4 |
| DPO dual-forward-pass OOM | `precompute_ref_log_probs=True` |
| bfloat16 not supported on GTX 1650 | Explicit float16 in configs |
| `merge_and_unload()` corrupts 4-bit adapters | Always load base + adapter separately |
| Laptop shutdown during training | `dataloader_pin_memory=False` |
| JSON truncated mid-output | `max_new_tokens=512` (full schema: 300–400 tokens) |

---

## Docker Images

Pre-built images are published to GitHub Container Registry on every push to `main`:

```bash
# Inference/serving image
docker pull ghcr.io/ashok007-cmd/structured-extraction-ft:latest

# Training image
docker pull ghcr.io/ashok007-cmd/structured-extraction-ft-train:main
```

---

## Contributing

Issues, discussions, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines, and [SECURITY.md](SECURITY.md) for reporting vulnerabilities privately.

---

## License

[Apache License 2.0](LICENSE) — © 2024–2026 Ashok Kumar V
