# Structured Extraction Fine-Tuning & Serving

[![CI](https://github.com/Ashok007-cmd/structured-extraction-ft/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashok007-cmd/structured-extraction-ft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

End-to-end pipeline for fine-tuning **Qwen2.5-0.5B-Instruct** on **structured JSON extraction** — converting unstructured business text into normalized entity-relationship JSON — and serving it as a real-time REST API.

The pipeline uses **QLoRA supervised fine-tuning (SFT)** followed by **DPO preference tuning**, both engineered to run on **4 GB VRAM** (NVIDIA GTX 1650). A production-style **FastAPI** inference service, Docker images, GitHub Actions CI, and full OSS contribution scaffolding are included.

---

## Why This Project?

Structured extraction from unstructured text is a case where **prompting alone fails reliably**. Even strong models produce inconsistent output schemas, hallucinated entities, and malformed JSON between calls. Fine-tuning bakes the exact output schema into model weights, producing deterministic, schema-compliant outputs — and a thin serving layer turns that into something applications can call directly.

---

## Architecture

```
┌─────────────────────────────────────┐
│         Training Pipeline           │
│  data/generate_dataset.py           │
│       ↓  5 000 SFT + 2 000 DPO     │
│  scripts/run_sft.py  (QLoRA SFT)   │
│       ↓  outputs/sft/adapter        │
│  scripts/run_dpo.py  (DPO)         │
│       ↓  outputs/dpo/adapter        │
│  scripts/evaluate.py               │
└────────────────┬────────────────────┘
                 │ LoRA adapter
                 ▼
┌─────────────────────────────────────┐
│         FastAPI Inference API        │
│  POST /v1/extract                   │
│  GET  /healthz  /readyz  /metrics   │  ← Prometheus
└─────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component details and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Docker deployment.

---

## Project Structure

```
structured-extraction-ft/
├── configs/
│   ├── sft_config.yaml          # SFT hyperparameters (QLoRA, GTX 1650 tuned)
│   └── dpo_config.yaml          # DPO hyperparameters
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator (7 scenarios)
│   ├── schemas/
│   │   └── extraction_schema.json
│   ├── sft_dataset/             # 5 000 chat-template examples  [gitignored]
│   └── dpo_dataset/             # 2 000 preference pairs         [gitignored]
├── scripts/
│   ├── run_sft.py               # Phase 1: QLoRA SFT
│   ├── run_dpo.py               # Phase 2: DPO preference tuning
│   ├── run_dpo_fast.py          # Lightweight DPO smoke-run (200 examples)
│   ├── evaluate.py              # Phase 3: multi-metric evaluation
│   ├── full_pipeline.py         # End-to-end: dataset → SFT → DPO → eval → report
│   ├── generate_report.py       # Markdown report from training logs + eval JSON
│   ├── load_test.py             # Async load-tester for the inference API
│   └── utils/
│       ├── config.py            # Pydantic config classes (SFTConfigData, DPOConfigData)
│       └── model_loader.py      # Unified quantized model + adapter loader
├── serving/
│   ├── api.py                   # FastAPI app with lifespan, semaphore, Prometheus metrics
│   ├── inference.py             # ExtractionModel: load / unload / extract
│   ├── schemas.py               # Pydantic request / response models
│   └── settings.py              # Pydantic-settings (env var overrides via EXTRACT_*)
├── tests/                       # pytest suite — 34 tests, all CPU-only
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── report.md                # Auto-generated training + evaluation report
├── .github/workflows/ci.yml     # Lint + test on every push / PR
├── Dockerfile                   # Training image
├── Dockerfile.serve             # Serving image
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## Quick Start — Training

```bash
git clone https://github.com/Ashok007-cmd/structured-extraction-ft.git
cd structured-extraction-ft
pip install -e ".[dev]"

# Generate synthetic datasets
python data/generate_dataset.py --sft-size 5000 --dpo-size 2000

# Run the full SFT → DPO → Eval pipeline
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 \
python scripts/full_pipeline.py
```

The pipeline skips any stage whose output already exists. Pass `--force` to re-run everything from scratch.

Run stages individually:

```bash
python scripts/run_sft.py configs/sft_config.yaml
python scripts/run_dpo.py configs/dpo_config.yaml
python scripts/evaluate.py
python scripts/generate_report.py
```

---

## Quick Start — Inference API

### Local

```bash
pip install -e ".[serving]"
export EXTRACT_ADAPTER_PATH=./outputs/dpo/adapter
make serve
```

### Docker

```bash
docker compose up --build
curl http://localhost:8000/healthz
```

### Example request

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
    "financials": [{"type": "funding_raised", "amount": 50000000, "currency": "$"}],
    "relationships": [],
    "metrics": []
  },
  "raw_output": "...",
  "valid_json": true,
  "schema_valid": true,
  "latency_ms": 864.2
}
```

### Environment variables (prefix `EXTRACT_`)

| Variable | Default | Description |
|---|---|---|
| `EXTRACT_MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-0.5B-Instruct` | Base model |
| `EXTRACT_ADAPTER_PATH` | `./outputs/dpo/adapter` | LoRA adapter directory |
| `EXTRACT_USE_4BIT` | `true` | 4-bit NF4 quantization |
| `EXTRACT_MAX_NEW_TOKENS` | `512` | Max generation tokens |
| `EXTRACT_MAX_REQUEST_CHARS` | `8000` | Input size cap |
| `EXTRACT_MAX_CONCURRENCY` | `1` | Parallel inference slots |

---

## Evaluation Results

Evaluated on 30 held-out examples (base / SFT / DPO).

| Metric | Base Model | SFT | DPO |
|---|---|---|---|
| **Entity F1** | 0.737 | 0.990 | **1.000** |
| **Entity Recall** | 0.674 | 0.982 | **1.000** |
| **Entity Precision** | 0.892 | 1.000 | **1.000** |
| **Date Normalization Recall** | 0% | **100%** | **100%** |
| **Financial Recall** | 12.5% | **100%** | 94.1% |
| **Structural Fidelity** | 1.000 | 1.000 | 1.000 |
| **Avg Hallucinations / output** | 0.75 | **0.0** | **0.0** |
| Valid JSON Rate | 80% | 37% | 57% |

> **On Valid JSON Rate:** The base model produces syntactically valid JSON more often but with completely wrong schemas (0% schema compliance). SFT/DPO models generate full structured outputs that are occasionally truncated at the token budget — `max_new_tokens=512` closes most of this gap. When the fine-tuned output is valid JSON, extraction quality is near-perfect.

Key improvements over baseline: Entity F1 +0.263, date normalization 0% → 100%, hallucinations eliminated (0.75 → 0.0 per output), financial recall 12.5% → 100%.

See [`docs/report.md`](docs/report.md) for training curves, per-step logs, and the full technical writeup.

---

## Hardware

| Component | Minimum | Tested On |
|---|---|---|
| GPU VRAM | 4 GB | GTX 1650 (4 GB) |
| System RAM | 8 GB | 12 GB |
| Disk | 5 GB | — |
| CUDA | 12.x | 13.0 |

**Memory-efficiency techniques:**

- 4-bit NF4 quantization — 0.5B model fits in ~250 MB VRAM
- `paged_adamw_8bit` — offloads optimizer states to CPU
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces CUDA allocator fragmentation
- `dataloader_pin_memory=False` — prevents pinned pages from exhausting system RAM
- DPO `precompute_ref_log_probs=True` — ref pass runs once, eliminating dual-forward-pass OOM

---

## Development

```bash
make install-dev      # pip install -e ".[dev,serving,tracking]" + pre-commit
make lint             # ruff check
make format           # ruff --fix + ruff format
make type             # mypy
make test             # pytest (34 tests, CPU-only, ~5 s)
make test-mem-capped  # same under a 2 GB systemd memory cap
```

---

## Known Challenges & Solutions

| Challenge | Solution |
|---|---|
| OOM on 4 GB VRAM with 1.5B model | Switched to 0.5B + 4-bit NF4 |
| DPO dual-forward-pass OOM | `precompute_ref_log_probs=True` |
| `bfloat16` not supported on GTX 1650 | Explicit `float16` in configs |
| `merge_and_unload()` corrupts 4-bit adapters | Always load base + adapter separately |
| Laptop shutdown during training | `dataloader_pin_memory=False` |
| JSON truncated mid-output | `max_new_tokens=512` (full schema needs 300–400 tokens) |

---

## Contributing

Issues, discussions, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines, and [SECURITY.md](SECURITY.md) for reporting vulnerabilities privately.

## License

[Apache License 2.0](LICENSE)
