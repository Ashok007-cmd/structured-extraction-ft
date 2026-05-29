# Project 4: Model Fine-Tuning

Fine-tuning a small language model (**Qwen2.5-0.5B-Instruct**) on **structured JSON extraction**
— converting unstructured business text into normalized entity-relationship JSON using
**LoRA/QLoRA SFT** followed by **DPO preference tuning**.

## Why This Task?

Structured extraction from unstructured text is a classic case where **prompting alone
is insufficient**. Even powerful LLMs produce inconsistent schemas, hallucinated entities,
and malformed outputs. Fine-tuning bakes the exact output schema into model weights.

## Project Structure

```
├── configs/
│   ├── sft_config.yaml        # SFT hyperparameters
│   └── dpo_config.yaml        # DPO hyperparameters
├── data/
│   ├── generate_dataset.py    # Synthetic dataset generator
│   ├── sft_dataset/           # 5,000 SFT examples (7 scenarios)
│   └── dpo_dataset/           # 2,000 DPO preference pairs
├── scripts/
│   ├── run_sft.py             # Phase 1: QLoRA supervised fine-tuning
│   ├── run_dpo.py             # Phase 2: DPO preference tuning
│   ├── evaluate.py            # Phase 3: Multi-metric evaluation
│   └── full_pipeline.py       # End-to-end: SFT → DPO → Eval
├── outputs/                   # Trained adapters + logs
├── docs/
│   └── report.md              # Full technical report
└── requirements.txt
```

## Quick Start

```bash
pip install torch transformers accelerate peft trl datasets bitsandbytes evaluate
python data/generate_dataset.py --sft-size 5000 --dpo-size 2000
HF_HUB_OFFLINE=1 python scripts/full_pipeline.py
```

## Results

| Metric | Base Model | SFT | SFT+DPO | Δ |
|--------|-----------|-----|---------|---|
| Valid JSON Rate | 12% | 96% | 98% | **+86 pp** |
| Entity F1 | 0.082 | 0.921 | 0.947 | **+0.865** |
| Hallucinations/output | 3.2 | 0.3 | 0.1 | **-3.1** |

## Hardware

NVIDIA GTX 1650 (4GB VRAM) — feasible with QLoRA 4-bit + paged_adamw_8bit.

## Key Challenge Solved

The entire pipeline runs on 4GB VRAM by using:
- 4-bit NF4 quantization (250MB for 0.5B model)
- Batch size 1 with gradient accumulation
- 8-bit paged AdamW optimizer
- HF_HUB_OFFLINE for cached model loading
