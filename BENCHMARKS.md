# Benchmarks

Performance measurements for the serving layer and training pipeline.

## Inference API — Latency & Throughput

Measured on **NVIDIA GTX 1650 (4 GB VRAM)**, CUDA 13.0, PyTorch 2.x, 4-bit NF4 quantization, `do_sample=False`.

| Metric | Value |
|--------|-------|
| Cold-start (model load) | ~35–45 s |
| Warm-up request (first inference) | ~3–5 s |
| p50 latency (steady-state) | ~850 ms |
| p90 latency | ~1 100 ms |
| p99 latency | ~1 400 ms |
| Max observed latency | ~1 800 ms |
| Throughput (`max_concurrency=1`) | ~0.9 req/s |
| Input size limit | 8 000 chars |
| Max output tokens | 512 |

> All latency figures are end-to-end (HTTP request → response), including tokenization, generation, and JSON parsing.

## Memory Usage

| Component | VRAM | System RAM |
|-----------|------|------------|
| Base model (4-bit NF4) | ~250 MB | ~800 MB |
| LoRA adapter overhead | ~15 MB | ~50 MB |
| KV cache (512 tokens) | ~80 MB | — |
| Peak during generation | ~420 MB | ~1.2 GB |

> The model comfortably fits in 4 GB VRAM with headroom for the OS and display server.

## Training Pipeline

| Stage | Examples | Duration (GTX 1650) | Peak VRAM |
|-------|----------|---------------------|-----------|
| Dataset generation | 5 000 SFT + 2 000 DPO | < 10 s | — |
| SFT (QLoRA, 1 epoch) | 2 000 (subsampled) | ~12 min | ~2.8 GB |
| DPO (1 epoch) | 1 800 | ~8 min | ~3.1 GB |
| Evaluation (3 models × 30 samples) | 90 total | ~15 min | ~2.8 GB |

> Training uses `paged_adamw_8bit`, `gradient_checkpointing=True`, `precompute_ref_log_probs=True` (DPO), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to stay within 4 GB.

## Load Test

Run against a warm local server with `python scripts/load_test.py --requests 20 --concurrency 1`:

```
{
  "throughput_rps": 0.91,
  "latency_s": {
    "min":  0.782,
    "mean": 0.864,
    "p50":  0.851,
    "p90":  0.963,
    "p99":  1.102,
    "max":  1.187
  },
  "success_count": 20,
  "failure_count": 0
}
```

## Evaluation Quality

| Metric | Base Model | After SFT | After DPO |
|--------|-----------|-----------|-----------|
| Entity F1 | 0.737 | 0.990 | **1.000** |
| Date Normalization Recall | 0% | 100% | 100% |
| Financial Recall | 12.5% | 100% | 94.1% |
| Hallucinations / output | 0.75 | 0.0 | 0.0 |

See [`docs/report.md`](docs/report.md) for full per-step training curves and per-metric breakdowns.
