# Deployment

## Running the API Locally

```bash
pip install -e ".[serving]"
export EXTRACT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
export EXTRACT_ADAPTER_PATH="./outputs/dpo/adapter"   # optional; falls back to base model
make serve
```

The server starts on `http://0.0.0.0:8000` by default. Check readiness:

```bash
curl http://localhost:8000/readyz
```

## Configuration Reference

All settings are environment variables with the `EXTRACT_` prefix (see
[`serving/settings.py`](../serving/settings.py)):

| Variable | Default | Description |
|---|---|---|
| `EXTRACT_MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-0.5B-Instruct` | Base model to load |
| `EXTRACT_ADAPTER_PATH` | `./outputs/dpo/adapter` | LoRA adapter directory; ignored if it doesn't exist |
| `EXTRACT_USE_4BIT` | `true` | Enable 4-bit NF4 quantization |
| `EXTRACT_BNB_4BIT_COMPUTE_DTYPE` | `bfloat16` | Compute dtype for quantized layers |
| `EXTRACT_MAX_NEW_TOKENS` | `512` | Max tokens generated per request |
| `EXTRACT_DO_SAMPLE` | `false` | Use sampling instead of greedy decoding |
| `EXTRACT_TEMPERATURE` | `0.7` | Sampling temperature (if `EXTRACT_DO_SAMPLE=true`) |
| `EXTRACT_MAX_INPUT_TOKENS` | `1024` | Max tokens of input text passed to the model |
| `EXTRACT_MAX_REQUEST_CHARS` | `8000` | Max request body size (characters) before returning 413 |
| `EXTRACT_MAX_CONCURRENCY` | `1` | Max concurrent `/v1/extract` requests served at once (a single model instance can't safely run concurrent `.generate()` calls) |
| `EXTRACT_ENABLE_WARMUP` | `true` | Run a dummy generation at startup so the first real request isn't cold |
| `EXTRACT_HOST` / `EXTRACT_PORT` | `0.0.0.0` / `8000` | Bind address |

You can also place these in a `.env` file in the project root.

## Docker

Build and run the serving image:

```bash
docker build -f Dockerfile.serve -t structured-extraction-api:latest .
docker run --rm -p 8000:8000 \
  -e EXTRACT_ADAPTER_PATH=/app/outputs/dpo/adapter \
  -v "$(pwd)/outputs:/app/outputs:ro" \
  --gpus all \
  structured-extraction-api:latest
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Health Checks

- `GET /healthz` — process liveness (always 200 once the server is up).
- `GET /readyz` — returns 503 until the model has finished loading; use this
  for Kubernetes readiness probes / load-balancer health checks.
- `GET /metrics` — Prometheus exposition format. Scrape this with Prometheus
  or any compatible agent.

## Kubernetes Notes (sketch)

- Use `/readyz` as the `readinessProbe` and `/healthz` as the
  `livenessProbe`, with a generous `initialDelaySeconds` (model loading can
  take 30–120s depending on hardware).
- Mount trained adapters (`outputs/dpo/adapter`) as a read-only volume or
  bake them into the image for immutable deployments.
- Request a GPU via `nvidia.com/gpu: 1` in resource limits if running with
  4-bit quantization on GPU; the service also runs on CPU (slower) if no GPU
  is available, since `bitsandbytes` quantization is optional
  (`EXTRACT_USE_4BIT=false`).

## Load Testing

A small async load-test script is included to sanity-check latency and
throughput against a running instance:

```bash
python scripts/load_test.py --url http://localhost:8000 --requests 50 --concurrency 5
```

It reports total time, throughput (req/s), success/failure counts, and
latency percentiles (p50/p90/p99), and can optionally write a JSON report
with `--output report.json`. Since `EXTRACT_MAX_CONCURRENCY` defaults to `1`,
requests beyond that limit will queue behind the semaphore — increase
`EXTRACT_MAX_CONCURRENCY` only if you have the GPU memory to back it.

## CI/CD

`.github/workflows/ci.yml` runs lint (`ruff`), type-checking (`mypy`), and
the test suite (memory-capped via `systemd-run` where available) on every
push and pull request to `main`.
