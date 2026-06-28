# Security Policy

## Supported Versions

This project is under active development. Security fixes are applied to the
`main` branch and included in the latest tagged release.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < latest | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public
GitHub issue**. Instead, report it privately using one of the following
channels:

* GitHub: open a [private security advisory](../../security/advisories/new)
  for this repository (preferred).
* Email the maintainers with a description of the issue, steps to reproduce,
  and any relevant logs or proof-of-concept code.

We will acknowledge your report within 5 business days and aim to provide a
fix or mitigation plan within 30 days, depending on severity.

## Scope

Areas of particular interest for security review:

* The FastAPI inference service (`serving/`) — input validation, request
  size limits, inference timeout, model/adapter path handling.
* Dependency supply chain (`requirements.txt`, `pyproject.toml`).
* Docker images and CI/CD workflows (`Dockerfile.serve` runs as non-root `appuser`).
* Handling of model artifacts and configuration files (no secrets should be
  committed or baked into images). Use `.env.example` as a template — never
  commit a `.env` file with real credentials.

## Security Architecture Notes

| Control | Implementation |
|---------|---------------|
| Non-root container | `appuser` in `Dockerfile.serve` |
| Input size cap | `EXTRACT_MAX_REQUEST_CHARS` (default 8 000 chars) |
| Inference timeout | `EXTRACT_INFERENCE_TIMEOUT_SECONDS` (default 120 s) |
| Concurrency cap | `EXTRACT_MAX_CONCURRENCY` semaphore |
| No arbitrary code | `trust_remote_code=False` in all model loading paths |
| Config validation | Only known dataclass fields accepted from YAML overrides |

In production, the API should be placed behind a reverse proxy (nginx, Caddy) or
API gateway with authentication (API keys, mTLS) — the current serving layer does
not include authentication by design to keep the inference path dependency-free.

## Disclosure Policy

We follow coordinated disclosure: once a fix is available, we will publish a
security advisory crediting the reporter (unless anonymity is requested).
