# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source project scaffolding: `LICENSE` (Apache-2.0), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates.
- Continuous integration workflow (lint, type-check, memory-capped tests).
- Pre-commit hooks for linting and formatting.
- Real-time inference API (`serving/`) built with FastAPI, including
  `/v1/extract`, `/healthz`, `/readyz`, and `/metrics` endpoints.
- `[project]` metadata and optional dependency groups in `pyproject.toml`.
- Serving Docker image and `docker-compose.yml`.
- Architecture and deployment documentation.

### Fixed
- Fixed an out-of-memory crash in `tests/test_run_sft.py` and
  `tests/test_run_dpo.py` caused by globally mocking `builtins.open` with a
  bare `MagicMock`, which made `yaml.safe_load()` loop forever and exhaust
  system memory. Tests now mock only the config-loading function and load a
  real config object pointed at a temporary output directory.

## [0.1.0] - Initial commit

### Added
- QLoRA SFT pipeline (`scripts/run_sft.py`) for structured JSON extraction
  using Qwen2.5-1.5B-Instruct.
- DPO preference-tuning pipeline (`scripts/run_dpo.py`).
- Dataset generation (`data/generate_dataset.py`) and evaluation
  (`scripts/evaluate.py`) tooling.
- Full pipeline orchestration (`scripts/full_pipeline.py`) and report
  generation (`scripts/generate_report.py`).
