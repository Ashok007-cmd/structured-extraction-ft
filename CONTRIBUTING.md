# Contributing

Thanks for your interest in contributing! This project is a QLoRA
fine-tuning pipeline plus a real-time inference API for structured JSON
extraction. Contributions of all kinds are welcome — bug reports, docs,
tests, new features, and performance improvements.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a feature branch off `main`:
   ```bash
   git checkout -b feature/my-change
   ```
2. Make your changes. Keep diffs focused — one logical change per PR.
3. Run the test suite before opening a PR:
   ```bash
   ./scripts/run_tests.sh
   ```
4. Run lint and type checks:
   ```bash
   ruff check .
   mypy .
   ```
5. Commit with a clear message describing *why* the change was made.
6. Push your branch and open a pull request against `main`.

## Pull Request Guidelines

- Fill out the PR template completely.
- Reference any related issues (`Fixes #123`).
- Add or update tests for any behavioral change.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Ensure CI passes (lint, type-check, tests).

## Memory-Safety Note for Tests

Several tests in `tests/` mock heavy ML dependencies (`transformers`, `trl`,
`datasets`). When mocking `main()` entry points that call config loaders:

- **Do not** globally patch `builtins.open` with a bare `MagicMock()`. If any
  code path passes the mock to `yaml.safe_load()`, PyYAML's reader will loop
  forever trying to detect EOF, growing memory until the process is OOM-killed.
- Instead, patch the specific config-loading function (e.g.
  `SFTConfigData.from_yaml`) and return a real config object loaded via a
  normal `open()` + `yaml.safe_load()` call, with `output_dir` redirected to
  pytest's `tmp_path` fixture.

## Reporting Bugs

Use the "Bug report" issue template. Include:
- Steps to reproduce
- Expected vs. actual behavior
- Environment (OS, Python version, GPU/VRAM if relevant)
- Relevant logs/tracebacks

## Code Style

- Python 3.10+, formatted and linted with `ruff` (see `pyproject.toml`).
- Type hints are encouraged; `mypy` runs in CI (non-strict mode).
- Keep functions small and focused; avoid premature abstraction.

## License

By contributing, you agree that your contributions will be licensed under
the [Apache License 2.0](LICENSE).
