# Contributing

Thank you for your interest in contributing to this project.

## Getting Started

```bash
git clone https://github.com/Ashok007-cmd/structured-extraction-ft.git
cd structured-extraction-ft
pip install -e ".[dev,serving]"
pre-commit install
```

## Development Workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Add or update tests as needed.
3. Ensure all checks pass:
   ```bash
   make lint      # ruff check
   make test      # pytest suite
   make type      # mypy
   ```
4. Open a pull request. Fill in the PR template.

## Code Style

- **Formatter / linter:** [ruff](https://docs.astral.sh/ruff/) — run `make format` before committing.
- **Type hints:** add them for all new public functions; mypy is non-blocking for now but coverage is growing.
- **Comments:** only when the *why* is non-obvious. No docstrings for internal helpers.

## Adding Scenarios or Corruption Types

Dataset generation lives in `data/generate_dataset.py`. New templates follow the same `TEMPLATES` list structure. New DPO corruption types go in `corrupt_json()`. Add a test in `tests/test_dataset_generator.py`.

## Reporting Issues

Use the GitHub issue tracker. For security vulnerabilities, follow the process in [SECURITY.md](SECURITY.md) — **do not** open a public issue.

## License

By contributing, you agree your changes will be licensed under the [Apache 2.0 License](LICENSE).
