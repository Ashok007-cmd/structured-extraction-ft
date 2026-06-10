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
  size limits, model/adapter path handling.
* Dependency supply chain (`requirements.txt`, `pyproject.toml`).
* Docker images and CI/CD workflows.
* Handling of model artifacts and configuration files (no secrets should be
  committed or baked into images).

## Disclosure Policy

We follow coordinated disclosure: once a fix is available, we will publish a
security advisory crediting the reporter (unless anonymity is requested).
