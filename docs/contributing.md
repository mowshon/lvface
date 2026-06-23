# Contributing

Changes to preprocessing, alignment, model resolution, or inference must preserve the frozen
golden embedding. Run Ruff, mypy, and the full pytest suite before opening a pull request.

Do not commit model weights or biometric photo datasets. Every committed image fixture must be
synthetic or openly licensed and documented in `tests/data/FIXTURES.md`.
