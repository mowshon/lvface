# Contributing

Install the development dependencies and run the complete local checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
```

Changes to preprocessing, alignment, model resolution, or ONNX inference must keep the frozen
golden-embedding tests green. Do not commit model weights, face-photo datasets, caches, or
generated build artifacts. New image fixtures must be synthetic or openly licensed and have
their source, license, and attribution recorded in `tests/data/FIXTURES.md`.
