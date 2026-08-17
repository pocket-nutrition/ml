# Contributing to Pocket Nutrition ML

Thanks for your interest in improving this service. It is a small, single-purpose FastAPI
microservice, so the contribution process is intentionally lightweight.

## Development setup

```bash
git clone https://github.com/pocket-nutrition/ml.git
cd ml

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt pytest

cp .env.example .env             # optional — defaults work without a model artifact
```

Run the service:

```bash
uvicorn app.main:app --reload --port 8000
```

Without a model artifact at `MODEL_PATH` (default `/models/model.joblib`), the service still
starts and answers `/health` and `/predict`, but `/predict` returns `result: null` (see
[README.md](README.md) for the full contract). You do not need a trained model to work on
most of the codebase — the test suite builds its own in-memory artifact (see
`tests/test_predict.py::_make_artifact`).

## Running tests

```bash
pytest tests/ -v
```

There is no code coverage gate or linter configuration in this repository — keep changes
consistent with the surrounding style (type hints on public functions, `from __future__
import annotations` where already used, English identifiers and comments).

## Test-first expectations

This is a prediction/transform service with subtle numeric behaviour (clamping, yield
factors, confidence scoring). For any change to `app/router.py` or `app/lookup.py`:

1. Add or update a test in `tests/test_predict.py` or `tests/test_lookup.py` that captures
   the behavior you're adding or fixing.
2. Confirm the test fails without your change and passes with it.
3. Run the full suite (`pytest tests/ -v`) before opening a pull request — it must pass.

Pull requests that change prediction behavior without an accompanying test will be asked to
add one.

## No hardcoded nutritional data

Read the "No hardcoded nutritional data" section in [CLAUDE.md](CLAUDE.md) before touching
`app/router.py`. This service is a pure cooking-transform function — it must not gain
per-ingredient constant dictionaries or bypass blocks. Fixes to nutritional data belong in
the training pipeline that produces the model artifact this service loads, not in this repo.

## Branch and PR workflow

1. Fork the repository and create a feature branch off `main`.
2. Make focused commits with clear messages (e.g. `fix: clamp negative yield factor`).
3. Ensure `pytest tests/ -v` passes locally.
4. Open a pull request describing the change and, for behavioral changes, what test covers it.
5. Be responsive to review feedback — small, focused PRs are easier to review and merge.

## Reporting issues

Please use the GitHub issue templates (bug report / feature request) under
`.github/ISSUE_TEMPLATE/`. Include the request payload and response you got from `/predict`
when reporting incorrect predictions — this service has no database, so reproducing an issue
requires the exact `raw_profile` and `cooking_type` used.
