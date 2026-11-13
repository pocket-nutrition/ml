# Contributing to Pocket Nutrition ML

Thanks for your interest in improving this service. It is a small, single-purpose FastAPI
microservice, so the contribution process is intentionally lightweight.

## Development setup

```bash
git clone https://github.com/pocket-nutrition/ml.git
cd ml
./setup.sh                       # creates .venv, installs deps, copies .env.example → .env
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Or set up manually:

```bash
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
factors, confidence scoring, cooked-profile lookups). For any change to `app/router.py` or
`app/lookup.py`:

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

Please open a GitHub issue describing the problem. Include the request payload and response
you got from `/predict` when reporting incorrect predictions — this service has no database,
so reproducing an issue requires the exact `raw_profile` and `cooking_type` used.

## CI and releases

- Every pull request runs `.github/workflows/ci.yml`. The `ci-success` check must be green before a pull
  request can be merged, the branch must be up to date with `main`, and pull requests are squash-merged
  so history stays linear.
- If you contribute from a fork, the workflow waits for a maintainer to approve it. It runs with a
  read-only token and no secrets.
- Every commit merged to `main` is released automatically with a CalVer version, `vYYYY.M.N`: the year, the
  month, then a counter that restarts at 1 each month (for example `v2026.11.1`). You never pick a version
  number or edit a changelog; release notes are generated from the merged pull requests.
- A release publishes the container image `ghcr.io/pocket-nutrition/ml`.
