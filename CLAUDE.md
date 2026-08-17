# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service does

`pocket-nutrition-ml` is a **Python FastAPI service** that returns probabilistic nutritional profiles for a meal after cooking. It is an internal service — only `pocket-nutrition-api` calls it.

The caller supplies the **raw nutritional profile** of each ingredient (per 100 g, from `ingredient_variants` in PostgreSQL). The service applies cooking physics via a trained regression model and returns the post-cooking values.

The ML service does **not** query any database. It is a pure cooking-transform function.

## API contract

```
POST /predict
{
  "cooking_type": "grilled",
  "cooking_duration_min": 20,
  "ingredients": [
    {
      "ingredient_id": "poulet_filet_sans_peau",
      "weight_g": 150,
      "measured_state": "raw",
      "raw_profile": {
        "energy_kcal": 110, "protein_g": 23.4, "fat_g": 1.5,
        "carbohydrates_g": 0, "fiber_g": 0,
        "water_g": 74, "sodium_mg": 63,
        "food_group": "meat",
        "is_plant": false
      }
    }
  ]
}

→
{
  "cooking_type": "grilled",
  "cooking_duration_min": 20,
  "ingredients": [{ "ingredient_id": "poulet_filet_sans_peau", "weight_g": 150, "unknown": false }],
  "result": {
    "energy_kcal": 165.0, "protein_g": 31.0, "fat_g": 3.6,
    "carbohydrates_g": 0.0, "fiber_g": null,
    "water_g": 55.0,
    "saturated_fat_g": 1.2, "potassium_mg": 310.0,
    "confidence": 0.87
  }
}
```

`raw_profile` **must always be supplied** by the caller. If it is `None`, the ingredient is immediately marked `unknown: true` and skipped — the service does not fall back to any internal index. The responsibility for resolving `raw_profile` (from Ciqual `ingredient_variants`, Open Food Facts, or any other reference source) belongs exclusively to `pocket-nutrition-api`.

## Architecture

```
app/
  main.py      # FastAPI app + lifespan (loads model on startup)
  router.py    # POST /predict — builds feature vector, model inference, sums across ingredients
  lookup.py    # is_known_ingredient() + resolve_cooking_method()
  model.py     # joblib artifact loader — get_model() returns None if artifact missing
  schemas.py   # Pydantic MealInput / RawProfile / PredictionResponse
tests/
  test_predict.py  # Endpoint tests with mock artifact
  test_lookup.py   # Unit tests for lookup helpers
```

## Commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000   # http://localhost:8000
pytest tests/ -v
```

No linter or coverage gate is configured in this repo. Tests build their own in-memory
model artifact (`tests/test_predict.py::_make_artifact`) — no real `model.joblib` is
required to run the suite.

## Kubernetes / CI-CD

No Kubernetes manifests and no CI pipeline are included yet. In a typical deployment, the model artifact would be mounted at `/models` from a persistent volume and the service redeployed whenever a new `model.joblib` is published. Add your own manifests and CI workflow (e.g. GitHub Actions) as needed for your environment.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and the PR workflow. Behavioral changes to `app/router.py` or `app/lookup.py` must come with a test in `tests/`.

## Conventions

- `raw_profile` is the only source of ingredient nutritional data. Never add a DB lookup or artifact-index fallback here.
- `food_group` and `is_plant` come from `raw_profile`; default to `"other"` / `false` when `raw_profile` is absent.
- `ingredient_id` values are ASCII slugs from the `pocket-nutrition/ingest` repo's `pipeline/utils.py:slugify()`, mirrored in the `pocket-nutrition/api` repo's `service/Slugify.kt`.
- `confidence` must always be in `[0.0, 1.0]`.
- All code, comments, and commit messages in English.

## No hardcoded nutritional data — CRITICAL

**Never add hardcoded nutrient dictionaries to this service.** This means no `_COOKED_OVERRIDES`, no `_RAW_OVERRIDES`, no per-ingredient constant dicts, and no bypass blocks that skip model inference for specific ingredients.

This service is a pure cooking-transform function. All nutritional data belongs in the pipeline:

| Problem | Wrong fix | Right fix |
|---------|-----------|-----------|
| Ingredient raw values differ from Ciqual 2025 | Hardcode values in `router.py` | Update `pocket-nutrition/data` with the new Ciqual release and retrain |
| Model predicts wrong cooked values | Hardcode cooked values in `router.py` | Add/fix cooking relationship YAMLs in `pocket-nutrition/knowledge`, retrain via `pocket-nutrition/ingest` |
| Ingredient not in model's known set | Hardcode raw profile in `router.py` | Add ingredient YAML to `pocket-nutrition/knowledge`, add source mapping, retrain |
| Specific ingredient+method always wrong | Patch result in `router.py` | Fix the training pair in the `pocket-nutrition/ingest` repo's `pipeline/transform.py` or add a yield factor |

The `cooked_normalized_profiles` field in the model artifact is the sanctioned extension point for storing pre-computed cooked profiles — it must be populated by the `pocket-nutrition/ingest` repo's `pipeline/train.py`, not patched into `router.py`.
