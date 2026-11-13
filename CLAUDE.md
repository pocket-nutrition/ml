# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service does

`pocket-nutrition-ml` is a **Python FastAPI service** that returns probabilistic nutritional profiles for a meal after cooking. It is designed to be called by an upstream API gateway.

The caller supplies the **raw nutritional profile** of each ingredient (per 100 g, read by the API from the Elasticsearch `product_search` index built by the ingest pipeline). The service applies cooking physics via a trained regression model and returns the post-cooking values.

The ML service does **not** query any database. It is a pure cooking-transform function.

## API contract

```
POST /predict
{
  "cooking_type": "grilled",
  "cooking_duration_min": 20,
  "ingredients": [
    {
      "ingredient_id": "chicken_breast",
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
  "ingredients": [{ "ingredient_id": "chicken_breast", "weight_g": 150, "unknown": false }],
  "result": {
    "energy_kcal": 165.0, "protein_g": 31.0, "fat_g": 3.6,
    "carbohydrates_g": 0.0, "fiber_g": null,
    "water_g": 55.0,
    "saturated_fat_g": 1.2, "potassium_mg": 310.0,
    "confidence": 0.87
  }
}
```

`raw_profile` **must always be supplied** by the caller. If it is `None`, or every nutrient in it is null, the ingredient is marked `unknown: true` and contributes no nutrients (it only lowers the meal confidence) — the service never predicts from an empty profile. When every ingredient of a meal is unknown, `result` is `null`. The responsibility for resolving `raw_profile` (from `product_search`: Ciqual/McCance/USDA via the knowledge mappings, or Open Food Facts) belongs exclusively to the API (`pocket-nutrition/api`).

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
./setup.sh                                  # first-time setup (venv, deps, .env)
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000   # http://localhost:8000
pytest tests/ -v
```

No linter or coverage gate is configured in this repo. Tests build their own in-memory
model artifact (`tests/test_predict.py::_make_artifact`) — no real `model.joblib` is
required to run the suite.

## Deployment

No Kubernetes manifests are included in this repository. In a typical deployment, the model
artifact would be mounted from a persistent volume at the path given by `MODEL_PATH` and the
service redeployed whenever a new `model.joblib` is published. See
[.github/workflows/ci.yml](.github/workflows/ci.yml) for the build/test pipeline; add your own
image-publish and deploy steps as needed for your environment.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and the PR workflow. Behavioral changes to
`app/router.py` or `app/lookup.py` must come with a test in `tests/`.

## Conventions

- `raw_profile` is the only source of ingredient nutritional data. Never add a DB lookup or artifact-index fallback here.
- `food_group` and `is_plant` come from `raw_profile`; default to `"other"` / `false` when `raw_profile` is absent.
- `ingredient_id` values are expected to be ASCII slugs produced by whatever upstream service resolves ingredient identity — this service does not generate or validate them.
- `confidence` must always be in `[0.0, 1.0]`.
- All code, comments, and commit messages in English.

## No hardcoded nutritional data — CRITICAL

**Never add hardcoded nutrient dictionaries to this service.** This means no `_COOKED_OVERRIDES`, no `_RAW_OVERRIDES`, no per-ingredient constant dicts, and no bypass blocks that skip model inference for specific ingredients.

This service is a pure cooking-transform function. All nutritional data belongs in the training
pipeline that produces `model.joblib`:

| Problem | Wrong fix | Right fix |
|---------|-----------|-----------|
| Ingredient raw values differ from Ciqual 2025 | Hardcode values in `router.py` | Update `pocket-nutrition/data` with the new Ciqual release and retrain |
| Model predicts wrong cooked values | Hardcode cooked values in `router.py` | Add/fix cooking relationship YAMLs in `pocket-nutrition/knowledge`, retrain via `pocket-nutrition/ingest` |
| Ingredient not in model's known set | Hardcode raw profile in `router.py` | Add ingredient YAML to `pocket-nutrition/knowledge`, add source mapping, retrain |
| Specific ingredient+method always wrong | Patch result in `router.py` | Fix the training pair in `pipeline/transform.py` of `pocket-nutrition/ingest` or add a yield factor |

All predictions and training targets are **per 100 g of RAW input**. A cooked weight is converted to its raw equivalent with the ingredient's yield before scaling. Never express targets per 100 g cooked: that applies the yield twice.

The artifact must contain `model`, `scaler`, `feature_columns` and `target_columns`; `load_model()` refuses to start otherwise.
