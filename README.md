# Pocket Nutrition — ML Service

Python FastAPI microservice that applies cooking-transform physics to raw nutritional
profiles. Given the raw (per-100g) nutritional profile of one or more ingredients plus a
cooking method, it returns a predicted post-cooking nutritional profile for the whole meal.

This service is stateless and does not query any database — the caller must supply each
ingredient's raw nutritional profile in the request body. It is part of the larger
[Pocket Nutrition](https://pocketnutrition.org) project.

## Prerequisites

- Python 3.12 (matches the `python:3.12-slim` base image in the [Dockerfile](Dockerfile))
- `pip`

## Running locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The service loads a trained model artifact at startup from the path given by the
`MODEL_PATH` environment variable (default `/models/model.joblib`, see
[.env.example](.env.example)). If no artifact is found at that path, `/predict` still
responds but returns `result: null` (or, for the `"raw"` cooking type, a profile with
`unknown: true` per ingredient) instead of failing.

## Running with Docker

```bash
docker build -t pocket-nutrition-ml .
docker run -p 8000:8000 -e MODEL_PATH=/models/model.joblib -v /path/to/models:/models pocket-nutrition-ml
```

The container runs as a non-root user and exposes port `8000`. There is no
`docker-compose.yml` in this repository — mount your own model artifact volume as shown above.

## Testing

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

## API contract

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /predict`

Request body (`MealInput`):

```json
{
  "cooking_type": "grilled",
  "cooking_duration_min": 20,
  "ingredients": [
    {
      "ingredient_id": "chicken_breast",
      "weight_g": 150,
      "measured_state": "raw",
      "raw_profile": {
        "energy_kcal": 110,
        "protein_g": 23.4,
        "fat_g": 1.5,
        "carbohydrates_g": 0,
        "fiber_g": 0,
        "water_g": 74,
        "sodium_mg": 63,
        "food_group": "meat",
        "is_plant": false
      }
    }
  ]
}
```

- `cooking_type` (string, required) — e.g. `"grilled"`, `"boiled"`, `"steamed"`, `"fried"`,
  `"roasted"`, `"cooked"`, or `"raw"` (skips the model and returns the raw profile,
  scaled by weight).
- `cooking_duration_min` (number, optional) — echoed back in the response, not currently
  used by the model.
- `ingredients` (array, required, at least one item):
  - `ingredient_id` (string, required)
  - `weight_g` (number, required, must be > 0)
  - `measured_state` (`"raw"` or `"cooked"`, default `"raw"`) — whether `weight_g` was
    measured before or after cooking. When `"cooked"`, the service converts back to an
    equivalent raw weight using a yield factor from the model artifact before scaling.
  - `raw_profile` (object, optional) — the ingredient's raw nutritional profile per 100 g,
    plus `food_group` (string, default `"other"`) and `is_plant` (bool, default `false`).
    If omitted, the ingredient is marked `unknown: true` in the response and is excluded
    from the nutrient totals (with a confidence penalty), rather than falling back to any
    internal lookup — this service holds no ingredient database of its own.

Response body (`PredictionResponse`):

```json
{
  "cooking_type": "grilled",
  "cooking_duration_min": 20,
  "ingredients": [
    { "ingredient_id": "chicken_breast", "weight_g": 150, "unknown": false }
  ],
  "result": {
    "energy_kcal": 165.0,
    "protein_g": 31.0,
    "fat_g": 3.6,
    "carbohydrates_g": 0.0,
    "fiber_g": null,
    "water_g": 55.0,
    "sugars_g": null,
    "sodium_mg": 94.5,
    "confidence": 0.87
  },
  "absorbed_result": null
}
```

- `result` — total nutritional profile for the whole meal (summed across ingredients,
  scaled by weight), or `null` if no model artifact is loaded. `confidence` is always in
  `[0.0, 1.0]`.
- `absorbed_result` — same shape as `result` but adjusted by each ingredient's
  absorption factors (e.g. bioavailability), when the model artifact provides them for
  at least one ingredient; otherwise `null`.

See `app/schemas.py` for the full list of nutrient fields supported on `raw_profile` and
`result` (energy, macronutrients, and a range of vitamins/minerals), and `CLAUDE.md` for
architecture notes and conventions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing expectations, and the
PR workflow.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
