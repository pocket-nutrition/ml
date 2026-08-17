from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from app.main import app

client = TestClient(app)

COOKING_METHODS = ["boiled", "cooked", "fried", "grilled", "roasted", "steamed"]
FOOD_GROUPS = ["cereal", "fish", "meat", "shellfish", "vegetable"]

FEATURE_COLS = [
    "energy_kcal", "water_g", "protein_g", "fat_g", "carbohydrates_g", "fiber_g", "sodium_mg",
]
TARGET_COLS = [
    "energy_kcal", "water_g", "protein_g", "fat_g", "carbohydrates_g",
    "fiber_g", "sugars_g", "calcium_mg", "iron_mg", "magnesium_mg",
    "sodium_mg", "vitamin_c_mg",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
    "cholesterol_mg", "potassium_mg", "phosphorus_mg", "zinc_mg",
    "vitamin_a_ug", "vitamin_d_ug", "vitamin_b6_mg", "vitamin_b12_ug",
    "niacin_mg",
]

CHICKEN_PROFILE = {
    "energy_kcal": 110.0, "water_g": 74.0, "protein_g": 23.0, "fat_g": 1.5,
    "carbohydrates_g": 0.0, "fiber_g": 0.0, "sodium_mg": 63.0,
    "food_group": "meat",
}
RICE_PROFILE = {
    "energy_kcal": 356.0, "water_g": 12.0, "protein_g": 7.0, "fat_g": 0.6,
    "carbohydrates_g": 78.0, "fiber_g": 1.4, "sodium_mg": 2.0,
    "food_group": "cereal",
}
BROCCOLI_PROFILE = {
    "energy_kcal": 34.0, "water_g": 89.0, "protein_g": 2.8, "fat_g": 0.4,
    "carbohydrates_g": 7.0, "fiber_g": 2.6, "sodium_mg": 33.0,
    "food_group": "vegetable",
}


def _make_artifact(food_index: Optional[dict] = None) -> dict:
    ohe_cooking = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore", categories=[COOKING_METHODS]
    )
    ohe_cooking.fit(np.array(COOKING_METHODS).reshape(-1, 1))

    ohe_fg = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore", categories=[FOOD_GROUPS]
    )
    ohe_fg.fit(np.array(FOOD_GROUPS).reshape(-1, 1))

    # 7 raw features + 6 cooking OHE + 5 food_group OHE = 18 total features
    raw_data = np.array([
        [110.0, 74.0, 23.0, 1.5, 0.0, 0.0, 63.0],
        [356.0, 12.0, 7.0, 0.6, 78.0, 1.4, 2.0],
        [34.0, 89.0, 2.8, 0.4, 7.0, 2.6, 33.0],
    ], dtype=float)
    scaler = StandardScaler()
    raw_scaled = scaler.fit_transform(raw_data)

    cooking_enc = ohe_cooking.transform([["grilled"], ["boiled"], ["steamed"]])
    fg_enc = ohe_fg.transform([["meat"], ["cereal"], ["vegetable"]])
    X = np.hstack([raw_scaled, cooking_enc, fg_enc])

    y = np.zeros((3, len(TARGET_COLS)), dtype=float)
    y[0, :6] = [165.0, 60.0, 31.0, 3.6, 0.0, 0.0]   # chicken grilled
    y[1, :6] = [130.0, 65.0, 2.5, 0.2, 28.0, 0.4]    # rice boiled
    y[2, :6] = [25.0, 92.0, 2.0, 0.3, 5.0, 2.0]      # broccoli steamed

    model = MultiOutputRegressor(XGBRegressor(n_estimators=10, max_depth=2, random_state=42, verbosity=0))
    model.fit(X, y)

    if food_index is None:
        food_index = {
            "poulet_filet": CHICKEN_PROFILE,
            "riz": RICE_PROFILE,
            "brocoli": BROCCOLI_PROFILE,
        }

    return {
        "model": model,
        "scaler": scaler,
        "cooking_encoder": ohe_cooking,
        "food_group_encoder": ohe_fg,
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLS,
        "food_index": food_index,
        "known_ingredients": sorted(food_index.keys()),
        "plant_ingredient_ids": ["brocoli", "riz"],
    }


def _with_profile(ingredient_id: str, weight_g: float, profile: dict, measured_state: str = "raw") -> dict:
    """Build an ingredient dict with raw_profile as required by the API contract."""
    return {
        "ingredient_id": ingredient_id,
        "weight_g": weight_g,
        "measured_state": measured_state,
        "raw_profile": {**profile, "is_plant": profile.get("food_group") in ("vegetable", "cereal")},
    }


def _meal(
    cooking_type: str = "grilled",
    ingredients: Optional[list] = None,
    cooking_duration_min: Optional[float] = None,
) -> dict:
    if ingredients is None:
        ingredients = [_with_profile("poulet_filet", 150, CHICKEN_PROFILE)]
    payload = {"cooking_type": cooking_type, "ingredients": ingredients}
    if cooking_duration_min is not None:
        payload["cooking_duration_min"] = cooking_duration_min
    return payload


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_response():
    response = client.post("/predict", json=_meal())
    assert response.status_code == 200
    data = response.json()
    assert data["cooking_type"] == "grilled"
    assert isinstance(data["ingredients"], list)


def test_predict_known_ingredient():
    artifact = _make_artifact()
    with patch("app.router.get_model", return_value=artifact):
        response = client.post("/predict", json=_meal())
    assert response.status_code == 200
    data = response.json()
    assert data["result"] is not None
    assert data["result"]["confidence"] > 0.0
    assert data["result"]["energy_kcal"] > 0.0
    assert data["ingredients"][0]["unknown"] is False


def test_predict_weight_scaling():
    artifact = _make_artifact()
    with patch("app.router.get_model", return_value=artifact):
        r100 = client.post("/predict", json=_meal(
            cooking_type="boiled",
            ingredients=[_with_profile("riz", 100, RICE_PROFILE)],
        )).json()
        r200 = client.post("/predict", json=_meal(
            cooking_type="boiled",
            ingredients=[_with_profile("riz", 200, RICE_PROFILE)],
        )).json()

    assert r200["result"]["energy_kcal"] == pytest.approx(r100["result"]["energy_kcal"] * 2, rel=1e-4)
    assert r200["result"]["protein_g"] == pytest.approx(r100["result"]["protein_g"] * 2, rel=1e-4)


def test_predict_unknown_cooking_method_reduces_confidence():
    artifact = _make_artifact()
    with patch("app.router.get_model", return_value=artifact):
        known = client.post("/predict", json=_meal(cooking_type="grilled")).json()
        unknown = client.post("/predict", json=_meal(cooking_type="microwaved")).json()
    assert known["result"]["confidence"] > unknown["result"]["confidence"]


def test_predict_unknown_ingredient_flagged_and_lower_confidence():
    # An ingredient without raw_profile should be marked unknown with reduced confidence.
    artifact = _make_artifact()
    no_profile_meal = {
        "cooking_type": "grilled",
        "ingredients": [{"ingredient_id": "poulet_filet", "weight_g": 150}],
    }
    with patch("app.router.get_model", return_value=artifact):
        response = client.post("/predict", json=no_profile_meal).json()
    assert response["ingredients"][0]["unknown"] is True
    assert response["result"]["confidence"] < 1.0


def test_predict_unknown_ingredient_still_predicts():
    artifact = _make_artifact(food_index={})  # no known ingredients
    with patch("app.router.get_model", return_value=artifact):
        response = client.post("/predict", json=_meal())
    assert response.status_code == 200
    data = response.json()
    assert data["result"] is not None
    assert data["result"]["energy_kcal"] is not None


def test_predict_no_model_returns_null_result():
    with patch("app.router.get_model", return_value=None):
        response = client.post("/predict", json=_meal())
    assert response.status_code == 200
    assert response.json()["result"] is None


def test_predict_cooking_duration_echoed():
    artifact = _make_artifact()
    with patch("app.router.get_model", return_value=artifact):
        response = client.post("/predict", json=_meal(cooking_duration_min=20.0))
    assert response.json()["cooking_duration_min"] == 20.0


def test_predict_multiple_ingredients_sums_nutrients():
    artifact = _make_artifact()
    with patch("app.router.get_model", return_value=artifact):
        single = client.post("/predict", json=_meal(
            cooking_type="grilled",
            ingredients=[_with_profile("poulet_filet", 150, CHICKEN_PROFILE)],
        )).json()
        multi = client.post("/predict", json=_meal(
            cooking_type="grilled",
            ingredients=[
                _with_profile("poulet_filet", 150, CHICKEN_PROFILE),
                _with_profile("poulet_filet", 150, CHICKEN_PROFILE),
            ],
        )).json()
    assert multi["result"]["energy_kcal"] == pytest.approx(single["result"]["energy_kcal"] * 2, rel=1e-4)
    assert len(multi["ingredients"]) == 2


def test_predict_plant_cholesterol_zeroed():
    artifact = _make_artifact()
    with patch("app.router.get_model", return_value=artifact):
        response = client.post("/predict", json=_meal(
            cooking_type="steamed",
            ingredients=[_with_profile("brocoli", 100, BROCCOLI_PROFILE)],
        )).json()
    assert response["result"]["cholesterol_mg"] == 0.0
    assert response["result"]["vitamin_b12_ug"] == 0.0


def test_predict_sodium_clamped():
    artifact = _make_artifact()
    chicken_high_sodium = {**CHICKEN_PROFILE, "sodium_mg": 100.0}
    with patch("app.router.get_model", return_value=artifact):
        response = client.post("/predict", json=_meal(
            cooking_type="grilled",
            ingredients=[_with_profile("poulet_filet", 100, chicken_high_sodium)],
        )).json()
    # sodium_mg ≤ raw_sodium_mg × 1.5 × (100/100) = 150 mg
    assert response["result"]["sodium_mg"] <= 150.0
