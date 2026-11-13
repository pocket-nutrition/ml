from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter

from app.lookup import resolve_cooking_method
from app.model import get_model
from app.schemas import IngredientInput, IngredientResult, MealInput, NutritionalResult, PredictionResponse

router = APIRouter()
log = logging.getLogger(__name__)

_DAIRY_GROUPS = frozenset({"dairy"})
_SODIUM_CLAMP_RATIO = 1.5
_SODIUM_ABS_CAP_PER_100G = 250.0

# Per-cooking-method upper bound on predicted fat (multiplier on raw fat, per 100 g raw basis).
# Grilling/roasting render fat out rather than concentrating it; fried absorbs oil.
# Applies only when raw_fat < 5.0g (lean ingredients).
_FAT_CAP_BY_METHOD: dict[str, float] = {
    "grilled": 2.3,
    "roasted": 2.3,
    "steamed": 1.5,
    "boiled":  1.5,
    "fried":   4.0,
    "cooked":  2.5,
}


def _atwater_energy(nutrients: dict) -> float:
    p = nutrients.get("protein_g") or 0.0
    f = nutrients.get("fat_g") or 0.0
    c = nutrients.get("carbohydrates_g") or 0.0
    fiber = nutrients.get("fiber_g") or 0.0
    available_carbs = max(0.0, c - fiber)
    return p * 4.0 + f * 9.0 + available_carbs * 4.0 + fiber * 2.0


_NO_YIELD_CONFIDENCE = 0.8


def _cooked_yield(ingredient_id: str, food_group: str, method: str, artifact: dict) -> float | None:
    """Cooked/raw mass ratio used to turn a cooked weight into its raw equivalent.

    Order: the ingredient's yield for this method, its generic "cooked" yield (Ciqual names
    like "Riz blanc, cuit" are stored under "cooked"), then the median known yield of its
    food group for this method (computed by ingest). None when nothing applies.
    """
    own = (artifact.get("yield_factors") or {}).get(ingredient_id, {})
    group = (artifact.get("group_yield_factors") or {}).get(food_group, {})
    for candidate in (own.get(method), own.get("cooked"), group.get(method)):
        if candidate and candidate > 0:
            return float(candidate)
    return None


def _known_nutrients(ing: IngredientInput) -> dict | None:
    """Raw nutrients supplied by the caller, or None when there is nothing to transform.

    A profile whose nutrient fields are all null carries no information: predicting from it
    would mean predicting from an all-zero feature vector, i.e. inventing values.
    """
    if ing.raw_profile is None:
        return None
    nutrients = ing.raw_profile.as_nutrient_dict()
    if all(v is None for v in nutrients.values()):
        return None
    return nutrients


def _build_feature_vector(
    raw: dict | None,
    feature_cols: list[str],
    canonical_method: str,
    food_group: str,
    artifact: dict,
) -> np.ndarray:
    # A nutrient the source did not measure is missing (NaN), not zero: the artifact's scaler
    # imputes it the same way training did, and XGBoost routes any remaining NaN natively.
    raw_vals = [
        float(raw[c]) if raw and raw.get(c) is not None else np.nan
        for c in feature_cols
    ]
    raw_scaled = artifact["scaler"].transform(np.array([raw_vals], dtype=float))

    cooking_encoder = artifact.get("cooking_encoder")
    if cooking_encoder is not None:
        cooking_vec = cooking_encoder.transform([[canonical_method]])
    else:
        le = artifact["label_encoder"]
        cooking_vec = np.array([[float(le.transform([canonical_method])[0])]])

    food_group_encoder = artifact.get("food_group_encoder")
    if food_group_encoder is not None:
        fg_vec = food_group_encoder.transform([[food_group]])
        return np.hstack([raw_scaled, cooking_vec, fg_vec])

    return np.hstack([raw_scaled, cooking_vec])


def _predict_raw(meal: MealInput, artifact: dict | None) -> PredictionResponse:
    """Return raw nutritional profiles from raw_index directly — no model inference."""
    if artifact is None:
        return PredictionResponse(
            cooking_type=meal.cooking_type,
            cooking_duration_min=meal.cooking_duration_min,
            ingredients=[
                IngredientResult(ingredient_id=ing.ingredient_id, weight_g=ing.weight_g, unknown=True)
                for ing in meal.ingredients
            ],
            result=None,
        )

    raw_index: dict = artifact.get("raw_index") or artifact.get("food_index", {})
    target_cols: list[str] = artifact["target_columns"]

    ingredient_results: list[IngredientResult] = []
    total_nutrients: dict[str, float] = {}
    ingredient_confs: list[float] = []

    for ing in meal.ingredients:
        profile = _known_nutrients(ing) or raw_index.get(ing.ingredient_id)
        unknown = profile is None
        if unknown:
            log.debug("ingredient_id '%s': no usable raw_profile (unknown=True)", ing.ingredient_id)
        ingredient_results.append(IngredientResult(
            ingredient_id=ing.ingredient_id,
            weight_g=ing.weight_g,
            unknown=unknown,
        ))
        if not unknown:
            scale = ing.weight_g / 100.0
            for col in target_cols:
                val = profile.get(col)
                if val is not None:
                    total_nutrients[col] = total_nutrients.get(col, 0.0) + max(0.0, float(val)) * scale
        ingredient_confs.append(0.7 if unknown else 1.0)

    if all(r.unknown for r in ingredient_results):
        return PredictionResponse(
            cooking_type=meal.cooking_type,
            cooking_duration_min=meal.cooking_duration_min,
            ingredients=ingredient_results,
            result=None,
        )

    confidence = round(sum(ingredient_confs) / len(ingredient_confs), 4) if ingredient_confs else 0.0
    nutrient_fields = {col: total_nutrients.get(col) for col in target_cols}
    return PredictionResponse(
        cooking_type=meal.cooking_type,
        cooking_duration_min=meal.cooking_duration_min,
        ingredients=ingredient_results,
        result=NutritionalResult(**nutrient_fields, confidence=confidence),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(meal: MealInput) -> PredictionResponse:
    artifact = get_model()
    log.info(
        "predict: cooking_type=%s ids=%s",
        meal.cooking_type,
        [(i.ingredient_id, i.measured_state, i.raw_profile is not None) for i in meal.ingredients],
    )
    # Fast path: "raw" cooking — skip model, return raw profiles directly.
    if meal.cooking_type.strip().lower() == "raw":
        return _predict_raw(meal, artifact)

    if artifact and "cooking_encoder" in artifact:
        cooking_categories = list(artifact["cooking_encoder"].categories_[0])
    elif artifact and "label_encoder" in artifact:
        cooking_categories = list(artifact["label_encoder"].classes_)
    else:
        cooking_categories = []

    canonical_method, method_known = resolve_cooking_method(meal.cooking_type, cooking_categories)
    method_conf = 1.0 if method_known else 0.8

    food_index = artifact.get("food_index", {}) if artifact else {}
    feature_cols = artifact["feature_columns"] if artifact else []

    ingredient_results: list[IngredientResult] = []
    total_nutrients: dict[str, float] = {}
    total_absorbed: dict[str, float] = {}
    ingredient_confs: list[float] = []

    for ing in meal.ingredients:
        raw = _known_nutrients(ing)
        unknown = raw is None
        ingredient_results.append(IngredientResult(
            ingredient_id=ing.ingredient_id,
            weight_g=ing.weight_g,
            unknown=unknown,
        ))

        if artifact is None:
            continue
        if unknown:
            # Nothing to transform: contribute no nutrients, only lower the meal confidence.
            log.debug("ingredient_id '%s': no usable raw_profile (unknown=True)", ing.ingredient_id)
            ingredient_confs.append(0.7)
            continue

        food_group = ing.raw_profile.food_group
        is_plant = ing.raw_profile.is_plant

        target_cols = artifact["target_columns"]
        X = _build_feature_vector(raw, feature_cols, canonical_method, food_group, artifact)
        y_pred = artifact["model"].predict(X)[0]

        # Clamp predicted fat and protein to biological maximums.
        # For plants: cooking conserves protein/fat per gram of raw input — cap at raw × 1.1.
        # For other foods: fat is capped per cooking method (frying absorbs oil).
        # All values are per 100 g of RAW input, so water loss does not concentrate them here.
        raw_fat = (raw.get("fat_g") or 0.0) if raw else 0.0
        raw_protein = (raw.get("protein_g") or 0.0) if raw else 0.0
        fat_pred_idx = target_cols.index("fat_g") if "fat_g" in target_cols else -1
        prot_pred_idx = target_cols.index("protein_g") if "protein_g" in target_cols else -1
        raw_carbs = (raw.get("carbohydrates_g") or 0.0) if raw else 0.0
        carb_pred_idx = target_cols.index("carbohydrates_g") if "carbohydrates_g" in target_cols else -1
        if is_plant:
            if fat_pred_idx >= 0 and raw_fat < 5.0:
                y_pred[fat_pred_idx] = min(float(y_pred[fat_pred_idx]), raw_fat * 1.1)
            if prot_pred_idx >= 0 and raw_protein < 15.0:
                y_pred[prot_pred_idx] = min(float(y_pred[prot_pred_idx]), raw_protein * 1.1)
        else:
            if fat_pred_idx >= 0 and raw_fat < 5.0:
                fat_cap_mult = _FAT_CAP_BY_METHOD.get(canonical_method, 2.5)
                y_pred[fat_pred_idx] = min(float(y_pred[fat_pred_idx]), raw_fat * fat_cap_mult)
            # If raw carbs are trace (< 1g/100g), cooking cannot generate carbohydrates —
            # zero the prediction for all non-plant, non-dairy ingredients regardless of food_group.
            # (Handles meat/fish that land in "other" due to missing training pairs.)
            if carb_pred_idx >= 0 and raw_carbs < 1.0 and food_group not in _DAIRY_GROUPS:
                y_pred[carb_pred_idx] = 0.0

        # Carbs conservation: cooking cannot synthesize new carbohydrates per 100g raw basis.
        # Cap predicted carbs at the raw value for all ingredients and cooking methods.
        if carb_pred_idx >= 0 and raw_carbs > 0:
            y_pred[carb_pred_idx] = min(float(y_pred[carb_pred_idx]), raw_carbs)

        # When the caller measured the cooked weight, convert to equivalent raw input
        # weight so the scale factor matches the model's training basis (per 100g raw).
        ing_conf = 1.0
        if ing.measured_state == "cooked":
            ing_yield = _cooked_yield(ing.ingredient_id, food_group, canonical_method, artifact)
            if ing_yield is None:
                # No yield at all: the cooked weight is used as raw weight, which can be far
                # off (meat loses ~25 %, rice gains ~150 %). Say so through the confidence.
                log.debug("no yield for '%s' %s: cooked weight used as raw", ing.ingredient_id, canonical_method)
                ing_conf = _NO_YIELD_CONFIDENCE
                raw_weight_g = ing.weight_g
            else:
                raw_weight_g = ing.weight_g / ing_yield
        else:
            raw_weight_g = ing.weight_g

        scale = raw_weight_g / 100.0
        per_ing = {col: max(0.0, float(val)) * scale for col, val in zip(artifact["target_columns"], y_pred)}

        if is_plant:
            per_ing["cholesterol_mg"] = 0.0
            per_ing["vitamin_b12_ug"] = 0.0

        if food_group in _DAIRY_GROUPS and prot_pred_idx >= 0 and raw_protein < 20.0:
            per_ing["protein_g"] = min(
                per_ing.get("protein_g", 0.0),
                raw_protein * 1.1 * scale,
            )

        raw_sodium_mg = (raw.get("sodium_mg") or 0.0) if raw else 0.0
        if raw_sodium_mg > 0:
            sodium_cap = raw_sodium_mg * _SODIUM_CLAMP_RATIO * scale
        else:
            sodium_cap = _SODIUM_ABS_CAP_PER_100G * scale
        if per_ing.get("sodium_mg", 0.0) > sodium_cap:
            per_ing["sodium_mg"] = sodium_cap

        atwater = _atwater_energy(per_ing)
        predicted_energy = per_ing.get("energy_kcal", 0.0)
        if atwater > 5.0 and predicted_energy > 0.0:
            ratio = predicted_energy / atwater
            if ratio > 1.4 or ratio < 0.6:
                per_ing["energy_kcal"] = atwater

        for col, val in per_ing.items():
            total_nutrients[col] = total_nutrients.get(col, 0.0) + val

        absorption = food_index.get(ing.ingredient_id, {}).get("absorption_factors") or {}
        for col, val in per_ing.items():
            factor = absorption.get(col, 1.0)
            total_absorbed[col] = total_absorbed.get(col, 0.0) + val * factor

        ingredient_confs.append(ing_conf)

    if artifact is None or all(r.unknown for r in ingredient_results):
        return PredictionResponse(
            cooking_type=meal.cooking_type,
            cooking_duration_min=meal.cooking_duration_min,
            ingredients=ingredient_results,
            result=None,
        )

    avg_conf = sum(ingredient_confs) / len(ingredient_confs)
    confidence = round(max(0.0, min(1.0, avg_conf * method_conf)), 4)

    nutrient_fields = {col: total_nutrients.get(col) for col in artifact["target_columns"]}

    absorbed_fields = {col: total_absorbed.get(col) for col in artifact["target_columns"]}
    has_absorption = any(
        food_index.get(ing.ingredient_id, {}).get("absorption_factors")
        for ing in meal.ingredients
    )
    absorbed_result = NutritionalResult(**absorbed_fields, confidence=confidence) if has_absorption else None

    return PredictionResponse(
        cooking_type=meal.cooking_type,
        cooking_duration_min=meal.cooking_duration_min,
        ingredients=ingredient_results,
        result=NutritionalResult(**nutrient_fields, confidence=confidence),
        absorbed_result=absorbed_result,
    )
