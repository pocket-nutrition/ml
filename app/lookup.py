"""Ingredient and cooking-method resolution helpers."""
from __future__ import annotations


def is_known_ingredient(ingredient_id: str, artifact: dict) -> bool:
    """Return True if ingredient_id has a known raw profile in the food index."""
    return ingredient_id in artifact.get("food_index", {})


def resolve_cooking_method(
    method: str,
    le_classes: list[str],
) -> tuple[str, bool]:
    """Resolve an incoming cooking method to one known by the label encoder.

    Returns (canonical_method, is_known). Falls back to "cooked" when unknown.
    """
    normalised = method.strip().lower()
    if normalised in le_classes:
        return normalised, True
    return "cooked", False
