from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RawProfile(BaseModel):
    """Raw nutritional profile per 100g supplied by the API from the product_search index (Elasticsearch)."""

    energy_kcal: Optional[float] = None
    water_g: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    sugars_g: Optional[float] = None
    fiber_g: Optional[float] = None
    calcium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    magnesium_mg: Optional[float] = None
    sodium_mg: Optional[float] = None
    vitamin_c_mg: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    monounsaturated_fat_g: Optional[float] = None
    polyunsaturated_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    phosphorus_mg: Optional[float] = None
    zinc_mg: Optional[float] = None
    vitamin_a_ug: Optional[float] = None
    vitamin_d_ug: Optional[float] = None
    vitamin_b6_mg: Optional[float] = None
    vitamin_b12_ug: Optional[float] = None
    niacin_mg: Optional[float] = None
    food_group: str = "other"
    is_plant: bool = False

    def as_nutrient_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude={"food_group", "is_plant"})


class IngredientInput(BaseModel):
    ingredient_id: str
    weight_g: float = Field(gt=0)
    measured_state: str = "raw"  # "raw" = weight before cooking; "cooked" = weight after cooking
    raw_profile: Optional[RawProfile] = None


class MealInput(BaseModel):
    cooking_type: str
    cooking_duration_min: Optional[float] = None
    ingredients: List[IngredientInput] = Field(min_length=1)


class IngredientResult(BaseModel):
    ingredient_id: str
    weight_g: float
    unknown: bool = False


class NutritionalResult(BaseModel):
    energy_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fiber_g: Optional[float] = None
    water_g: Optional[float] = None
    sugars_g: Optional[float] = None
    calcium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    magnesium_mg: Optional[float] = None
    sodium_mg: Optional[float] = None
    vitamin_c_mg: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    monounsaturated_fat_g: Optional[float] = None
    polyunsaturated_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    phosphorus_mg: Optional[float] = None
    zinc_mg: Optional[float] = None
    vitamin_a_ug: Optional[float] = None
    vitamin_d_ug: Optional[float] = None
    vitamin_b6_mg: Optional[float] = None
    vitamin_b12_ug: Optional[float] = None
    niacin_mg: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence coefficient")


class PredictionResponse(BaseModel):
    cooking_type: str
    cooking_duration_min: Optional[float]
    ingredients: List[IngredientResult]
    result: Optional[NutritionalResult]
    absorbed_result: Optional[NutritionalResult] = None
