from app.lookup import is_known_ingredient, resolve_cooking_method

LE_CLASSES = ["boiled", "cooked", "fried", "grilled", "roasted", "steamed"]


def test_is_known_ingredient_found():
    artifact = {"food_index": {"poulet_filet": {}, "riz": {}}}
    assert is_known_ingredient("poulet_filet", artifact) is True


def test_is_known_ingredient_not_found():
    artifact = {"food_index": {"poulet_filet": {}, "riz": {}}}
    assert is_known_ingredient("fictional_ingredient", artifact) is False


def test_is_known_ingredient_empty_artifact():
    assert is_known_ingredient("poulet_filet", {}) is False


def test_is_known_ingredient_missing_key():
    assert is_known_ingredient("riz", {"model": "something"}) is False


def test_resolve_known_cooking_method():
    method, known = resolve_cooking_method("grilled", LE_CLASSES)
    assert method == "grilled"
    assert known is True


def test_resolve_known_cooking_method_case_insensitive():
    method, known = resolve_cooking_method("Grilled", LE_CLASSES)
    assert method == "grilled"
    assert known is True


def test_resolve_unknown_cooking_method_fallback():
    method, known = resolve_cooking_method("microwaved", LE_CLASSES)
    assert method == "cooked"
    assert known is False


def test_resolve_cooking_method_always_returns_known_class():
    method, _ = resolve_cooking_method("sous vide", LE_CLASSES)
    assert method in LE_CLASSES
