"""Artifact loading: a malformed artifact must fail at startup, not degrade silently."""
from __future__ import annotations

import joblib
import pytest

from app import model as model_module

_VALID = {"model": object(), "scaler": object(), "feature_columns": ["energy_kcal"], "target_columns": ["energy_kcal"]}


def _load(tmp_path, monkeypatch, artifact):
    path = tmp_path / "model_v2.joblib"
    joblib.dump(artifact, path)
    monkeypatch.setattr(model_module, "MODEL_PATH", path)
    monkeypatch.setattr(model_module, "_model", None)
    model_module.load_model()
    return model_module.get_model()


def test_valid_artifact_is_loaded(tmp_path, monkeypatch):
    assert _load(tmp_path, monkeypatch, _VALID)["target_columns"] == ["energy_kcal"]


@pytest.mark.parametrize("missing", ["model", "scaler", "feature_columns", "target_columns"])
def test_artifact_missing_a_required_key_fails_at_load(tmp_path, monkeypatch, missing):
    artifact = {k: v for k, v in _VALID.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        _load(tmp_path, monkeypatch, artifact)


def test_absent_artifact_leaves_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(model_module, "_model", None)
    model_module.load_model()
    assert model_module.get_model() is None
