import logging
import os
from pathlib import Path

import joblib

log = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/model.joblib"))

_REQUIRED_KEYS = ("model", "scaler", "feature_columns", "target_columns")

_model = None


def load_model() -> None:
    global _model
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        missing = [k for k in _REQUIRED_KEYS if k not in artifact]
        if missing:
            # Fail at startup: a partial artifact would silently predict the wrong nutrient set.
            raise ValueError(f"Model artifact {MODEL_PATH} lacks required keys: {missing}")
        _model = artifact
        log.info("Model loaded from %s", MODEL_PATH)
    else:
        log.warning("No model artifact found at %s — predictions will return confidence=0", MODEL_PATH)


def get_model():
    return _model
