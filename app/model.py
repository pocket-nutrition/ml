import logging
import os
from pathlib import Path

import joblib

log = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/model.joblib"))

_model = None


def load_model() -> None:
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    else:
        log.warning("No model artifact found at %s — predictions will return confidence=0", MODEL_PATH)


def get_model():
    return _model
