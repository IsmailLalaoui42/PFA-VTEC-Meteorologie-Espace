"""Common base class for forecasters and detectors."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


class BaseForecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    def save(self, path) -> None:
        if joblib is None:
            raise RuntimeError("joblib required for save/load")
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        if joblib is None:
            raise RuntimeError("joblib required for save/load")
        return joblib.load(path)
