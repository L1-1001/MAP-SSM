import numpy as np

from utils.metrics import mae, mse


def test_metrics() -> None:
    prediction = np.array([1.0, 3.0])
    target = np.array([2.0, 1.0])
    assert mse(prediction, target) == 2.5
    assert mae(prediction, target) == 1.5
