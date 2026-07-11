import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def planted_data():
    """A synthetic dataset with a planted worst subgroup (x1 > 0.6) that has
    substantially higher per-sample loss than the rest of the data."""
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    x3 = rng.uniform(0, 1, n)
    worst = x1 > 0.6  # planted worst subgroup

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    subset_labels = worst.astype(int)
    # loss is much higher inside the worst subgroup
    test_loss = rng.normal(0.1, 0.02, n) + 0.5 * worst

    # a downstream evaluation set: model predicts poorly inside the worst subset
    y_test = rng.integers(0, 2, n)
    test_preds = np.where(worst, rng.uniform(0.3, 0.7, n), y_test * 0.9 + 0.05)

    return {
        "X": X,
        "subset_labels": subset_labels,
        "test_loss": test_loss,
        "worst": worst,
        "y_test": y_test,
        "test_preds": np.clip(test_preds, 0.01, 0.99),
    }
