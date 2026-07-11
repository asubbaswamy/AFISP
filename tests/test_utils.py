import numpy as np
import pytest

from afisp import utils


def test_clip_predictions():
    preds = np.array([0.0, 0.5, 1.0])
    out = utils.clip_predictions(preds, upper_bound=0.9, lower_bound=0.1)
    assert out.min() >= 0.1 and out.max() <= 0.9
    assert out[1] == 0.5


def test_brier():
    y = np.array([1, 0, 1])
    p = np.array([0.8, 0.2, 0.6])
    np.testing.assert_allclose(utils.brier(y, p), (y - p) ** 2)


def test_zero_one_loss():
    assert np.array_equal(
        utils.zero_one_loss(np.array([1, 0]), np.array([1, 1])),
        np.array([0.0, 1.0]),
    )


def test_mse():
    np.testing.assert_allclose(
        utils.mse(np.array([1.0, 2.0]), np.array([1.5, 2.0])),
        np.array([0.25, 0.0]),
    )


def test_cross_entropy_positive():
    ce = utils.cross_entropy(np.array([1, 0]), np.array([0.9, 0.1]))
    assert np.all(ce > 0)


def test_cohens_d_sign():
    a = np.array([1.0, 1.1, 0.9, 1.05])
    b = np.array([0.0, 0.1, -0.1, 0.05])
    assert utils.cohens_d(a, b) > 0


def test_bootstrap_ci_shape():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    p = rng.uniform(0, 1, 100)
    m, l, u = utils.bootstrap_ci(
        y, p, n_bootstrap=50, loss=lambda yt, pp: np.mean((yt - pp) ** 2)
    )
    assert l <= m <= u


def test_torch_surrogate_optional():
    pytest.importorskip("torch")
    out = utils.torch_roc_auc_surrogate(np.array([1, 0, 1]), np.array([0.8, 0.2, 0.7]))
    assert out is not None
