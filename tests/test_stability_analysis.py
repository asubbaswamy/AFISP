import numpy as np

from afisp import WorstSubsetFinder


def test_worst_subset_finder_smoke():
    rng = np.random.default_rng(0)
    n = 300
    # WorstSubsetFinder.fit fancy-indexes X[train_idxs], so pass a numpy array
    X = rng.uniform(0, 1, (n, 3))
    losses = rng.normal(0.1, 0.02, n) + 0.5 * (X[:, 0] > 0.6)

    wsf = WorstSubsetFinder(subset_fractions=[0.2, 0.5, 1.0], cv=3)
    r_hats = wsf.fit(X, losses)
    assert r_hats.shape == (3,)

    masks = wsf.subset_masks()
    assert len(masks) == 3
    assert masks[0].shape == (n,)

    cis = wsf.confidence_intervals()
    assert cis.shape == (3, 2)
    # lower bound below upper bound
    assert np.all(cis[:, 0] <= cis[:, 1])

    idx, eff = wsf.find_max_effect_size()
    assert 0 <= idx < 3
