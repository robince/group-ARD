import numpy as np
import pytest
from numpy.testing import assert_allclose

from groupard import GroupARDRegression


def make_synthetic(n_samples=200, n_features=10, n_informative=3, noise=0.1, rng=0):
    rng = np.random.RandomState(rng)
    X = rng.randn(n_samples, n_features)
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.randn(n_informative)
    y = X @ coef + noise * rng.randn(n_samples)
    return X, y, coef


def test_attributes_and_shapes_with_list_groups():
    X, y, _ = make_synthetic(n_samples=100, n_features=8, n_informative=2, rng=42)
    groups_list = [10, 10, 10, 20, 20, 30, 30, 30]  # list input, non-contiguous labels

    est = GroupARDRegression(prior='GroupARD', groups=groups_list, tol=1e-6, max_iter=200)
    est.fit(X, y)

    n_features = X.shape[1]
    assert est.coef_.shape == (n_features,)
    assert isinstance(est.intercept_, float)
    assert est.sigma_.shape == (n_features, n_features)
    assert est.lambda_.shape == (n_features,)
    assert est.groups_.shape == (n_features,)
    assert est.active_mask_.shape == (n_features,)
    # groups_ are normalized 0..G-1
    uniq = np.unique(est.groups_)
    assert uniq.min() == 0
    assert uniq.max() == len(uniq) - 1


def test_equivalence_ard_groupard_per_feature():
    X, y, _ = make_synthetic(n_samples=120, n_features=12, n_informative=4, rng=123)
    n_features = X.shape[1]

    ard = GroupARDRegression(prior='ARD', tol=1e-6, max_iter=400, alpha_init=1.0)
    g_ard = GroupARDRegression(prior='GroupARD', groups=np.arange(n_features), tol=1e-6, max_iter=400, alpha_init=1.0)

    ard.fit(X, y)
    g_ard.fit(X, y)

    assert_allclose(ard.coef_, g_ard.coef_, rtol=1e-3, atol=1e-3)


def test_equivalence_ridge_groupard_single_group():
    X, y, _ = make_synthetic(n_samples=150, n_features=10, n_informative=5, rng=321)
    n_features = X.shape[1]

    ridge = GroupARDRegression(prior='Ridge', tol=1e-6, max_iter=400, alpha_init=1.0)
    g_ridge = GroupARDRegression(prior='GroupARD', groups=np.zeros(n_features, dtype=int), tol=1e-6, max_iter=400, alpha_init=1.0)

    ridge.fit(X, y)
    g_ridge.fit(X, y)

    assert_allclose(ridge.coef_, g_ridge.coef_, rtol=1e-3, atol=1e-3)


def test_sample_weight_ones_matches_unweighted():
    X, y, _ = make_synthetic(n_samples=80, n_features=7, n_informative=3, rng=7)
    groups = np.arange(X.shape[1])

    est1 = GroupARDRegression(prior='GroupARD', groups=groups, tol=1e-6, max_iter=300)
    est2 = GroupARDRegression(prior='GroupARD', groups=groups, tol=1e-6, max_iter=300)

    est1.fit(X, y)
    est2.fit(X, y, sample_weight=np.ones(X.shape[0]))

    assert_allclose(est1.coef_, est2.coef_, rtol=1e-8, atol=1e-8)
    assert_allclose(est1.intercept_, est2.intercept_, rtol=1e-8, atol=1e-8)


def test_predict_shape():
    X, y, _ = make_synthetic(n_samples=60, n_features=6, n_informative=2, rng=11)
    groups = [0, 0, 1, 1, 2, 2]

    est = GroupARDRegression(prior='GroupARD', groups=groups, tol=1e-6, max_iter=200)
    est.fit(X, y)
    y_pred = est.predict(X)

    assert y_pred.shape == (X.shape[0],)
    assert np.isfinite(y_pred).all()


def test_smoke_fit_predict():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 5)
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.randn(20)
    groups = [0, 0, 1, 1, 2]

    est = GroupARDRegression(prior='GroupARD', groups=groups, max_iter=50, tol=1e-4)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y_pred.shape == (20,)
    assert np.isfinite(y_pred).all()


def test_extend_groups_extend_behavior():
    # Provide fewer group labels than features and extend
    rng = np.random.RandomState(1)
    X = rng.randn(40, 6)
    y = X[:, 0] + 0.2 * rng.randn(40)
    groups_short = [10, 10, 20]  # list, non-contiguous, shorter than n_features

    est = GroupARDRegression(prior='GroupARD', groups=groups_short, extend_groups=True, max_iter=50, tol=1e-4)
    est.fit(X, y)

    # Expect a single new group assigned to all extra features
    assert est.groups_.shape == (6,)
    first_part = est.groups_[:3]
    extra_part = est.groups_[3:]
    assert np.unique(first_part).size == 2  # came from labels 10 and 20
    assert np.unique(extra_part).size == 1  # all extras share the same new group
    # New group id should not be among the first part's groups
    assert extra_part[0] not in set(first_part.tolist())


def test_extend_groups_truncate_behavior():
    # Provide more group labels than features and truncate
    rng = np.random.RandomState(2)
    X = rng.randn(30, 4)
    y = X[:, 0] - X[:, 1] + 0.1 * rng.randn(30)
    groups_long = [0, 0, 1, 1, 2, 2]

    est = GroupARDRegression(prior='GroupARD', groups=groups_long, extend_groups=True, max_iter=50, tol=1e-4)
    est.fit(X, y)

    assert est.groups_.shape == (4,)
    assert_allclose(est.groups_, np.array([0, 0, 1, 1]))


def test_groups_length_mismatch_raises():
    rng = np.random.RandomState(3)
    X = rng.randn(20, 5)
    y = X[:, 0] + 0.1 * rng.randn(20)
    groups_bad = [0, 0, 1]  # wrong length and no extend

    est = GroupARDRegression(prior='GroupARD', groups=groups_bad, extend_groups=False, max_iter=10)
    with pytest.raises(ValueError):
        est.fit(X, y)
