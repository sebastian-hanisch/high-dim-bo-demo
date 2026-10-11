"""Orakel-Tests: Gauß-Prozess und Vorzeichentest gegen unabhängige Rechenwege (scikit-learn, Handformel, scipy)."""

import numpy as np
import pytest

import hdb_algorithm as A
import hdb_constants as C
import hdb_evaluation as ev


def test_gp_posterior_equals_scikit_learn_for_fixed_hyperparameters():
    sk = pytest.importorskip("sklearn.gaussian_process")
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    rng = np.random.default_rng(11)
    for trial in range(8):
        d = 2 + trial % 3
        X = rng.random((10, d))
        y = np.sin(3 * X[:, 0]) + 0.3 * X[:, 1]
        ls = rng.uniform(0.3, 1.5, d)
        var = 1.7
        gp = A.GaussianProcess()
        gp.X = X
        gp.y_mean, gp.y_std = float(y.mean()), float(y.std())
        gp.lengthscales, gp.variance = ls, var
        gp = A._refresh(gp, X, y)
        gp.y_mean, gp.y_std = float(y.mean()), float(y.std())
        Xs = rng.random((6, d))
        mu, sd = gp.predict(Xs)
        kern = ConstantKernel(var, "fixed") * Matern(length_scale=ls, nu=2.5, length_scale_bounds="fixed")
        z = (y - y.mean()) / y.std()
        ref = sk.GaussianProcessRegressor(kern, alpha=1e-6 + 1e-8, optimizer=None).fit(X, z)
        m_ref, s_ref = ref.predict(Xs, return_std=True)
        assert np.allclose(mu, m_ref * y.std() + y.mean(), atol=1e-6), trial
        assert np.allclose(sd, s_ref * y.std(), atol=1e-4), trial


def test_expected_improvement_equals_numerical_integration():
    from scipy.integrate import quad
    from scipy.stats import norm
    for mu, sd, best in [(0.3, 0.7, 0.0), (-0.5, 0.2, 0.1), (1.0, 2.0, 0.5)]:
        ref = quad(lambda f: max(best - f, 0.0) * norm.pdf(f, mu, sd), mu - 12 * sd, mu + 12 * sd)[0]
        assert A.expected_improvement(np.array([mu]), np.array([sd]), best)[0] == pytest.approx(ref, abs=1e-8)


def test_sign_test_matches_the_exact_binomial_by_hand():
    # 9 von 10: p = 2 * (C(10,9) + C(10,10)) / 2^10 = 22/1024
    from math import comb
    p_hand = 2 * sum(comb(10, k) for k in (9, 10)) / 2 ** 10
    assert p_hand == pytest.approx(22 / 1024)
    from scipy.stats import binomtest
    assert binomtest(9, 10, 0.5).pvalue == pytest.approx(p_hand)


def test_study_paired_stats_recompute_from_the_raw_runs():
    raw = ev.study()["runs"]
    for key in ("hartmann_12", "ackley_6"):
        a = np.array(raw[key]["bo_ard"]["best"])[:, 99]
        b = np.array(raw[key]["random"]["best"])[:, 99]
        s = ev.paired_stats(key.split("_")[0], int(key.split("_")[1]), "bo_ard", "random", 100)
        assert s["better"] == int((a < b - 1e-9).sum()) and s["worse"] == int((a > b + 1e-9).sum())
        assert s["better"] + s["worse"] + s["equal"] == 10


def test_study_is_complete_and_monotone():
    raw = ev.study()["runs"]
    for p in C.PROBLEMS:
        for d in C.DIMENSIONS[p]:
            rec = raw[ev.study_key(p, d)]
            assert set(rec) == set(C.METHODS)
            for m, r in rec.items():
                arr = np.array(r["best"])
                assert arr.shape == (10, C.STUDY_BUDGET), (p, d, m)
                assert np.all(np.diff(arr, axis=1) <= 1e-12)                  # bester Wert nie schlechter
                assert np.all(arr - C.GLOBAL_MIN[p] >= -1e-6)                 # nie unter dem Minimum
