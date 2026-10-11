"""Einheiten des Kerns an Handrechnungen und Mini-Instanzen (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import hdb_algorithm as A
import hdb_scenario as S


def test_matern_kernel_hand_values():
    x = np.array([[0.0]])
    # r = 0: k = Varianz
    assert A.matern52(x, x, np.array([1.0]), 2.0)[0, 0] == pytest.approx(2.0)
    # r = 1, Längenmaß 1: (1 + √5 + 5/3) exp(−√5)
    k = A.matern52(x, np.array([[1.0]]), np.array([1.0]), 1.0)[0, 0]
    assert k == pytest.approx((1 + np.sqrt(5) + 5 / 3) * np.exp(-np.sqrt(5)))
    # Längenmaß 2 und Abstand 2 geben dasselbe r = 1
    k2 = A.matern52(x, np.array([[2.0]]), np.array([2.0]), 1.0)[0, 0]
    assert k2 == pytest.approx(k)


def test_large_lengthscale_hides_an_input():
    a = np.array([[0.2, 0.1]])
    b = np.array([[0.2, 0.9]])
    ls = np.array([1.0, 1e6])
    assert A.matern52(a, b, ls, 1.0)[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_expected_improvement_hand_values():
    # mu = best, sd = 1: EI = φ(0) = 0,3989...
    assert A.expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)[0] == pytest.approx(0.3989422804)
    # sicher besser: sd → 0, EI = best − mu
    assert A.expected_improvement(np.array([-1.0]), np.array([1e-12]), 0.0)[0] == pytest.approx(1.0)
    # sicher schlechter: EI = 0
    assert A.expected_improvement(np.array([1.0]), np.array([1e-12]), 0.0)[0] == pytest.approx(0.0, abs=1e-12)


def test_gp_interpolates_noise_free_data(mini):
    X, y = mini
    gp = A.GaussianProcess().fit(X, y)
    mu, sd = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3)
    assert np.all(sd < 1e-2)


def test_ard_learns_that_the_second_input_is_irrelevant(mini):
    rng = np.random.default_rng(3)
    X = rng.random((25, 2))
    y = np.sin(4 * X[:, 0])
    gp = A.GaussianProcess().fit(X, y)
    assert gp.lengthscales[1] > 3 * gp.lengthscales[0]


def test_isotropic_has_one_lengthscale():
    rng = np.random.default_rng(1)
    X = rng.random((12, 4))
    gp = A.GaussianProcess(isotropic=True).fit(X, X[:, 0])
    assert len(set(np.round(gp.lengthscales, 12))) == 1


def test_trust_region_rules():
    # drei Erfolge in Folge: Weiten um 1,6
    h, s, f = 0.4, 0, 0
    for _ in range(3):
        h, s, f = A.update_trust_region(h, s, f, True, 10)
    assert h == pytest.approx(0.64) and s == 0 and f == 0
    # Obergrenze 0,8
    h, s, f = 0.6, 2, 0
    h, s, f = A.update_trust_region(h, s, f, True, 10)
    assert h == pytest.approx(0.8)
    # max(4, d/2) Misserfolge: Halbieren (d = 12 -> 6)
    h, s, f = 0.4, 0, 0
    for _ in range(5):
        h, s, f = A.update_trust_region(h, s, f, False, 12)
    assert h == pytest.approx(0.4) and f == 5
    h, s, f = A.update_trust_region(h, s, f, False, 12)
    assert h == pytest.approx(0.2) and f == 0
    # Untergrenze 0,02
    h, s, f = 0.03, 0, 3
    h, s, f = A.update_trust_region(h, s, f, False, 4)
    assert h == pytest.approx(0.02)


def test_candidates_stay_in_the_cube_and_in_the_box():
    rng = np.random.default_rng(0)
    X = rng.random((6, 5))
    y = rng.random(6)
    c = A.candidate_set(X, y, rng)
    assert c.min() >= 0 and c.max() <= 1
    centre = np.full(5, 0.5)
    t = A.candidate_set(X, y, rng, trust=(centre, 0.1))
    assert t.min() >= 0.4 - 1e-12 and t.max() <= 0.6 + 1e-12


def test_hartmann_and_ackley_known_values():
    # Hartmann6 am bekannten Minimum
    xstar = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    assert S.hartmann6(xstar) == pytest.approx(-3.32237, abs=1e-4)
    # zusätzliche Eingaben ohne Wirkung
    assert S.hartmann6(np.concatenate([xstar, np.random.default_rng(0).random(30)])) == S.hartmann6(xstar)
    # Ackley: Minimum 0 bei x = 0,5
    assert S.ackley(np.full(7, 0.5)) == pytest.approx(0.0, abs=1e-9)
    assert S.ackley(np.full(7, 0.1)) > 1.0


def test_random_search_is_reproducible_and_bo_beats_random_on_a_bowl():
    f = lambda x: float(((x - 0.3) ** 2).sum())
    X1, y1 = A.random_search(f, 3, 30, np.random.default_rng(5))
    X2, y2 = A.random_search(f, 3, 30, np.random.default_rng(5))
    assert np.array_equal(y1, y2)
    wins = 0
    for seed in range(5):
        _, yb, _ = A.bo_search(f, 3, 20, np.random.default_rng(seed), mode="ard")
        _, yr = A.random_search(f, 3, 20, np.random.default_rng(seed))
        wins += yb.min() < yr.min()
    assert wins == 5


def test_budget_not_larger_than_start_design_does_not_crash():
    f = lambda x: float(x.sum())
    X, y, ls = A.bo_search(f, 4, 3, np.random.default_rng(0))
    assert len(y) == 3 and len(ls) == 4
