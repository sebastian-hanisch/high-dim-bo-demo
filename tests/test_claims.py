"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (vorgerechnete Studie: 10 feste Läufe je Zelle, Budget 100)."""

import pytest

import hdb_evaluation as ev


def _med(p, d, m, cp=100):
    return ev.table(p, d)[m][cp]["median"]


def test_hartmann_random_is_flat_and_far_from_the_optimum():
    for d in (6, 12, 24, 48):
        assert 1.2 < _med("hartmann", d, "random") < 1.7


def test_ard_bo_reaches_the_optimum_at_6_and_stays_close_up_to_48_inputs():
    assert _med("hartmann", 6, "bo_ard") == pytest.approx(0.007, abs=0.01)
    assert _med("hartmann", 12, "bo_ard") == pytest.approx(0.12, abs=0.03)
    assert _med("hartmann", 24, "bo_ard") == pytest.approx(0.12, abs=0.03)
    assert _med("hartmann", 48, "bo_ard") == pytest.approx(0.13, abs=0.03)
    assert _med("hartmann", 48, "cma") == pytest.approx(0.56, abs=0.05)
    assert _med("hartmann", 48, "random") == pytest.approx(1.27, abs=0.05)


def test_bo_beats_random_and_cma_in_ten_of_ten_runs_on_hartmann():
    for d in (6, 12, 24, 48):
        s = ev.paired_stats("hartmann", d, "bo_ard", "random", 100)
        assert (s["better"], s["worse"]) == (10, 0) and s["p"] == pytest.approx(0.00195, abs=1e-4)
        c = ev.paired_stats("hartmann", d, "cma", "bo_ard", 100)
        assert (c["better"], c["worse"]) == (0, 10)


def test_ard_hides_the_irrelevant_inputs_at_the_upper_bound():
    for d in (12, 24, 48):
        s = ev.length_scales_summary("hartmann", d)
        assert s["inactive"] == pytest.approx(10.0, abs=0.05)
        assert 1.5 < s["active"] < 2.6


def test_ackley_gain_shrinks_with_dimension_but_stays_at_24():
    assert _med("ackley", 24, "bo_iso") == pytest.approx(4.84, abs=0.1) and _med("ackley", 24, "random") == pytest.approx(7.52, abs=0.1)
    ratios = [_med("ackley", d, "bo_iso") / _med("ackley", d, "random") for d in (2, 6, 12, 24)]
    assert ratios[0] < ratios[1] < ratios[2] < ratios[3]
    s = ev.paired_stats("ackley", 24, "bo_ard", "random", 100)
    assert (s["better"], s["worse"]) == (9, 1) and s["p"] == pytest.approx(0.0215, abs=0.002)


def test_no_variant_is_provably_different_from_ard_on_ackley_from_6_inputs():
    s2 = ev.paired_stats("ackley", 2, "bo_tr", "bo_ard", 100)
    assert (s2["better"], s2["worse"]) == (9, 1)                              # nur bei 2 Eingaben: Vertrauensbereich besser
    for d in (6, 12, 24):
        for k in ("bo_iso", "bo_scaled", "bo_tr"):
            assert ev.paired_stats("ackley", d, k, "bo_ard", 100)["p"] > 0.05, (d, k)


def test_trust_region_is_better_at_6_and_worse_at_48_inputs_on_hartmann():
    a = ev.paired_stats("hartmann", 6, "bo_tr", "bo_ard", 100)
    b = ev.paired_stats("hartmann", 48, "bo_tr", "bo_ard", 100)
    assert (a["better"], a["worse"]) == (9, 1) and (b["better"], b["worse"]) == (1, 9)


def test_ard_fit_is_expensive_in_high_dimension_and_isotropic_is_not():
    assert ev.table("hartmann", 48)["bo_ard"]["seconds"] > 60
    assert ev.table("hartmann", 48)["bo_iso"]["seconds"] < 30
