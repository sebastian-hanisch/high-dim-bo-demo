"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import hdb_constants as C
import hdb_evaluation as ev
import hdb_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["seed"] <= C.SEED_MAX
        assert preset["dim"] in C.LIVE_DIMENSIONS["hartmann"]
        assert preset["method"] in C.LIVE_METHODS


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert ev.Settings(dim=p["dim"], n=p["n"], seed=p["seed"]) == ev.Settings()
    assert p["method"] == P.SETTING_SPECS["method_radio"].default


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_snap_to_choice():
    assert P.snap_to_choice(11, (6, 8, 10, 12), 10) in (10, 12)
    assert P.snap_to_choice(7, (6, 8, 10, 12), 10) in (6, 8)
    assert P.snap_to_choice("nope", C.LIVE_METHODS, "bo_ard") == "bo_ard"
