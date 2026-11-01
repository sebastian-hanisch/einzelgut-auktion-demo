"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, Regler-Zuordnung, Permalink-Begrenzung."""

import eza_constants as C
import eza_presets as P
from eza_scenario import Settings


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_first_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    defaults = {key: P.SETTING_SPECS[state_key].default for key, state_key in P.PRESET_KEYS.items()}
    assert {**defaults, **first} == defaults


def test_presets_differ_from_each_other():
    defaults = {key: P.SETTING_SPECS[state_key].default for key, state_key in P.PRESET_KEYS.items()}
    full = [tuple(sorted({**defaults, **p}.items())) for p in C.PRESETS.values()]
    assert len(set(full)) == len(full)


def test_setting_specs_cover_every_preset_key_and_every_settings_field():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len(P.SETTING_SPECS) == len(Settings.__dataclass_fields__)


def test_defaults_are_inside_their_own_bounds():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_slowest_live_learning_setting_is_within_the_documented_range():
    assert C.ROUNDS_MAX == 30000 and max(C.LEARN_NS) == 4 and C.DEFAULT_ROUNDS in range(C.ROUNDS_MIN, C.ROUNDS_MAX + 1)
