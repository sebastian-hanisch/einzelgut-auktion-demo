"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

import eza_constants as C
from eza_distributions import KINDS
from eza_learning import FORMATS, RULES
from eza_scenario import Settings


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


SETTING_SPECS = {
    "dist_select": SettingSpec("verteilung", str, C.DIST_DEFAULT, choices=KINDS),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "r_slider": SettingSpec("r", int, C.DEFAULT_R, C.R_MIN, C.R_MAX),
    "v0_slider": SettingSpec("v0", int, C.DEFAULT_V0, C.V0_MIN, C.V0_MAX),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX),
    "w_slider": SettingSpec("w", float, C.DEFAULT_W, C.W_MIN, C.W_MAX),
    "rule_select": SettingSpec("regel", str, "Hedge", choices=RULES),
    "fmt_select": SettingSpec("format", str, "Erstpreis", choices=FORMATS),
    "nl_select": SettingSpec("lernende", int, 2, choices=C.LEARN_NS),
    "rounds_slider": SettingSpec("runden", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX),
    "rate_select": SettingSpec("rate", float, C.DEFAULT_RATE, choices=C.RATE_FACTORS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"dist": "dist_select", "n": "n_slider", "r": "r_slider", "v0": "v0_slider", "rho": "rho_slider", "w": "w_slider", "rule": "rule_select", "fmt": "fmt_select",
               "n_learn": "nl_select", "rounds": "rounds_slider", "rate": "rate_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if value != value:      # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: was das Preset nicht nennt, bekommt den Standardwert (so ist das Preset danach genau das aktive)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def settings_from_state() -> Settings:
    """Die Einstellungen aus dem Session State (vor dem Zeichnen der Regler gesetzt durch Voreinstellung, Permalink oder Standard)."""
    g = st.session_state.get
    d = {k: g(k, SETTING_SPECS[k].default) for k in SETTING_SPECS}
    return Settings(dist=str(d["dist_select"]), n=int(d["n_slider"]), r=float(d["r_slider"]), v0=float(d["v0_slider"]), rho=float(d["rho_slider"]), w=float(d["w_slider"]),
                    rule=str(d["rule_select"]), fmt=str(d["fmt_select"]), n_learn=int(d["nl_select"]), rounds=int(d["rounds_slider"]), rate=float(d["rate_select"]), seed=int(d["seed_input"]))


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
