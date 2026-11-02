"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Knopf (auch jeden ▶️-Knopf einzeln), fährt jeden Regler an seine
Grenzen und durch alle Auswahlwerte, die Kombination aller Minima und aller Maxima, und prüft, dass kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere
st.plotly_chart ohne eindeutiges key=) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen (min == max)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import eza_constants as C
import eza_learning as LE
import eza_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _edge_values(spec):
    return list(spec.choices) if spec.choices is not None else [spec.lo, spec.hi]


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title[0].value and len(at.tabs) == 5


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert "▶️ Lernlauf starten" in labels and "PDF erstellen" in labels
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_learning_button_shows_a_result_and_a_changed_setting_marks_it_stale():
    at = _fresh_app()
    at.button(key="learn_run").click().run(timeout=TIMEOUT)
    assert not at.exception and not [i for i in at.info if "Noch kein Lauf" in i.value]
    assert any("letzter Gebotsfaktor" in w.value for w in at.warning)
    _regler(at, "rounds_slider").set_value(2000).run(timeout=TIMEOUT)
    assert any("neu starten" in i.value for i in at.info)


def test_the_pdf_button_makes_a_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["pdf_bytes"][:4] == b"%PDF"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_every_regler_at_its_edges_does_not_raise(state_key):
    for value in _edge_values(P.SETTING_SPECS[state_key]):
        at = _fresh_app()
        _regler(at, state_key).set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, value, [str(e) for e in at.exception])


def _all_edges(pick):
    at = _fresh_app()
    for state_key, spec in P.SETTING_SPECS.items():
        _regler(at, state_key).set_value(pick(_edge_values(spec)))
    at.run(timeout=TIMEOUT)
    return at


def test_all_minima_together_and_all_maxima_together_do_not_raise():
    lows = _all_edges(lambda edges: edges[0])
    assert not lows.exception, [str(e) for e in lows.exception]
    highs = _all_edges(lambda edges: edges[-1])
    assert not highs.exception, [str(e) for e in highs.exception]
    highs.button(key="learn_run").click().run(timeout=TIMEOUT)          # Maximum: Q-Lernen (letzte Regel), Zweitpreis, 4 Bieter, 30.000 Runden, Rate 30
    assert not highs.exception, [str(e) for e in highs.exception]


@pytest.mark.parametrize("rule", LE.RULES)
@pytest.mark.parametrize("fmt", LE.FORMATS)
def test_learning_run_at_the_maximum_for_every_rule_and_format(rule, fmt):
    at = _fresh_app()
    _regler(at, "rule_select").set_value(rule)
    _regler(at, "fmt_select").set_value(fmt)
    _regler(at, "nl_select").set_value(4)
    _regler(at, "rounds_slider").set_value(C.ROUNDS_MAX)
    at.run(timeout=TIMEOUT)
    at.button(key="learn_run").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("letzter Gebotsfaktor" in w.value for w in at.warning)


def test_a_permalink_with_wild_values_is_limited_to_the_bounds():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "9999"
    at.query_params["r"] = "-5"
    at.query_params["rho"] = "nan"
    at.query_params["verteilung"] = "unsinn"
    at.query_params["rate"] = "7"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["r_slider"] == C.R_MIN
    assert at.session_state["rho_slider"] == C.DEFAULT_RHO and at.session_state["dist_select"] == C.DIST_DEFAULT and at.session_state["rate_select"] == C.DEFAULT_RATE


def test_the_tab_for_the_learners_is_marked_yellow_and_names_the_negative_finding():
    at = _fresh_app()
    texts = " ".join(w.value for w in at.warning)
    assert "Gelb markiert" in texts and "mit Standardrate landet kein Verfahren im Gleichgewicht" in texts
