"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12): Schwellen und jede Zahl im Text an handgesetzten Werten; je Preset die erwartete Verdikt-Zeile (§11a Punkt 7)."""

import numpy as np
import pytest

import eza_constants as C
import eza_evaluation as V
from eza_scenario import Settings

ARR = np.array([0.0, 1.0])


def main_view(**kw):
    base = dict(kind="gleich", n=2, r=50.0, v0=20.0, spa=41.667, spa_virtual=41.667, fpa=41.667, p_sale=0.75, utility=46.667, utility_plain=37.067, r_star=60.0, u_star=47.733,
                r_grid=60.0, u_grid=47.733, loss=3.0, efficient=40.0, vs=ARR, bs=ARR)
    base.update(kw)
    return V.MainView(**base)


def test_main_verdict_names_utility_plain_and_best():
    kind, text = V.verdict_main(main_view())
    assert kind == "info"                                        # 46,67 liegt mehr als 1 % unter 47,73
    for needle in ("50 Euro", "41,67", "46,67", "75,0 %", "37,07", "60,0", "47,73", "Reserve aus φ(r) = Eigenwert"):
        assert needle in text, needle
    assert V.verdict_main(main_view(r=60.0, utility=47.733))[0] == "success"


def test_main_verdict_for_irregular_distribution_and_single_bidder():
    kind, text = V.verdict_main(main_view(kind="mix", r_star=None, u_star=None, r_grid=90.0, u_grid=5.0, utility=4.0, n=2))
    assert "unregulär" in text and "90,0" in text and kind == "info"
    kind, text = V.verdict_main(main_view(n=1))
    assert "keinen Wettbewerb" in text


def equivalence_view(**kw):
    base = dict(kind="gleich", n=2, values=(20.0, 50.0, 80.0), first_bids=(10.0, 25.0, 40.0), allpay_bids=(2.0, 12.5, 32.0), rev_second=33.3333, rev_second_virtual=33.3333, rev_first=33.3333,
                rev_allpay=33.3334, gap_first=1e-9, gap_allpay=0.0, sim={}, vs=ARR, bs=ARR, vs_allpay=ARR, bs_allpay=ARR)
    base.update(kw)
    return V.EquivalenceView(**base)


def test_equivalence_verdict_thresholds_and_numbers():
    kind, text = V.verdict_equivalence(equivalence_view())
    assert kind == "success" and "Erlösäquivalenz" in text and "33,33" in text and "beim Wert 50 Euro bietet er 25,00 Euro (50,0 % darunter)" in text
    kind, text = V.verdict_equivalence(equivalence_view(rev_first=36.0))
    assert kind == "warning" and "Erlösäquivalenz" not in text and "2,6667" in text           # größte Abweichung 36,0 − 33,3333
    assert V.verdict_equivalence(equivalence_view(n=1))[0] == "info"


def test_risk_verdict_branches():
    neutral = V.RiskView(n=2, rho=1.0, factor=0.5, rev_first=33.33, rev_second=33.33, gap=0.0, sim=(33.3, 0.1))
    assert V.verdict_risk(neutral)[0] == "success" and "Erlösäquivalenz gilt" in V.verdict_risk(neutral)[1]
    averse = V.RiskView(n=2, rho=0.5, factor=2 / 3, rev_first=44.44, rev_second=33.33, gap=0.0, sim=(44.4, 0.1))
    kind, text = V.verdict_risk(averse)
    assert kind == "warning" and "0,667" in text and "1,3333" in text and "bricht" in text
    assert V.verdict_risk(V.RiskView(n=1, rho=0.5, factor=0.0, rev_first=0.0, rev_second=0.0, gap=0.0, sim=(0.0, 0.0)))[0] == "info"


def reserve_view(**kw):
    base = dict(kind="gleich", n=2, v0=20.0, rs=ARR, us=ARR, r_grid=60.0, u_grid=47.733, r_star=60.0, u_star=47.733, u_plain=37.067, p_sale_star=0.64, loss_star=4.46, loss_star_pct=11.16,
                first_at_reserve=41.0, second_at_reserve=41.0, bid_at_reserve=50.0, gap_first=0.0, r=50.0)
    base.update(kw)
    return V.ReserveView(**base)


def test_reserve_verdict_numbers_and_irregular_warning():
    kind, text = V.verdict_reserve(reserve_view())
    assert kind == "success"
    for needle in ("60,0 Euro", "47,73", "37,07", "10,67", "28,77 %", "36,0 %", "4,46", "11,16"):
        assert needle in text, needle
    kind, text = V.verdict_reserve(reserve_view(r_star=None, u_star=None))
    assert kind == "warning" and "unregulär" in text


def test_bulow_verdict_branches():
    row = dict(n=2, best=41.667, plain_next=50.0, ratio=1.2, holds=True, extra=1, p_star=0.5)
    kind, text = V.verdict_bulow("gleich", 0.0, row)
    assert kind == "success" and "41,67" in text and "50,00" in text and "1,200" in text and "50,0 %" in text
    bad = dict(n=1, best=32.33, plain_next=31.23, ratio=0.966, holds=False, extra=2, p_star=1.1)
    kind, text = V.verdict_bulow("exp", 20.0, bad)
    assert kind == "warning" and "Voraussetzung „Bieterwert ≥ Eigenwert“ verletzt" in text and "2 zusätzliche Bieter" in text
    kind, text = V.verdict_bulow("mix", 0.0, dict(bad, extra=None))
    assert "unregulär" in text and "mehr als 200 zusätzliche Bieter" in text


def test_bulow_flat_verdict_counts_rows():
    rows = [dict(n=n, holds=n != 3) for n in range(1, 6)]
    kind, text = V.verdict_bulow_flat("exp", rows, 20.0)
    assert kind == "warning" and "4 von 5 Zeilen" in text and "erste Zeile ohne die Aussage ist n = 3" in text
    assert V.verdict_bulow_flat("gleich", [dict(n=1, holds=True)], 0.0)[0] == "success"


def asym_view(w):
    return V.AsymView(w=w, v1=ARR, b_strong=ARR, v2=ARR, b_weak=ARR, rev_second=20.8333, rev_first=22.9503, rev_myerson=32.29, misallocation=0.0833, loss_first=0.0189, loss_myerson=0.154, top_bid=33.3)


def test_asymmetry_verdict_branches():
    kind, text = V.verdict_asym(asym_view(0.5))
    assert kind == "warning" and "bis 50 Euro" in text and "1,1016" in text and "8,3 %" in text and "der Zweitpreis mit Geboten gleich den Werten nie" in text
    kind, text = V.verdict_asym(asym_view(1.0))
    assert kind == "info" and "Gleich starke Bieter" in text and "keine falsche Zuteilung" in text


def test_learning_verdict_direction():
    res = dict(rule="Hedge", fmt="Erstpreis", n=2, alpha_last=0.456, alpha_eq=0.5, revenue=0.3026, revenue_eq=0.3333)
    kind, text = V.verdict_learning(res)
    assert kind == "warning" and "unter dem Gleichgewicht" in text and "0,456" in text and "0,044" in text
    assert "über dem Gleichgewicht" in V.verdict_learning(dict(res, alpha_last=0.58))[1]


# --- je Preset die erwartete Verdikt-Zeile (ein Preset ohne Effekt darf keinen Effekt melden) -----------------------------------------------------
def settings_of(name):
    p = C.PRESETS[name]
    return Settings(dist=p.get("dist", "gleich"), n=p.get("n", C.DEFAULT_N), r=float(p.get("r", C.DEFAULT_R)), v0=float(p.get("v0", C.DEFAULT_V0)), rho=p.get("rho", C.DEFAULT_RHO),
                    w=p.get("w", C.DEFAULT_W))


EXPECTED = {
    # Preset: (Art des Auktions-Verdikts, Art des Risiko-Verdikts, Art des Asymmetrie-Verdikts, Art des Bulow-Verdikts)
    "Zwei Bieter": ("info", "success", "info", "success"),
    "Viele Bieter": ("success", "success", "info", "success"),
    "Reserve-Gipfel": ("success", "success", "info", "success"),
    "Risikoscheu": ("info", "warning", "info", "success"),
    "Ein Bieter mehr statt Mindestgebot": ("info", "info", "info", "success"),
    "Unregulär (Bulow–Klemperer versagt)": ("info", "info", "info", "warning"),
    "Asymmetrisch": ("info", "success", "warning", "success"),
    "Lernende": ("info", "success", "info", "success"),
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_has_the_expected_verdicts(name):
    s = settings_of(name)
    main = V.verdict_main(V.analyse(s))[0]
    risk = V.verdict_risk(V.risk_view(s, 2, 2000))[0]
    asym = V.verdict_asym(V.asym_view(s))[0]
    bk = V.bulow_view(s)
    bulow = V.verdict_bulow(s.dist, s.v0, bk.row)[0]
    expected = EXPECTED[name]
    assert (main, risk, asym, bulow) == expected, (name, main, risk, asym, bulow)


def test_every_preset_is_covered_by_the_table_above():
    assert set(EXPECTED) == set(C.PRESETS)


def test_default_state_reports_no_effect_for_neutral_risk_and_symmetry():
    s = Settings()
    assert V.verdict_risk(V.risk_view(s, 2, 2000))[0] == "success"
    assert V.verdict_asym(V.asym_view(s))[0] == "info"


# --- Verdikte über die vorgerechneten Reihen an handgesetzten Tabellen ---------------------------------------------------------------------------------
def test_table_verdicts_count_cells():
    rows = [dict(kind="gleich", n=n, formula=10.0, rev_virtual=10.0, rev_order=10.0, rev_first=10.0, rev_allpay=10.0 if n < 4 else 10.5, gap_first=0.0, gap_allpay=1e-9,
                 sim_second=dict(mean=10.0, se=0.01), sim_first=dict(mean=10.0, se=0.01), sim_allpay=dict(mean=10.0, se=0.01)) for n in (2, 3, 4)]
    text = V.verdict_e1_tables({"e1": rows})[1]
    assert "in 3 von 3 Zeilen" in text and "9 von 9 Zellen" in text and "5,0e-01" in text         # größte Abweichung 0,5 bei n = 4
    e2 = [dict(n=n, rho=rho, ratio=ratio) for n, rho, ratio in ((2, 1.0, 1.0), (2, 0.5, 1.3), (5, 0.5, 1.1), (5, 0.25, 1.2))]
    text = V.verdict_e2_tables({"e2": e2})[1]
    assert "in 3 von 3 Zellen" in text and "1,1000 bis 1,3000" in text and "sinkt mit n: ja" in text


def _cell(mean, eq=0.5, se=0.001, eta=None, rule="Hedge"):
    return {"eta": eta, "rule": rule, "rounds": C.LEARN_T[rule], "alpha_last": {"mean": mean, "se": se}, "alpha_eq": eq}


def test_e6_table_verdict_follows_the_counts_not_a_fixed_claim():
    """Das Verdikt zu den Lernverfahren sagt „kein Lernverfahren landet im Gleichgewicht“ nur, wenn es in allen Zellen mit Standardrate so ist; sonst nennt es die Zahlen."""
    far, near = _cell(0.3), _cell(0.5)
    kind, text = V.verdict_e6_tables({"cells": [far, far]})
    assert kind == "warning" and "landet kein Lernverfahren im Gleichgewicht: in 2 von 2 Zellen" in text
    kind, text = V.verdict_e6_tables({"cells": [far, near, near]})
    assert kind == "warning" and "kein Lernverfahren" not in text and "in 1 von 3 Zellen" in text and "übrigen 2 Zellen" in text
    kind, text = V.verdict_e6_tables({"cells": [near, near]})
    assert kind == "success" and "in allen 2 Zellen" in text and "kein Lernverfahren" not in text
    kind, text = V.verdict_e6_tables({"cells": [_cell(0.3, eta=1.0)]})                 # feste Rate zählt nicht zur Standardrate
    assert kind == "info" and "Keine Zellen mit Standardrate" in text
