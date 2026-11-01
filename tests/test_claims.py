"""Jede Zahl aus der README und den Hilfetexten, nachgerechnet (DEMO-PLAYBOOK §4).

Drei Quellen, alle plattformsicher:
* die gespeicherten Messreihen (`data/tables.json`, `data/lernen.json`; Erzeuger-Hash geprüft in `test_measurements.py`): die README-Zahl ist der gespeicherte Wert, gerundet;
* exakte Rechnung (Brüche, Formeln, Identitäten): Gleitkommagleichheit nur mit `approx`;
* die Zeilenbauer der App (`eza_evaluation.*_table`): jede Zeile der README-Tabellen steht als Text in der README.
Keine Zeit-Assertion, keine exakten Zufallswerte einzelner Läufe. Zählungen über das Raster („in k von m Zellen“) stehen als Zählungen im Test, nicht als Urteilswort."""

import re
from fractions import Fraction as F
from pathlib import Path

import pytest

import eza_constants as C
import eza_evaluation as V
import eza_experiments as XP
import eza_hull as HU
import eza_learning as LE
import eza_texts as T
from eza_distributions import get_dist

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def german(x, nd):
    """Eigene Fassung des deutschen Zahlenformats (unabhängig von V.de): Dezimalkomma, Tausenderpunkt."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "T").replace(".", ",").replace("T", ".")


def row_of(label):
    """Die Tabellenzeile der Hypothese (eine Zeile der README, beginnt mit `| Hn |`)."""
    return next(line for line in README.splitlines() if line.startswith(f"| {label} |"))


def has(row, *needles):
    missing = [n for n in needles if n not in row]
    assert not missing, missing


def md_row(d):
    return "| " + " | ".join(str(v) for v in d.values()) + " |"


@pytest.fixture(scope="module")
def tables():
    return XP.load_tables()


@pytest.fixture(scope="module")
def lernen():
    return XP.load_lernen()


def cell(rows, **conds):
    return XP.find(rows, **conds)


# --- README und App: dieselben Texte, dieselben Tabellenzeilen ---------------------------------------------------------------------------------
def test_assumption_table_is_the_same_text_in_app_and_readme():
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def test_readme_tables_are_built_by_the_same_row_builders(tables, lernen):
    builders = (V.e1_table(tables), V.e1_sim_table(tables), V.e1_other_table(tables), V.e1_control_table(tables), V.e2_table(tables), V.e3_table(tables), V.e3_other_table(tables),
                V.e3_sim_table(tables), V.e3_fpa_table(tables), V.e4_table(tables, "gleich", 0.0), V.e4_table(tables, "exp", 20.0), V.e4_table(tables, "mix", 0.0), V.e5_table(tables),
                V.e5_check_table(tables), V.e6_table(lernen), V.e6_horizon_table(lernen))
    for rows in builders:
        assert rows
        for row in rows:
            assert md_row(row) in README, row


def test_hypothesis_table_has_h1_to_h10_with_a_clear_stand():
    rows = re.findall(r"^\| (H\d+) \|(?:\\\||[^|])*\|\s*\*\*([^*]+)\*\*", README, flags=re.M)
    assert [h for h, _ in rows] == [f"H{i}" for i in range(1, 11)]
    assert [a.split()[0].rstrip(",") for _, a in rows] == ["Ja"] * 6 + ["Nein"] * 3 + ["Ja"]


# --- Hypothesenzeilen --------------------------------------------------------------------------------------------------------------------------
def test_h1_bid_at_value_fifty_and_the_gap_of_the_checker(tables):
    g = [c for c in tables["e1"] if c["kind"] == "gleich"]
    row = row_of("H1")
    # Handrechnung: bei Gleichverteilung bietet der Wert 50 den Faktor (n − 1)/n: 25 / 33,33 / 40 / 45
    for n, expected in ((2, 25.0), (3, 100 / 3), (5, 40.0), (10, 45.0)):
        assert cell(g, n=n)["bid_at_50"] == pytest.approx(expected, rel=1e-6)
        assert german(expected, 2) in row
    has(row, f"{len(g)} von {len(g)} Zeilen")
    worst = max(max(c["gap_first"], c["gap_allpay"]) for c in g)
    assert f"{worst:.1e}".replace(".", ",") in row and worst < 1e-6
    control = tables["e1control"]
    assert [c["value"] for c in control] == [30.0, 60.0, 90.0] and all(c["gap_wrong_opponent"] > 0.4 for c in control)
    has(row, "Lücken " + ", ".join(german(c["gap_wrong_opponent"], 2) for c in control) + " Euro")


def test_h2_revenue_equivalence_counts(tables):
    g = [c for c in tables["e1"] if c["kind"] == "gleich"]
    row = row_of("H2")
    worst = 0.0
    for c in g:
        formula = 100.0 * (c["n"] - 1) / (c["n"] + 1)
        for key in ("rev_virtual", "rev_order", "rev_first", "rev_allpay"):
            worst = max(worst, abs(c[key] - formula))
    assert worst < 1e-4 and f"{worst:.1e}".replace(".", ",") in row
    has(row, f"in {len(g)} von {len(g)} Zeilen", german(100 / 3, 2) + " Euro bei n = 2", german(900 / 11, 2) + " Euro bei n = 10")
    sim_ok = sum(1 for c in g for k in ("sim_second", "sim_first", "sim_allpay") if abs(c[k]["mean"] - 100.0 * (c["n"] - 1) / (c["n"] + 1)) < 5 * c[k]["se"] + 0.02)
    assert sim_ok == 27
    has(row, "in 27 von 27 Zellen")
    others = [c for c in tables["e1"] if c["kind"] != "gleich"]
    assert len(others) == 6 and all(abs(c[k] - c["rev_order"]) < 1e-4 for c in others for k in ("rev_virtual", "rev_first", "rev_allpay"))
    has(row, "in 6 von 6 Zeilen gleich")


def test_h2_other_distributions_by_closed_forms(tables):
    """Exponentiell (Mittel 50): der k-kleinste von n Werten hat den Erwartungswert 50 · Σ_{j=1..k} 1/(n − j + 1); der zweithöchste ist der (n−1)-kleinste.
    n = 2: 50/2 = 25; n = 5: 50 · (1/5 + 1/4 + 1/3 + 1/2) = 64,1667."""
    exp = [c for c in tables["e1"] if c["kind"] == "exp"]
    assert float(50 * F(1, 2)) == 25.0 and cell(exp, n=2)["rev_order"] == pytest.approx(25.0, abs=1e-3)
    assert float(50 * (F(1, 5) + F(1, 4) + F(1, 3) + F(1, 2))) == pytest.approx(cell(exp, n=5)["rev_order"], abs=1e-3)
    # Mischung, n = 2: 0,81 · 10/3 + 0,18 · 5 + 0,01 · (90 + 10/3) (beide niedrig, einer niedrig, beide hoch)
    mix = [c for c in tables["e1"] if c["kind"] == "mix"]
    assert float(F(81, 100) * F(10, 3) + F(18, 100) * 5 + F(1, 100) * (90 + F(10, 3))) == pytest.approx(cell(mix, n=2)["rev_order"], abs=1e-3)


def test_h3_risk_aversion_counts(tables):
    row = row_of("H3")
    rho_rows = [c for c in tables["e2"] if c["rho"] < 1.0]
    assert len(rho_rows) == 9 and sum(1 for c in rho_rows if c["ratio"] > 1.0) == 9
    has(row, "in 9 von 9 Zellen")
    for n, rho in ((2, 0.25), (10, 0.75), (5, 0.5)):
        # Verhältnis = λ · n/(n − 1) mit λ = (n − 1)/(n − 1 + ρ), also n/(n − 1 + ρ)
        assert cell(tables["e2"], n=n, rho=rho)["ratio"] == pytest.approx(n / (n - 1 + rho), rel=1e-12)
    assert cell(tables["e2"], n=2, rho=0.25)["ratio"] == pytest.approx(1.6) and cell(tables["e2"], n=10, rho=0.75)["ratio"] == pytest.approx(10 / 9.75)
    has(row, "1,6000 bei n = 2 und ρ = 0,25 bis 1,0256 bei n = 10 und ρ = 0,75")
    per_n = {n: max(c["ratio"] for c in rho_rows if c["n"] == n) for n in C.E2_NS}
    assert per_n[2] > per_n[5] > per_n[10]
    has(row, " / ".join(german(per_n[n], 4) for n in (2, 5, 10)) + " bei n = 2 / 5 / 10", "gleich 1,0000 in 3 von 3 Zellen")
    assert all(abs(c["ratio"] - 1.0) < 1e-12 for c in tables["e2"] if c["rho"] == 1.0)


def test_h4_reserve_and_gain_counts(tables):
    row = row_of("H4")
    g = [c for c in tables["e3"] if c["kind"] == "gleich"]
    assert len(g) == 15 and all(abs(c["r_star"] - (100 + c["v0"]) / 2) < 1e-6 and abs(c["r_grid"] - c["r_star"]) < 0.011 for c in g)
    has(row, "in 15 von 15 Zellen", "r\\* = 50 / 60 / 70 Euro bei Eigenwert 0 / 20 / 40, n = 1 bis 10")
    # Handrechnung Gewinn bei Eigenwert 0: U(r*)/U(0) − 1 mit U(0) = (n−1)/(n+1) und U(1/2) = (n−1)/(n+1) + 2^−n/(n+1) → Gewinn = 2^−n/(n−1)
    for n in (2, 3, 5, 10):
        assert cell(g, v0=0.0, n=n)["gain_pct"] == pytest.approx(100 * 2.0 ** (-n) / (n - 1), rel=2e-3)      # n = 10: Differenz zweier Erlöse um 81,8, Gitterfehler 1e-6 Euro
    has(row, "25,0 / 6,25 / 0,78 / 0,011 % bei n = 2 / 3 / 5 / 10")
    assert [round(cell(g, v0=20.0, n=n)["gain_pct"], 3) for n in (1, 2, 5, 10)] == [80.0, 28.777, 2.29, 0.081]
    has(row, "80,0 / 28,8 / 2,3 / 0,081 % bei n = 1 / 2 / 5 / 10")
    # Wohlfahrtsverlust bei Eigenwert 0: P(Höchstwert < r*) · E[v | v < 1/2 des Höchsten]: n = 2: ∫₀^½ v · 2v dv = 1/12 von E[max] = 2/3 → 12,5 %
    assert cell(g, v0=0.0, n=2)["loss_pct"] == pytest.approx(100 * (1 / 12) / (2 / 3), rel=1e-5)
    has(row, "12,5 / 1,56 / 0,049 % des effizienten Werts bei n = 2 / 5 / 10")
    assert cell(g, v0=0.0, n=5)["loss_pct"] == pytest.approx(1.5625, abs=2e-3) and cell(g, v0=0.0, n=10)["loss_pct"] == pytest.approx(0.0488, abs=5e-4)
    fpa = tables["e3fpa"]
    assert len(fpa) == 3 and all(c["rev_first"] == pytest.approx(c["rev_second"], rel=1e-6) for c in fpa)
    has(row, "in 3 von 3 Zellen")


def test_h5_bulow_klemperer_counts(tables):
    row = row_of("H5")
    ok_regular = sum(V.e4_counts(tables, k, 0.0)[0] for k in ("gleich", "exp", "beta"))
    assert ok_regular == 30 and sum(V.e4_counts(tables, k, 0.0)[1] for k in ("gleich", "exp", "beta")) == 30
    assert V.e4_counts(tables, "exp", 20.0) == (9, 10) and V.e4_counts(tables, "gleich", 20.0) == (10, 10) and V.e4_counts(tables, "beta", 20.0) == (10, 10)
    assert V.e4_counts(tables, "mix", 0.0) == (0, 15)
    has(row, "30 von 30 Zeilen", "9 von 10", "0 von 15 Zeilen")
    g = {r["n"]: r for r in tables["e4"]["gleich|0.0"]}
    # Handrechnung U[0,1]: Festpreis 1/2 gibt 1/4, zwei Bieter 1/3: Verhältnis 4/3; n = 10: 11 Bieter ohne Mindestgebot 10/12 gegen bestes Verfahren (10−1)/11 + 2^−10/11
    assert g[1]["ratio"] == pytest.approx(4 / 3, rel=1e-4) and g[10]["ratio"] == pytest.approx((10 / 12) / (9 / 11 + 2.0 ** -10 / 11), rel=1e-4)
    has(row, "Verhältnis 1,3333 bei n = 1 bis 1,0184 bei n = 10")
    x = {r["n"]: r for r in tables["e4"]["exp|20.0"]}
    assert not x[1]["holds"] and x[1]["plain_next"] < x[1]["best"]
    has(row, f"bei n = 1 {german(x[1]['plain_next'], 2)} gegen {german(x[1]['best'], 2)} Euro (Verhältnis {german(x[1]['ratio'], 4)})")
    assert (german(x[1]["plain_next"], 2), german(x[1]["best"], 2), german(x[1]["ratio"], 4)) == ("31,23", "32,33", "0,9661")
    m = {r["n"]: r for r in tables["e4"]["mix|0.0"]["rows"]}
    assert (german(m[1]["plain_next"], 2), german(m[1]["best"], 2)) == ("4,53", "9,00")
    assert [m[n]["extra"] for n in (1, 2, 3, 5, 10)] == [3, 4, 6, 7, 10]
    has(row, "bei n = 1 4,53 gegen 9,00 Euro", "zusätzliche Bieter 3 / 4 / 6 / 7 / 10 bei n = 1 / 2 / 3 / 5 / 10")
    assert not any(r["holds"] for r in m.values())


def test_h5_mixture_numbers_are_stable_under_a_finer_hull():
    """Die Zahlen der Mischung (zusätzliche Bieter, n = 1: 9,00) bleiben bei 200.001 statt 100.001 Stützstellen der Hülle (Rasterzahl auf Stabilität geprüft, §11a Punkt 5)."""
    d = get_dist("mix")
    q, r = HU.revenue_curve(d, 200001)
    hull = HU.upper_hull(q, r)
    assert [HU.bulow_klemperer_row(d, n, 0.0, q, r, hull)["extra"] for n in (1, 2, 3, 5, 10)] == [3, 4, 6, 7, 10]
    assert HU.optimal_utility_from_hull(q, r, 1, 0.0, hull) == pytest.approx(9.0, abs=2e-3)
    assert not any(HU.bulow_klemperer_row(d, n, 0.0, q, r, hull)["holds"] for n in (1, 2, 3, 5, 10, 20, 60))


def test_h6_asymmetry_numbers(tables):
    row = row_of("H6")
    e5 = {c["w"]: c for c in tables["e5"]}
    # Handrechnung w = 1/2: Zweitpreis 5/24 (20,83 Euro); Erstpreis gegen Zweitpreis = 1,1016 (Integral, Tabelle)
    assert e5[0.5]["rev_second"] == pytest.approx(100 * 5 / 24)
    assert [round(e5[w]["ratio_first"], 4) for w in (0.9, 0.75, 0.5, 0.25)] == [1.0033, 1.0218, 1.1016, 1.2976] and e5[1.0]["ratio_first"] == pytest.approx(1.0, abs=1e-8)
    has(row, "1,0033 / 1,0218 / 1,1016 / 1,2976 bei w = 0,9 / 0,75 / 0,5 / 0,25")
    assert [round(100 * e5[w]["misallocation"], 2) for w in (0.9, 0.75, 0.5, 0.25)] == [2.37, 5.36, 8.33, 7.5]
    has(row, "2,37 % / 5,36 % / 8,33 % / 7,50 %")
    assert [round(100 * e5[w]["loss_first"], 2) for w in (0.9, 0.75, 0.5, 0.25)] == [0.10, 0.61, 1.89, 2.17]
    has(row, "0,10 % / 0,61 % / 1,89 % / 2,17 %")
    assert round(e5[1.0]["ratio_myerson"], 2) == 1.25 and round(e5[0.25]["ratio_myerson"], 2) == 2.48
    has(row, "1,25- bis 2,48-fach", "12,5 % bis 19,4 %")
    assert round(100 * e5[1.0]["loss_myerson"], 1) == 12.5 and round(100 * e5[0.25]["loss_myerson"], 1) == 19.4
    # Zweitpreis: Zuteilung immer an den höheren Wert (Gebot = Wert), der Erstpreis in 8,33 % (w = 1/2) nicht: 1/12 exakt
    assert e5[0.5]["misallocation"] == pytest.approx(1 / 12, rel=1e-4)
    # und die Simulation stimmt innerhalb von 5 Standardfehlern (plus 0,02 Euro): 15 von 15 Zellen
    ok = sum(1 for w, c in e5.items() for k, ref in (("second", "rev_second"), ("first", "rev_first"), ("myerson", "rev_myerson")) if abs(c["sim"][k]["mean"] - c[ref]) < 5 * c["sim"][k]["se"] + 0.02)
    assert ok == 15


def test_h7_h8_h9_learning_numbers(lernen):
    cells = [c for c in lernen["cells"] if c["eta"] is None and c["rounds"] == C.LEARN_T[c["rule"]]]
    assert len(cells) == 10
    off = [c for c in cells if abs(c["alpha_last"]["mean"] - c["alpha_eq"]) > 3 * max(c["alpha_last"]["se"], 1e-9) + 0.01]
    assert len(off) == 10
    h7, h8, h9 = row_of("H7"), row_of("H8"), row_of("H9")
    has(h7, "in 10 von 10 Zellen")
    for rule, n, fmt, text in (("Hedge", 2, "Erstpreis", "Hedge 0,456 ± 0,002"), ("Exp3", 2, "Erstpreis", "Exp3 0,422 ± 0,002"), ("Q-Lernen", 2, "Erstpreis", "Q-Lernen 0,584 ± 0,005")):
        c = V.learn_cell(lernen, rule, n, fmt)
        assert f"{rule} {german(c['alpha_last']['mean'], 3)} ± {german(c['alpha_last']['se'], 3)}" == text
        assert text in h7
    # Gleichgewicht n = 2: 0,5; Hedge und Exp3 liegen in allen 8 Zellen darunter, Q-Lernen im Erstpreis darüber, im Zweitpreis darunter
    hx = [c for c in cells if c["rule"] != "Q-Lernen"]
    assert len(hx) == 8 and all(c["alpha_last"]["mean"] < c["alpha_eq"] for c in hx)
    q_e, q_z = V.learn_cell(lernen, "Q-Lernen", 2, "Erstpreis"), V.learn_cell(lernen, "Q-Lernen", 2, "Zweitpreis")
    assert q_e["alpha_last"]["mean"] > q_e["alpha_eq"] and q_z["alpha_last"]["mean"] < q_z["alpha_eq"]
    has(h9, "in 8 von 8 Zellen unter dem Gleichgewicht", f"darüber ({german(q_e['alpha_last']['mean'], 3)} ± {german(q_e['alpha_last']['se'], 3)})",
        f"darunter ({german(q_z['alpha_last']['mean'], 3)} ± {german(q_z['alpha_last']['se'], 3)})")
    rev = lambda rule, n, fmt: V.learn_cell(lernen, rule, n, fmt)["revenue"]
    assert rev("Hedge", 2, "Zweitpreis")["mean"] > rev("Hedge", 2, "Erstpreis")["mean"]
    assert rev("Exp3", 2, "Erstpreis")["mean"] > rev("Exp3", 2, "Zweitpreis")["mean"]
    assert rev("Q-Lernen", 2, "Erstpreis")["mean"] > rev("Q-Lernen", 2, "Zweitpreis")["mean"]
    assert rev("Hedge", 4, "Zweitpreis")["mean"] > rev("Hedge", 4, "Erstpreis")["mean"]
    gap = abs(rev("Exp3", 4, "Erstpreis")["mean"] - rev("Exp3", 4, "Zweitpreis")["mean"])
    assert gap < 2 * (rev("Exp3", 4, "Erstpreis")["se"] + rev("Exp3", 4, "Zweitpreis")["se"])           # gleich innerhalb der Streuung
    has(h8, "Hedge Zweitpreis höher (0,3217 gegen 0,3026)", "Exp3 Erstpreis höher (0,2853 gegen 0,2750)", "Q-Lernen Erstpreis höher (0,3915 gegen 0,2789)",
        "Hedge Zweitpreis höher (0,5824 gegen 0,5581)", "Exp3 gleich innerhalb der Streuung (0,5218 ± 0,0005 gegen 0,5216 ± 0,0007)", "0,3333 bei n = 2, 0,6000 bei n = 4")
    assert LE.equilibrium_revenue(2) == pytest.approx(1 / 3) and LE.equilibrium_revenue(4) == pytest.approx(0.6)


def test_h9_raster_trap_in_exact_fractions_and_the_fixed_rate_runs(lernen):
    assert LE.class_fixed_points() == [F(9, 20), F(1, 2)]
    fixed = V.learn_cell(lernen, "Hedge", 2, "Erstpreis", 20000, C.LEARN_FIXED_ETA)
    hits = sum(1 for x in fixed["alpha_last_blocks"] if abs(x - 0.45) < 5e-4)
    other = [x for x in fixed["alpha_last_blocks"] if abs(x - 0.45) >= 5e-4]
    assert hits == 3 and len(other) == 1 and other[0] == pytest.approx(0.5, abs=1e-6)
    has(row_of("H9"), "in 3 von 4 Läufen bei 0,450")
    assert "in 3 von 4 Läufen bei 0,450 (auf drei Stellen) und im übrigen bei 0,500" in README
    fixed_z = V.learn_cell(lernen, "Hedge", 2, "Zweitpreis", 20000, C.LEARN_FIXED_ETA)
    assert all(abs(x - 1.0) < 5e-4 for x in fixed_z["alpha_last_blocks"])


def test_h10_threshold_pattern_is_proved_in_fractions_and_measured(tables):
    row = row_of("H10")
    for n in range(1, 13):
        r_n = F(n - 1, n + 1)                                  # Zweitpreis ohne Mindestgebot, U[0,1]
        best = r_n + F(1, 2 ** n) / (n + 1)                    # Zweitpreis mit Reserve 1/2: (n−1)/(n+1) + r^n − 2n r^(n+1)/(n+1) bei r = 1/2
        direct = r_n + F(1, 2) ** n - F(2 * n, n + 1) * F(1, 2) ** (n + 1)
        assert best == direct
        nxt = F(n, n + 2)
        assert (best - r_n) / (nxt - r_n) == F(n + 2, 2 ** (n + 1))
    g = {r["n"]: r for r in tables["e4"]["gleich|0.0"]}
    for n in range(1, 11):
        assert g[n]["p_star"] == pytest.approx((n + 2) / 2 ** (n + 1), abs=2e-6)
    has(row, " / ".join(german(g[n]["p_star"], 4) for n in (1, 2, 3, 4, 5, 6, 7, 10)) + " bei n = 1 / 2 / 3 / 4 / 5 / 6 / 7 / 10")


# --- Kernfrage, Korrekturen, Methodik ------------------------------------------------------------------------------------------------------------
def test_kernfrage_numbers(tables, lernen):
    text = README.split("## Kernfrage")[1].split("## Befunde und Korrekturen")[0]
    g = [c for c in tables["e1"] if c["kind"] == "gleich"]
    has(text, "33,33 Euro bei zwei und 81,82 Euro bei zehn Bietern", "25,00 Euro bei zwei und 45,00 Euro bei zehn Bietern", "1,3333-fach", "1,1016-fach", "8,33 %", "um 25,0 %", "12,5 % des effizienten Werts",
        "30 von 30 Zeilen", "0 von 15", "9 von 10 Zeilen bei Exponentialverteilung mit Eigenwert 20 Euro", "in 10 von 10 Zellen")
    assert cell(g, n=2)["bid_at_50"] == pytest.approx(25.0) and cell(g, n=10)["bid_at_50"] == pytest.approx(45.0)
    assert cell(tables["e2"], n=2, rho=0.5)["ratio"] == pytest.approx(4 / 3)


def test_correction_numbers(lernen):
    corr = README.split("## Befunde und Korrekturen gegenüber dem Plan")[1].split("## Modell")[0]
    q_e, q_z = V.learn_cell(lernen, "Q-Lernen", 2, "Erstpreis"), V.learn_cell(lernen, "Q-Lernen", 2, "Zweitpreis")
    has(corr, f"im Zweitpreis {german(q_z['alpha_last']['mean'], 3)} statt 0,909", f"im Erstpreis {german(q_e['alpha_last']['mean'], 3)} (Plan 0,577)")
    # Raster: in Brüchen (Plan nannte 0,495 für die Beste Antwort auf 0,50)
    fine = [F(k, 200) for k in range(201)]
    assert LE.class_best_reply(F(1, 2), fine) == F(1, 2) and LE.class_best_reply(F(9, 20), fine) == F(47, 100)
    has(corr, "der Wert des Plans war ein Rundungsartefakt der Quadratur", "**0,500**", "Auf 0,45 ist die Beste Antwort 0,47")
    has(corr, "für n = 1 bis 12", "für n = 1 bis 10", "200.001", "100.001", "20.001 Punkten", "Port 9221")
    assert "0,33 s" in corr and "1,65 s" in corr
    # Hash der Zahlen aus dem Plan: 3 / 4 / 6 / 7 / 10 zusätzliche Bieter (siehe test_h5_*)


def test_method_numbers_match_the_code():
    import eza_auction as A

    assert A.GRID_N == 20001 and A.BR_BIDS == 6001 and HU.HULL_N == 100001
    method = README.split("## Methodik")[1].split("## Befunde (gemessen")[0]
    has(method, "20.001 Punkten", "6.001 Gebote")
    assert C.BLOCKS == 4 and C.SIM_AUCTIONS == 200_000 and C.ASYM_SIM_AUCTIONS == 400_000 and C.LEARN_BLOCKS == 4
    has(README.split("## Befunde (gemessen")[1].split("| Nr. |")[0], "4 Blöcke zu je 200.000 Auktionen (Asymmetrie: 400.000), Lernen 4 Läufe")
    assert C.LEARN_T == {"Hedge": 20_000, "Exp3": 100_000, "Q-Lernen": 300_000}


# --- Hilfetexte der Presets und der App ----------------------------------------------------------------------------------------------------------
def test_preset_help_numbers_come_from_the_tables(tables):
    g = [c for c in tables["e1"] if c["kind"] == "gleich"]
    help_ = C.PRESET_HELP
    assert german(cell(g, n=2)["rev_order"], 2) in help_["Zwei Bieter"]
    assert f"{german(cell(g, n=10)['bid_at_50'], 0)} Euro" in help_["Viele Bieter"] and f"bei zwei Bietern {german(cell(g, n=2)['bid_at_50'], 0)} Euro" in help_["Viele Bieter"]
    assert german(cell(g, n=10)["rev_order"], 2) in help_["Viele Bieter"]
    e3 = [c for c in tables["e3"] if c["kind"] == "gleich"]
    top, plain = cell(e3, v0=20.0, n=2)["u_order"], cell(e3, v0=20.0, n=2)["u_plain"]
    assert (german(top, 2), german(plain, 2)) == ("47,73", "37,07") and "47,73 Euro statt 37,07 Euro" in help_["Reserve-Gipfel"]
    assert C.PRESETS["Reserve-Gipfel"]["r"] == cell(e3, v0=20.0, n=2)["r_star"] == 60
    c2 = cell(tables["e2"], n=2, rho=0.5)
    assert (german(c2["rev_first"], 2), german(c2["rev_second"], 2)) == ("44,44", "33,33") and "44,44 Euro" in help_["Risikoscheu"] and "33,33 Euro" in help_["Risikoscheu"]
    bk = {r["n"]: r for r in tables["e4"]["gleich|0.0"]}
    assert (german(bk[1]["best"], 2), german(bk[1]["plain_next"], 2)) == ("25,00", "33,33") and "25,00 Euro" in help_["Ein Bieter mehr statt Mindestgebot"]
    m = {r["n"]: r for r in tables["e4"]["mix|0.0"]["rows"]}
    assert (german(m[1]["best"], 2), german(m[1]["plain_next"], 2)) == ("9,00", "4,53") and "9,00 Euro" in help_["Unregulär (Bulow–Klemperer versagt)"] and "4,53 Euro" in help_["Unregulär (Bulow–Klemperer versagt)"]
    e5 = cell(tables["e5"], w=0.5)
    assert (german(e5["rev_first"], 2), german(e5["rev_second"], 2), german(100 * e5["misallocation"], 2)) == ("22,95", "20,83", "8,33")
    assert "22,95 Euro statt 20,83 Euro" in help_["Asymmetrisch"] and "8,33 %" in help_["Asymmetrisch"]
    assert LE.equilibrium_alpha(2, "Erstpreis") == 0.5 and "0,5" in help_["Lernende"] and C.PRESETS["Lernende"]["rounds"] == 20000 and "20.000 Runden" in help_["Lernende"]


def test_assumption_numbers_come_from_the_tables(tables, lernen):
    text = T.ASSUMPTIONS_MARKDOWN
    c = cell(tables["e2"], n=2, rho=0.5)
    c10 = cell(tables["e2"], n=10, rho=0.5)
    assert "44,44 statt 33,33 Euro" in text and german(c["rev_first"], 2) == "44,44" and german(c["rev_second"], 2) == "33,33" and german(c["ratio"], 4) == "1,3333" and "1,3333" in text
    assert "mit zehn Bietern nur noch 1,0526" in text and german(c10["ratio"], 4) == "1,0526" and c10["ratio"] == pytest.approx(10 / 9.5)
    e5 = cell(tables["e5"], w=0.5)
    assert "22,95 statt 20,83 Euro (Verhältnis 1,1016)" in text and german(e5["ratio_first"], 4) == "1,1016" and "8,33 %" in text
    m = {r["n"]: r for r in tables["e4"]["mix|0.0"]["rows"]}
    x = {r["n"]: r for r in tables["e4"]["exp|20.0"]}
    assert "0 von 15 Zeilen" in text and "9,00 Euro, zwei Bieter ohne Mindestgebot 4,53 Euro" in text and (german(m[1]["best"], 2), german(m[1]["plain_next"], 2)) == ("9,00", "4,53")
    assert "31,23 Euro (zwei Bieter) gegen 32,33 Euro (bestes Verfahren)" in text and (german(x[1]["plain_next"], 2), german(x[1]["best"], 2)) == ("31,23", "32,33")
    assert "9 von 10 Zeilen" in text and V.e4_counts(tables, "exp", 20.0) == (9, 10) and V.e4_counts(tables, "mix", 0.0)[0] == 0
    hedge = V.learn_cell(lernen, "Hedge", 2, "Erstpreis")
    assert german(hedge["alpha_last"]["mean"], 3) == "0,456" and "0,456 statt 0,5 (4 Läufe)" in text and hedge["rounds"] == 20000


def test_verdicts_over_the_stored_tables_report_the_counts(tables, lernen):
    assert "in 9 von 9 Zeilen" in V.verdict_e1_tables(tables)[1] and "27 von 27 Zellen" in V.verdict_e1_tables(tables)[1]
    assert "9 von 9 Zellen" in V.verdict_e2_tables(tables)[1]
    assert "15 von 15 Zellen" in V.verdict_e3_tables(tables)[1]
    assert "Gleichverteilung 10/10, Exponentiell 10/10, Beta(2, 2) 10/10" in V.verdict_e4_tables(tables)[1] and "Exponentiell 9/10" in V.verdict_e4_tables(tables)[1] and "Mischung (unregulär): 0/15" in V.verdict_e4_tables(tables)[1]
    assert "4 von 4 asymmetrischen Zeilen" in V.verdict_e5_tables(tables)[1]
    assert "in 10 von 10 Zellen" in V.verdict_e6_tables(lernen)[1]


def test_literature_statements_name_what_was_not_read():
    lit = README.split("## Literatur")[1].split("## Tests")[0]
    has(lit, "im Volltext gelesen", "Nicht im Volltext geprüft", "Myerson", "Vickrey 1961", "Riley und Samuelson 1981", "Banchio und Skrzypacz 2022", "nicht reproduziert")
    assert "S. 180–194" in lit and "86(1)" in lit and "S. 58–73" in lit and "6(1)" in lit


def test_the_readme_ends_with_the_footer_and_names_the_port():
    assert README.rstrip().splitlines()[-1].startswith("Diese Demo ist Teil des Portfolios von")
    assert "9221" in README


def test_grid_error_statement_in_the_readme(tables):
    """„Gitterwerte tragen den Fehler der Trapezregel (kleiner als 10⁻⁴ Euro in den gezeigten Stellen)“: der Abstand der beiden Rechenwege des Zweitpreises und des Erstpreises zu ihnen
    liegt in allen gespeicherten Zellen unter 1e-4 Euro."""
    assert "kleiner als 10⁻⁴ Euro in den gezeigten Stellen" in README
    worst = 0.0
    for c in tables["e1"]:
        worst = max(worst, max(abs(c[k] - c["rev_order"]) for k in ("rev_virtual", "rev_first", "rev_allpay")))
    for c in tables["e3"] + tables["e3sim"]:
        worst = max(worst, abs(c["u_virtual"] - c["u_order"]))
    for c in tables["e3fpa"]:
        worst = max(worst, abs(c["rev_first"] - c["rev_second"]))
    assert worst < 1e-4
