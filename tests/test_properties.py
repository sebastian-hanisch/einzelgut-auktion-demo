"""Eigenschaftstests und Zweigtests (Nullspalten-Regel, DEMO-PLAYBOOK §0): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab, und was invariant sein muss, ist es
(Skalierung, Umbenennen der Bieter, Monotonie in n, Gebot höchstens der Wert, Hülle über der Kurve, bestes Verfahren über jedem einfachen Mindestgebot)."""

import itertools
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import eza_asymmetry as AS
import eza_auction as A
import eza_hull as HU
import eza_learning as LE
from eza_distributions import KINDS, get_dist

ROOT = Path(__file__).resolve().parent.parent


# --- Invarianten -------------------------------------------------------------------------------------------------------------------------------
def test_renaming_the_bidders_changes_neither_price_nor_winners_value():
    bids = [30, 70, 55, 55, 10]
    base = A.second_price_outcome(bids, 20)
    for perm in itertools.permutations(range(5), 5):
        permuted = [bids[i] for i in perm]
        winner, price = A.second_price_outcome(permuted, 20)
        assert price == base[1] and permuted[winner] == bids[base[0]]


def test_scaling_all_values_and_the_reserve_scales_the_enumerated_revenue():
    for c in (F(1, 3), 2, 7):
        assert A.spa_revenue_enum([c * v for v in (1, 2, 5)], 3, c * 2) == c * A.spa_revenue_enum([1, 2, 5], 3, 2)


@pytest.mark.parametrize("kind", KINDS)
def test_revenue_rises_with_the_number_of_bidders_in_every_format(kind):
    d = get_dist(kind)
    for fn in (A.spa_revenue_order, A.fpa_revenue, A.allpay_revenue):
        series = [fn(d, n) for n in range(1, 11)]
        assert all(b > a for a, b in zip(series, series[1:])), (kind, fn.__name__, series)


@pytest.mark.parametrize("kind", KINDS)
def test_bids_never_exceed_the_value_and_rise_with_it(kind):
    d = get_dist(kind)
    for n in (2, 5, 12):
        vs, bs = A.fpa_bid_function(d, n)
        assert np.all(bs <= vs + 1e-9) and np.all(np.diff(bs) >= -1e-9)
        vsa, bsa = A.allpay_bid_function(d, n)
        assert np.all(bsa <= vsa + 1e-9) and np.all(np.diff(bsa) >= -1e-9)
        assert np.all(bsa <= bs + 1e-4)                 # All-pay-Gebote liegen unter den Erstpreis-Geboten (Gebot × Gewinnwahrscheinlichkeit)


@pytest.mark.parametrize("kind", KINDS)
def test_more_bidders_pull_the_first_price_bid_towards_the_value(kind):
    d = get_dist(kind)
    v = 5.0 if kind == "mix" else 50.0
    bids = []
    for n in (2, 3, 5, 10):
        vs, bs = A.fpa_bid_function(d, n)
        bids.append(float(np.interp(v, vs, bs)) / v)
    assert all(b > a for a, b in zip(bids, bids[1:])), (kind, bids)


def test_best_reserve_is_at_least_as_good_as_any_other_reserve():
    d = get_dist("gleich")
    for n, v0 in ((1, 0.0), (2, 20.0), (5, 40.0)):
        r, u = A.best_reserve_on_grid(d, n, v0)
        assert all(A.seller_utility(d, n, rr, v0) <= u + 1e-5 for rr in range(0, 101, 7))      # 1e-5: Trapezregel auf dem Gitter


def test_the_best_mechanism_beats_every_simple_reserve_and_the_plain_auction():
    for kind in ("gleich", "exp", "beta"):
        d = get_dist(kind)
        for n in (1, 2, 4):
            for v0 in (0.0, 20.0):
                best = HU.optimal_utility_regular(d, n, v0)
                assert best >= HU.plain_utility(d, n, v0) - 1e-9
                assert all(best >= A.seller_utility(d, n, r, v0) - 1e-4 for r in (0.0, 25.0, 50.0, 75.0))


@pytest.mark.parametrize("kind", KINDS)
def test_the_hull_lies_above_the_revenue_curve(kind):
    q, r = HU.revenue_curve(get_dist(kind), 20001)
    h = HU.upper_hull(q, r)
    assert np.all(np.interp(q, q[h], r[h]) >= r - 1e-9)
    assert HU.optimal_utility_from_hull(q, r, 1, 0.0, h) >= float(r.max()) - 1e-9


def test_reserve_does_not_depend_on_n_and_rises_with_the_sellers_value():
    d = get_dist("gleich")
    found = {n: [A.best_reserve_on_grid(d, n, v0)[0] for v0 in (0.0, 20.0, 40.0)] for n in (1, 2, 5, 10)}
    for series in found.values():
        assert series == pytest.approx([50.0, 60.0, 70.0], abs=0.01)


def test_risk_averse_advantage_of_the_first_price_shrinks_with_n():
    ratios = [A.crra_factor(n, 0.5) * n / (n - 1) for n in (2, 3, 5, 10, 20)]
    assert all(a > b > 1.0 for a, b in zip(ratios, ratios[1:]))


# --- Zweigtests: jede Variante weicht ab ---------------------------------------------------------------------------------------------------------
def test_first_and_second_price_differ_under_risk_aversion_and_agree_without():
    d = get_dist("gleich")
    for n in (2, 5):
        second = A.spa_revenue_order(d, n)
        assert A.crra_factor(n, 1.0) * 100.0 * n / (n + 1) == pytest.approx(second)         # ρ = 1: gleich
        for rho in (0.75, 0.5, 0.25):
            assert A.crra_factor(n, rho) * 100.0 * n / (n + 1) > second * 1.01


def test_first_and_second_price_differ_under_asymmetry_and_agree_without():
    assert AS.first_price_revenue(1.0) == pytest.approx(AS.second_price_revenue(1.0), rel=1e-8)
    for w in (0.9, 0.5, 0.25):
        assert AS.first_price_revenue(w) > AS.second_price_revenue(w) * 1.003


def test_bulow_klemperer_fails_for_the_mixture_and_for_the_exponential_with_a_high_own_value_but_not_for_the_uniform():
    mix = get_dist("mix")
    q, r = HU.revenue_curve(mix, 40001)
    hull = HU.upper_hull(q, r)
    assert not any(HU.bulow_klemperer_row(mix, n, 0.0, q, r, hull)["holds"] for n in (1, 2, 5, 10))
    exp = get_dist("exp")
    assert not HU.bulow_klemperer_row(exp, 1, 20.0)["holds"] and HU.bulow_klemperer_row(exp, 2, 20.0)["holds"]
    gl = get_dist("gleich")
    assert all(HU.bulow_klemperer_row(gl, n, v0)["holds"] for n in (1, 2, 5) for v0 in (0.0, 20.0))


def test_the_hull_changes_the_mixture_but_not_the_regular_distributions():
    for kind, changes in (("mix", True), ("gleich", False), ("exp", False), ("beta", False)):
        q, r = HU.revenue_curve(get_dist(kind), 20001)
        h = HU.upper_hull(q, r)
        assert (len(h) < len(q) - 5) == changes, kind


@pytest.mark.parametrize("rule", ("Hedge", "Exp3"))
def test_hedge_and_exp3_leave_the_equilibrium_in_the_first_price_after_a_short_run(rule):
    res = LE.run(rule, 2, "Erstpreis", 6000, 11)
    assert abs(res["alpha_last"] - res["alpha_eq"]) > 0.02


@pytest.mark.parametrize("rule", LE.RULES)
def test_the_learning_rate_changes_the_result(rule):
    low = LE.run(rule, 2, "Erstpreis", 3000, 5, 0.25)
    high = LE.run(rule, 2, "Erstpreis", 3000, 5, 10.0)
    assert low["alpha_last"] != high["alpha_last"]


def test_first_and_second_price_learning_differ_in_what_they_learn():
    erst = LE.run("Hedge", 2, "Erstpreis", 4000, 5)
    zweit = LE.run("Hedge", 2, "Zweitpreis", 4000, 5)
    assert zweit["alpha_last"] > erst["alpha_last"] + 0.2


# --- Zufall: gleicher Seed, gleiches Ergebnis, auch in einem zweiten Prozess ----------------------------------------------------------------------
def test_simulation_and_learning_are_deterministic_in_the_seed():
    d = get_dist("exp")
    assert A.simulate_revenue(d, 3, "erst", 5, 20000) == A.simulate_revenue(d, 3, "erst", 5, 20000)
    assert A.simulate_revenue(d, 3, "erst", 5, 20000) != A.simulate_revenue(d, 3, "erst", 6, 20000)
    assert LE.run("Exp3", 2, "Zweitpreis", 1500, 9) == LE.run("Exp3", 2, "Zweitpreis", 1500, 9)


def test_results_do_not_depend_on_the_hash_seed_of_the_process():
    code = ("import sys; sys.path.insert(0, %r); import eza_learning as LE, eza_auction as A; from eza_distributions import get_dist; "
            "print(repr(LE.run('Hedge', 2, 'Erstpreis', 1500, 3)['alpha_last']), repr(A.simulate_revenue(get_dist('beta'), 3, 'allpay', 7, 20000)))" % str(ROOT))
    outputs = []
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": hash_seed}, check=True, timeout=120)
        outputs.append(out.stdout.strip())
    assert outputs[0] == outputs[1] and outputs[0]
