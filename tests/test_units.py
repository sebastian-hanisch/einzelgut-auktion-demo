"""Einheiten an den Mini-Instanzen aus conftest.py (Handrechnung), vor allem Regeln des Kerns: Zweitpreis-Ausgang, Aufzählung, Gebotsfunktionen, Reserve, Asymmetrie, Lernschritte."""

from fractions import Fraction as F

import numpy as np
import pytest

import eza_asymmetry as AS
import eza_auction as A
import eza_hull as HU
import eza_learning as LE
from eza_distributions import cumtrapz, get_dist, suffix_trapz


# --- Zweitpreis-Ausgang und Aufzählung ------------------------------------------------------------------------------------------------------------
def test_second_price_outcome_by_hand():
    assert A.second_price_outcome([5, 3, 4], 0) == (0, 4)              # höchstes Gebot 5, zweithöchstes 4
    assert A.second_price_outcome([5, 3, 4], 4.5) == (0, 4.5)           # Mindestgebot über dem zweiten Gebot: er zahlt das Mindestgebot
    assert A.second_price_outcome([5, 3, 4], 6) == (None, 0)            # unter dem Mindestgebot: kein Zuschlag
    assert A.second_price_outcome([7], 3) == (0, 3)                     # ein Bieter zahlt das Mindestgebot
    assert A.second_price_outcome([4, 4], 0) == (0, 4)                  # Gleichstand: der erste gewinnt, Preis = das gleiche Gebot
    assert A.second_price_outcome([], 3) == (None, 0)


def test_enumerated_revenue_of_the_mini_instance(values_123):
    v = values_123
    assert A.spa_revenue_enum(v["support"], v["n"], 0) == v["revenue_plain"]
    assert A.spa_revenue_enum(v["support"], v["n"], v["reserve"]) == v["revenue_reserve"]
    assert F(sum(1 for a in v["support"] for b in v["support"] if max(a, b) < v["reserve"]), 9) == v["unsold"]


def test_enumerated_revenue_scales_with_values_and_reserve():
    base = A.spa_revenue_enum([1, 2, 3], 2, 2)
    assert A.spa_revenue_enum([10, 20, 30], 2, 20) == 10 * base
    assert A.spa_revenue_enum([F(1, 2), 1, F(3, 2)], 2, 1) == base / 2


def test_first_price_utility_by_hand():
    assert A.first_price_utility(30, 60, F(1, 2)) == 15               # (60 − 30) · ½


# --- Verteilungen -----------------------------------------------------------------------------------------------------------------------------------
def test_uniform_by_hand():
    d = get_dist("gleich")
    assert d.cdf(25) == 0.25 and d.pdf(25) == 0.01 and d.ppf(0.25) == 25
    assert d.virtual_value(70) == pytest.approx(40.0)                  # 2v − 100
    assert d.regular


def test_exponential_by_hand():
    d = get_dist("exp")
    assert d.cdf(50) == pytest.approx(1 - np.exp(-1))
    assert d.virtual_value(80) == pytest.approx(30.0)                  # v − Mittel
    assert d.ppf(1 - np.exp(-1)) == pytest.approx(50.0)


def test_beta_by_hand():
    d = get_dist("beta")
    assert d.cdf(50) == pytest.approx(0.5) and d.cdf(25) == pytest.approx(3 / 16 - 2 / 64)      # 3·(¼)² − 2·(¼)³ = 5/32
    assert d.pdf(50) == pytest.approx(0.015)                           # 6·½·½/100
    assert d.ppf(0.5) == pytest.approx(50.0)


def test_mixture_by_hand():
    d = get_dist("mix")
    assert d.cdf(10) == pytest.approx(0.9) and d.cdf(50) == pytest.approx(0.9) and d.cdf(95) == pytest.approx(0.95)
    assert d.pdf(5) == pytest.approx(0.09) and d.pdf(50) == 0.0 and d.pdf(95) == pytest.approx(0.01)
    assert d.ppf(0.45) == pytest.approx(5.0) and d.ppf(0.95) == pytest.approx(95.0)
    assert np.isnan(d.virtual_value(50)) and not d.regular
    assert d.phi_times_density(50) == pytest.approx(-0.1)             # 0 − (1 − 0,9)


def test_cumulative_trapezoid_is_exact_for_linear_integrands():
    x = np.linspace(0, 2, 5)
    assert np.allclose(cumtrapz(x, x), x ** 2 / 2)
    assert np.allclose(suffix_trapz(x, x), 2 - x ** 2 / 2)


def test_unknown_distribution_is_refused():
    with pytest.raises(ValueError):
        get_dist("normal")


# --- Gebotsfunktionen --------------------------------------------------------------------------------------------------------------------------------
def test_first_price_bid_of_two_uniform_bidders_is_half_the_value(uniform_hand):
    d = get_dist("gleich")
    vs, bs = A.fpa_bid_function(d, 2)
    for v in (20.0, 60.0, 100.0):
        assert np.interp(v, vs, bs) == pytest.approx(v / 2)


def test_first_price_bid_with_reserve_starts_at_the_reserve():
    d = get_dist("gleich")
    vs, bs = A.fpa_bid_function(d, 2, r=50.0)
    assert vs[0] == 50.0 and bs[0] == pytest.approx(50.0)                # Gebot beim Wert r ist r
    # Handrechnung n = 2, r = 50, v = 80: b = v − ∫_50^80 (x/80) dx = 80 − (80² − 50²)/(2·80) = 80 − 24,375 = 55,625
    assert np.interp(80.0, vs, bs) == pytest.approx(55.625, abs=1e-6)


def test_first_price_bid_of_a_single_bidder_is_the_reserve():
    d = get_dist("gleich")
    vs, bs = A.fpa_bid_function(d, 1, r=30.0)
    assert np.allclose(bs, 30.0)


def test_allpay_bid_of_two_uniform_bidders():
    d = get_dist("gleich")
    vs, bs = A.allpay_bid_function(d, 2)
    for v in (30.0, 60.0, 100.0):
        assert np.interp(v, vs, bs) == pytest.approx(v * v / 200.0)    # b = (n−1)/n · v^n / 100^(n−1) = v²/200
    assert np.all(A.allpay_bid_function(d, 1)[1] == 0.0)


def test_risk_averse_bid_factor_by_hand(risk_half):
    assert A.crra_factor(2, 0.5) == pytest.approx(float(risk_half["factor"]))
    assert A.crra_factor(2, 1.0) == 0.5 and A.crra_factor(5, 1.0) == 0.8


# --- Erlöse -------------------------------------------------------------------------------------------------------------------------------------------
def test_revenues_of_the_uniform_pair(uniform_hand):
    d = get_dist("gleich")
    target = 100 * float(uniform_hand["revenue_plain"])
    assert A.spa_revenue_virtual(d, 2) == pytest.approx(target, rel=1e-6)
    assert A.spa_revenue_order(d, 2) == pytest.approx(target, rel=1e-6)
    assert A.fpa_revenue(d, 2) == pytest.approx(target, rel=1e-6)
    assert A.allpay_revenue(d, 2) == pytest.approx(target, rel=1e-6)


def test_reserve_revenue_and_utility_by_hand(uniform_hand):
    d = get_dist("gleich")
    assert A.spa_revenue_order(d, 2, 50.0) == pytest.approx(100 * float(uniform_hand["revenue_reserve"]), rel=1e-6)
    assert A.spa_revenue_virtual(d, 2, 50.0) == pytest.approx(100 * float(uniform_hand["revenue_reserve"]), rel=1e-6)
    assert A.fpa_revenue(d, 2, 50.0) == pytest.approx(100 * float(uniform_hand["revenue_reserve"]), rel=1e-6)
    assert A.sale_probability(d, 2, 50.0) == pytest.approx(0.75)
    assert A.seller_utility(d, 2, 50.0, 20.0) == pytest.approx(100 * float(uniform_hand["utility_20"]), rel=1e-6)


def test_one_bidder_pays_the_reserve():
    d = get_dist("gleich")
    assert A.spa_revenue_order(d, 1, 50.0) == pytest.approx(25.0)      # Preis 50, Verkaufswahrscheinlichkeit ½
    assert A.spa_revenue_virtual(d, 1, 50.0) == pytest.approx(25.0)
    assert A.spa_revenue_order(d, 1, 0.0) == 0.0


def test_optimal_reserve_by_hand(uniform_hand):
    d = get_dist("gleich")
    assert A.optimal_reserve(d, 0.0) == pytest.approx(100 * float(uniform_hand["reserve_v0_0"]), abs=1e-9)
    assert A.optimal_reserve(d, 50.0) == pytest.approx(100 * float(uniform_hand["reserve_v0_half"]), abs=1e-9)
    assert A.optimal_reserve(get_dist("exp"), 20.0) == pytest.approx(70.0, abs=1e-6)       # φ = v − 50
    with pytest.raises(ValueError):
        A.optimal_reserve(get_dist("mix"), 0.0)


def test_reserve_grid_search_finds_the_formula():
    d = get_dist("gleich")
    for n in (1, 2, 5):
        r, u = A.best_reserve_on_grid(d, n, 20.0)
        assert r == pytest.approx(60.0, abs=0.01)


def test_welfare_loss_by_hand():
    d = get_dist("gleich")
    # n = 1, v0 = 0, r = 50: der Slot bleibt unverkauft, wenn v < 50; Verlust E[v · 1{v < 50}] = 50²/200 = 12,5
    assert A.welfare_loss(d, 1, 50.0, 0.0) == pytest.approx(12.5, rel=1e-6)
    assert A.welfare_loss(d, 1, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert A.efficient_value(d, 1, 0.0) == pytest.approx(50.0)


# --- Bulow–Klemperer ---------------------------------------------------------------------------------------------------------------------------------
def test_bulow_klemperer_single_bidder(uniform_hand):
    d = get_dist("gleich")
    row = HU.bulow_klemperer_row(d, 1, 0.0)
    assert row["best"] == pytest.approx(100 * float(uniform_hand["bk_best"]), rel=1e-4)
    assert row["plain_next"] == pytest.approx(100 * float(uniform_hand["bk_next"]), rel=1e-6)
    assert row["p_star"] == pytest.approx(float(uniform_hand["bk_threshold"]), rel=1e-4)
    assert row["holds"] and row["extra"] == 1


def test_hull_of_a_concave_curve_is_the_curve_itself():
    q = np.linspace(0, 1, 11)
    r = q * (1 - q)
    assert list(HU.upper_hull(q, r)) == list(range(11))


def test_hull_removes_a_dent():
    q = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r = np.array([0.0, 2.0, 1.0, 2.0, 0.0])           # Delle bei q = 2: die Hülle springt von q = 1 über 3
    assert list(HU.upper_hull(q, r)) == [0, 1, 3, 4]


# --- Asymmetrie ------------------------------------------------------------------------------------------------------------------------------------
def test_asymmetry_constants_by_hand(asymmetry_half):
    h = asymmetry_half
    assert AS.asym_constant(0.5) == float(h["c"]) and AS.top_bid(0.5) == pytest.approx(float(h["top"]))
    assert float(AS.inverse_strong(AS.top_bid(0.5), 0.5)) == pytest.approx(float(h["x_top"]))
    assert float(AS.inverse_weak(AS.top_bid(0.5), 0.5)) == pytest.approx(float(h["y_top"]))
    assert AS.second_price_revenue(0.5) == pytest.approx(100 * float(h["second"]))
    assert AS.myerson_revenue(0.5) == pytest.approx(100 * float(h["myerson"]))


def test_symmetric_case_bids_half_the_value():
    assert AS.asym_constant(1.0) == 0.0
    assert float(AS.bid_strong(0.8, 1.0)) == 0.4 and float(AS.bid_weak(0.8, 1.0)) == 0.4


def test_asymmetric_bids_invert_the_inverse_functions():
    for w in (0.9, 0.5, 0.25):
        v1 = np.array([0.1, 0.5, 1.0])
        assert np.allclose(AS.inverse_strong(AS.bid_strong(v1, w), w), v1)
        v2 = np.array([0.1, 0.5, 1.0]) * w
        assert np.allclose(AS.inverse_weak(AS.bid_weak(v2, w), w), v2)


# --- Lernen --------------------------------------------------------------------------------------------------------------------------------------------
def test_payoff_of_all_alphas_by_hand():
    pay = LE.payoff_all_alphas(0.8, 0.4, "Erstpreis")
    assert pay[10] == 0.0                                              # α = 0,5: Gebot 0,4 = Gegnergebot, Gleichstand verloren
    assert pay[11] == pytest.approx(0.8 - 0.44)                        # α = 0,55: Gebot 0,44 gewinnt und zahlt 0,44
    assert LE.payoff_all_alphas(0.8, 0.4, "Zweitpreis")[11] == pytest.approx(0.8 - 0.4)    # Zweitpreis zahlt das Gegnergebot
    assert pay[0] == 0.0 and np.all(pay[:11] == 0.0)


def test_auction_outcome_by_hand():
    assert LE.auction_outcome(np.array([0.3, 0.5, 0.4]), "Erstpreis") == (1, 0.5, 0.4, 0.5)
    assert LE.auction_outcome(np.array([0.3, 0.5, 0.4]), "Zweitpreis") == (1, 0.5, 0.4, 0.4)
    assert LE.auction_outcome(np.array([0.5, 0.5]), "Zweitpreis")[0] == 0     # Gleichstand: kleinster Index


def test_hedge_and_exp3_probabilities_by_hand():
    p = LE.hedge_probabilities(np.zeros((2, 21)), 0.3)
    assert np.allclose(p, 1 / 21)
    losses = np.zeros((1, 21))
    losses[0, 0] = 1.0
    p = LE.hedge_probabilities(losses, np.log(2))
    assert p[0, 0] == pytest.approx(0.5 / (20 + 0.5)) and p[0, 1] == pytest.approx(1 / 20.5)
    q = LE.exp3_probabilities(np.zeros((1, 21)), 0.1)
    assert np.allclose(q, 1 / 21)


def test_draw_index_by_inversion():
    p = np.array([0.2, 0.5, 0.3])
    assert [LE.draw_index(p, u) for u in (0.0, 0.19, 0.2, 0.69, 0.7, 0.999)] == [0, 0, 1, 1, 2, 2]


def test_q_update_by_hand():
    assert LE.q_update(0.2, 0.6, 0.5) == pytest.approx(0.4)
    assert LE.q_update(0.3, 0.3, 0.9) == pytest.approx(0.3)


def test_equilibrium_of_the_learning_game():
    assert LE.equilibrium_alpha(2, "Erstpreis") == 0.5 and LE.equilibrium_alpha(4, "Erstpreis") == 0.75 and LE.equilibrium_alpha(2, "Zweitpreis") == 1.0
    assert LE.equilibrium_revenue(2) == pytest.approx(1 / 3) and LE.equilibrium_revenue(4) == pytest.approx(0.6)


def test_class_game_payoff_by_hand():
    # a = λ = 1/2: (1 − a)(1/2 − 1/6) = 1/2 · 1/3 = 1/6
    assert LE.class_game_payoff(F(1, 2), F(1, 2)) == F(1, 6)
    # a = 1/4, λ = 1/2 (a < λ): (1 − a) a / (3 λ) = (3/4)(1/4)/(3/2) = 1/8
    assert LE.class_game_payoff(F(1, 4), F(1, 2)) == F(1, 8)
    assert LE.class_game_payoff(F(0), F(1, 2)) == 0
