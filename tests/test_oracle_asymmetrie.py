"""Orakel für die asymmetrische Erstpreis-Auktion: Differentialgleichungen des Gleichgewichts, Randwerte, scipy `dblquad` der Erlöse, Beste-Antwort-Gitter beider Bieter, Simulation
(SplitMix64) der Erlöse und der falschen Zuteilung, exakte Bruchrechnung des Myerson-Erlöses."""

from fractions import Fraction as F

import numpy as np
import pytest

import eza_asymmetry as AS
import eza_experiments as XP

pytest.importorskip("scipy")
from scipy import integrate  # noqa: E402

WS = (1.0, 0.9, 0.75, 0.5, 0.25)


@pytest.mark.parametrize("w", WS)
def test_first_order_conditions_by_difference_quotient(w):
    """x′(y − b) = x und y′(x − b) = y (Bedingungen erster Ordnung beider Bieter) per zentralem Differenzenquotient."""
    bb = AS.top_bid(w)
    bs = np.linspace(1e-3, bb - 1e-3, 500)
    h = 1e-6
    x, y = AS.inverse_strong(bs, w), AS.inverse_weak(bs, w)
    dx = (AS.inverse_strong(bs + h, w) - AS.inverse_strong(bs - h, w)) / (2 * h)
    dy = (AS.inverse_weak(bs + h, w) - AS.inverse_weak(bs - h, w)) / (2 * h)
    assert np.max(np.abs(dx * (y - bs) - x)) < 1e-7
    assert np.max(np.abs(dy * (x - bs) - y)) < 1e-7


@pytest.mark.parametrize("w", WS)
def test_boundary_values(w):
    bb = AS.top_bid(w)
    assert float(AS.inverse_strong(bb, w)) == pytest.approx(1.0, abs=1e-12)
    assert float(AS.inverse_weak(bb, w)) == pytest.approx(w, abs=1e-12)
    assert float(AS.inverse_strong(0.0, w)) == 0.0 and float(AS.inverse_weak(0.0, w)) == 0.0


def test_boundary_values_in_exact_fractions(asymmetry_half):
    w = F(1, 2)
    c = (w * w - 1) / (w * w)
    bb = w / (1 + w)
    assert c == asymmetry_half["c"] and bb == asymmetry_half["top"]
    assert 2 * bb / (1 + c * bb * bb) == 1 and 2 * bb / (1 - c * bb * bb) == w


@pytest.mark.parametrize("w", WS)
def test_best_response_gaps_of_both_bidders(w):
    gap = XP.asym_best_response_gap(w)
    assert gap["gap_strong"] < 1e-6 and gap["gap_weak"] < 1e-6


def test_best_response_checker_catches_a_wrong_weak_bid():
    """Negativkontrolle: bietet der schwache Bieter fälschlich v/2 (die symmetrische Funktion), ist der starke Bieter nicht mehr auf der Besten Antwort."""
    w = 0.5
    grid = np.linspace(0.0, 0.5, 20001)
    gaps = []
    for v1 in (0.6, 0.9, 1.0):
        win = np.minimum(2 * grid / w, 1.0)               # P(v2/2 < b) bei v2 ~ U[0, w]
        u = (v1 - grid) * win
        b_eq = float(AS.bid_strong(v1, w))
        u_eq = (v1 - b_eq) * min(2 * b_eq / w, 1.0)
        gaps.append(float(u.max() - u_eq))
    assert max(gaps) > 1e-3


@pytest.mark.parametrize("w", WS)
def test_revenues_by_double_integration(w):
    """E[min(v1, v2)] und E[max Gebot] als Doppelintegral über (v1, v2) mit Dichte 1/w (scipy `dblquad`), gegen die Formeln des Kerns."""
    spa = integrate.dblquad(lambda v2, v1: min(v1, v2) / w, 0, 1, 0, w, epsabs=1e-11)[0]
    assert AS.second_price_revenue(w) == pytest.approx(100 * spa, rel=1e-6)
    fpa = integrate.dblquad(lambda v2, v1: max(float(AS.bid_strong(v1, w)), float(AS.bid_weak(v2, w))) / w, 0, 1, 0, w, epsabs=1e-10)[0]
    assert AS.first_price_revenue(w) == pytest.approx(100 * fpa, rel=2e-5)


def exact_myerson(w: F) -> F:
    """∫₀^w [1 − (1 + z)(w + z)/(4w)] dz + ∫_w^1 (1 − z)/2 dz in Brüchen über Stammfunktionen (Polynome)."""
    first = w - (1 / (4 * w)) * (w * w + w * w / 2 + w ** 3 / 2 + w ** 3 / 3)
    second = (1 - w) ** 2 / 4
    return first + second


@pytest.mark.parametrize("w", (F(1), F(9, 10), F(3, 4), F(1, 2), F(1, 4)))
def test_myerson_revenue_exact(w):
    assert AS.myerson_revenue(float(w)) == pytest.approx(100 * float(exact_myerson(w)), rel=1e-12)


def test_myerson_revenue_at_w_one_is_five_twelfths(uniform_hand, asymmetry_half):
    assert exact_myerson(F(1)) == uniform_hand["revenue_reserve"]       # zwei gleiche U[0,1]: 5/12 = Zweitpreis mit Reserve ½
    assert exact_myerson(F(1, 2)) == asymmetry_half["myerson"]


@pytest.mark.parametrize("w", (0.9, 0.5, 0.25))
def test_simulation_of_revenues_and_misallocation(w):
    """Vier Blöcke: Erlöse der drei Verfahren und der Anteil falscher Zuteilung im Erstpreis gegen die Rechnung des Kerns (5 Standardfehler)."""
    m = 200_000
    res = {"zweit": [], "erst": [], "myerson": [], "falsch": [], "verlust": [], "my_verlust": []}
    for b in range(4):
        u = XP.uniform_block(XP.seed_of(f"orakel|{w}|{b}"), 2 * m).reshape(2, m)
        v1, v2 = u[0], u[1] * w
        b1, b2 = AS.bid_strong(v1, w), AS.bid_weak(v2, w)
        res["zweit"].append(100 * np.mean(np.minimum(v1, v2)))
        res["erst"].append(100 * np.mean(np.maximum(b1, b2)))
        gewinnt1 = b1 > b2
        res["falsch"].append(np.mean(np.where(gewinnt1, v1 < v2, v2 < v1)))
        res["verlust"].append(1 - np.mean(np.where(gewinnt1, v1, v2)) / np.mean(np.maximum(v1, v2)))
        res["myerson"].append(XP._asym_sim(w, XP.seed_of(f"orakel|{w}|{b}"), "myerson"))
        p1, p2 = 2 * v1 - 1, 2 * v2 - w
        one, two = (p1 >= p2) & (p1 > 0), (p2 > p1) & (p2 > 0)
        res["my_verlust"].append(1 - np.mean(np.where(one, v1, np.where(two, v2, 0.0))) / np.mean(np.maximum(v1, v2)))
    exact_values = {"zweit": AS.second_price_revenue(w), "erst": AS.first_price_revenue(w), "myerson": AS.myerson_revenue(w),
                    "falsch": AS.first_price_misallocation(w)[0], "verlust": AS.first_price_misallocation(w)[1], "my_verlust": AS.myerson_welfare_loss(w)}
    for key, exact in exact_values.items():
        vals = np.array(res[key])
        assert abs(vals.mean() - exact) < 5 * vals.std(ddof=1) / 2.0 + 1e-9, (w, key, vals.mean(), exact)


def test_weak_bidder_bids_more_than_the_strong_one_at_equal_value():
    """Bei gleichem Wert bietet der schwache Bieter mehr (er schattiert weniger), der starke weniger."""
    for w in (0.9, 0.5, 0.25):
        v = np.linspace(0.05, w, 20)
        assert np.all(AS.bid_weak(v, w) > AS.bid_strong(v, w))
        assert np.all(AS.bid_weak(v, w) <= v / 1.0) and np.all(AS.bid_strong(v, w) <= v)


def test_symmetric_limit_is_the_symmetric_equilibrium():
    assert AS.first_price_revenue(1.0) == pytest.approx(100 / 3, rel=1e-8)
    assert AS.first_price_misallocation(1.0) == (0.0, 0.0)
