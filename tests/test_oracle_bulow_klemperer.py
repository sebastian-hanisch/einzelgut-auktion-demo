"""Orakel für Bulow–Klemperer: bestes Verfahren (Hülle der Erlöskurve) gegen scipy `ConvexHull`, gegen die Reserve aus φ(r*) = v0 (Weg A), gegen den besten Festpreis (n = 1) und gegen
Monte-Carlo; die einfache Auktion gegen die Quantilen-Integralform. Mit den Handwerten aus dem Aufsatz-Beispiel (U[0, 1]: Festpreis ½ gibt ¼, zwei Bieter ⅓)."""

import numpy as np
import pytest

import eza_auction as A
import eza_hull as HU
from eza_distributions import get_dist
from eza_rng import uniform_block

pytest.importorskip("scipy")
from scipy.spatial import ConvexHull  # noqa: E402

KINDS = ("gleich", "exp", "beta", "mix")


def scipy_upper_hull_indices(q, r):
    """Obere Hülle mit Qhull: Ecken der konvexen Hülle der Punkte (q, R) plus (q, tiefer Boden), nur die oberen Ecken."""
    floor = np.full_like(q, r.min() - 1.0)
    pts = np.concatenate([np.column_stack([q, r]), np.column_stack([q[[0, -1]], floor[[0, -1]]])])
    hull = ConvexHull(pts)
    idx = sorted(i for i in hull.vertices if i < len(q))
    return np.array(idx)


def test_hull_matches_qhull_on_the_mixture_and_on_a_dented_curve():
    q, r = HU.revenue_curve(get_dist("mix"), 20001)
    mine = HU.upper_hull(q, r)
    ref = scipy_upper_hull_indices(q, r)
    assert list(mine) == list(ref)
    assert not np.array_equal(mine, np.arange(len(q)))                 # die Hülle ändert die Mischung wirklich
    q2 = np.linspace(0, 1, 41)
    r2 = np.sin(3 * np.pi * q2) + q2
    assert list(HU.upper_hull(q2, r2)) == list(scipy_upper_hull_indices(q2, r2))


def test_hull_value_is_at_least_the_curve_everywhere():
    for kind in KINDS:
        q, r = HU.revenue_curve(get_dist(kind), 20001)
        h = HU.upper_hull(q, r)
        assert np.all(np.interp(q, q[h], r[h]) >= r - 1e-9)


@pytest.mark.parametrize("kind", ("gleich", "exp", "beta"))
def test_regular_distribution_hull_equals_the_reserve_route(kind):
    """Bei regulärer Verteilung ist R konkav, die Hülle ändert nichts, und das beste Verfahren ist der Zweitpreis mit der Reserve aus φ(r*) = v0 (Weg A)."""
    d = get_dist(kind)
    q, r = HU.revenue_curve(d, 100001)
    hull = HU.upper_hull(q, r)
    for v0 in (0.0, 20.0):
        for n in (1, 2, 5, 10):
            via_hull = HU.optimal_utility_from_hull(q, r, n, v0, hull)
            via_reserve = HU.optimal_utility_regular(d, n, v0)
            assert via_hull == pytest.approx(via_reserve, rel=2e-4, abs=1e-3), (kind, v0, n)


@pytest.mark.parametrize("kind", KINDS)
def test_single_bidder_optimum_is_the_best_posted_price(kind):
    """n = 1: das beste Verfahren ist ein Festpreis p mit Nutzen p·(1 − F(p)) + v0·F(p); Gittersuche über p."""
    d = get_dist(kind)
    q, r = HU.revenue_curve(d, 200001)
    ps = np.linspace(0.0, d.hi if kind != "exp" else 600.0, 120001)
    for v0 in (0.0, 20.0):
        best_price = float(np.max(ps * (1 - d.cdf(ps)) + v0 * d.cdf(ps)))
        assert HU.optimal_utility_from_hull(q, r, 1, v0) == pytest.approx(best_price, rel=2e-4, abs=2e-3), (kind, v0)


def test_mixture_single_bidder_optimum_is_nine():
    """Preis 90 verkauft mit Wahrscheinlichkeit 0,1 (Erlös 9), Preis 10 mit 0,9 (Erlös 9); die Gittersuche und die Hülle liefern 9."""
    d = get_dist("mix")
    q, r = HU.revenue_curve(d, 400001)
    assert HU.optimal_utility_from_hull(q, r, 1, 0.0) == pytest.approx(9.0, abs=2e-3)
    assert float(r.max()) == pytest.approx(9.0, abs=2e-3)


def test_optimal_utility_by_monte_carlo_of_the_ironed_marginal_revenue():
    """E[max(MR̄(q₁), v0)] mit q₁ = kleinstes der n Quantile: Hüllensteigung an zufälligen q₁ ausgewertet (SplitMix64-Ströme), gegen die Gewichtsformel."""
    d = get_dist("mix")
    q, r = HU.revenue_curve(d, 20001)
    h = HU.upper_hull(q, r)
    qs, rs = q[h], r[h]
    slopes = np.diff(rs) / np.diff(qs)
    for n, v0 in ((2, 0.0), (5, 20.0)):
        draws = [float(np.mean(np.maximum(slopes[np.clip(np.searchsorted(qs, uniform_block(4000 + n + b, 200_000 * n).reshape(-1, n).min(axis=1), side="right") - 1, 0, len(slopes) - 1)], v0)))
                 for b in range(4)]
        se = np.std(draws, ddof=1) / 2.0
        assert abs(np.mean(draws) - HU.optimal_utility_from_hull(q, r, n, v0, h)) < 5 * se + 0.01


@pytest.mark.parametrize("kind", KINDS)
def test_plain_revenue_two_routes(kind):
    """Zweitpreis ohne Mindestgebot: Quantilen-Integral n(n−1)∫R(q)(1−q)^(n−2) dq gegen die Wertgitter-Rechnung (Weg B)."""
    d = get_dist(kind)
    q, r = HU.revenue_curve(d, 400001)
    for n in (2, 3, 6):
        assert HU.plain_revenue_from_curve(q, r, n) == pytest.approx(A.spa_revenue_order(d, n), rel=2e-4, abs=2e-3), (kind, n)
    assert HU.plain_revenue_from_curve(q, r, 1) == 0.0


def test_example_of_the_paper_scaled_to_euro(uniform_hand):
    """Aufsatz-Beispiel U[0, 1]: Festpreis ½ gibt ¼, zwei Bieter ohne Mindestgebot ⅓, Teilnahmeschwelle ¾; hier in Euro (mal 100)."""
    d = get_dist("gleich")
    assert HU.plain_utility(d, 1, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert HU.plain_utility(d, 2, 0.0) == pytest.approx(100 * float(uniform_hand["bk_next"]), rel=1e-6)
    target = HU.optimal_utility_regular(d, 1, 0.0)
    assert target == pytest.approx(100 * float(uniform_hand["bk_best"]), rel=1e-6)
    assert HU.participation_threshold(d, 1, 0.0, target) == pytest.approx(float(uniform_hand["bk_threshold"]), rel=1e-6)


def test_participation_threshold_pattern_of_the_uniform_rows():
    """p*(n) = (n + 2)/2^(n+1) für n = 1 bis 7 (Muster aus den Zeilen, nicht bewiesen)."""
    d = get_dist("gleich")
    for n in range(1, 8):
        target = HU.optimal_utility_regular(d, n, 0.0)
        assert HU.participation_threshold(d, n, 0.0, target) == pytest.approx((n + 2) / 2 ** (n + 1), abs=2e-6), n


def test_extra_bidders_needed_by_brute_force_count():
    """Mischung: zusätzliche Bieter k bis die einfache Auktion das beste Verfahren erreicht, nachgezählt mit der Quantilen-Form der einfachen Auktion."""
    d = get_dist("mix")
    q, r = HU.revenue_curve(d, 200001)
    hull = HU.upper_hull(q, r)
    for n in (1, 2, 3, 5):
        best = HU.optimal_utility_from_hull(q, r, n, 0.0, hull)
        k = next(k for k in range(1, 100) if HU.plain_revenue_from_curve(q, r, n + k) >= best)
        assert HU.extra_bidders_needed(d, n, 0.0, best) == k


@pytest.mark.parametrize("kind", ("gleich", "exp", "beta"))
def test_bulow_klemperer_statement_in_simulation(kind):
    """Die Aussage selbst, ohne Hülle und ohne Integrale gerechnet: das beste Verfahren mit n Bietern ist bei regulärer Verteilung der Zweitpreis mit Reserve r* (simuliert, SplitMix64),
    die einfache Auktion mit n + 1 Bietern der Zweitpreis ohne Mindestgebot (simuliert); der Unterschied liegt in 4 Blöcken zu je 100.000 Auktionen über 5 Standardfehlern."""
    d = get_dist(kind)
    for n in (1, 2, 3):
        r_star = A.optimal_reserve(d, 0.0)
        best = np.array([A.simulate_revenue(d, n, "zweit", 5000 + 10 * n + b, 100_000, r=r_star) for b in range(4)])
        plain = np.array([A.simulate_revenue(d, n + 1, "zweit", 6000 + 10 * n + b, 100_000) for b in range(4)])
        gap = plain.mean() - best.mean()
        se = np.sqrt(best.var(ddof=1) / 4 + plain.var(ddof=1) / 4)
        assert gap > 5 * se, (kind, n, gap, se)
