"""Orakel der Auktionsrechnung gegen einen anderen Rechenweg: scipy-Integration (`quad`), `brentq`/`minimize_scalar`, scipy-Verteilungen, Differentialgleichung des Gleichgewichts,
Beste-Antwort-Gitter und Simulation. Der Kern rechnet mit Trapezregel auf einem Gitter und eigener Bisektion; scipy kommt nur hier vor (ORACLE-PLAYBOOK, DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import eza_auction as A
from eza_distributions import get_dist

scipy = pytest.importorskip("scipy")
from scipy import integrate, optimize, stats  # noqa: E402

KINDS = ("gleich", "exp", "beta", "mix")


def sp_cdf(kind):
    """Verteilungsfunktion aus scipy bzw. (Mischung) unabhängig von Hand geschrieben."""
    if kind == "gleich":
        return stats.uniform(0, 100).cdf
    if kind == "exp":
        return stats.expon(scale=50).cdf
    if kind == "beta":
        return stats.beta(2, 2, scale=100).cdf
    return lambda v: 0.9 * min(max(v / 10, 0.0), 1.0) + 0.1 * min(max((v - 90) / 10, 0.0), 1.0)


def sp_pdf(kind):
    if kind == "gleich":
        return stats.uniform(0, 100).pdf
    if kind == "exp":
        return stats.expon(scale=50).pdf
    if kind == "beta":
        return stats.beta(2, 2, scale=100).pdf
    return lambda v: 0.09 if 0 <= v < 10 else (0.01 if 90 < v <= 100 else 0.0)


def sp_hi(kind):
    return 1200.0 if kind == "exp" else 100.0


def bid_quad(kind, n, r, v):
    """Gleichgewichtsgebot des Erstpreises direkt aus der Formel: b = v − ∫_r^v (F(x)/F(v))^(n−1) dx (scipy quad)."""
    cdf = sp_cdf(kind)
    pts = [p for p in (10.0, 90.0) if r < p < v] if kind == "mix" else None
    val = integrate.quad(lambda x: (cdf(x) / cdf(v)) ** (n - 1), r, v, points=pts, epsabs=1e-11, epsrel=1e-11, limit=200)[0]
    return v - val


def spa_quad(kind, n, r):
    """Zweitpreis-Erlös über die Ordnungsstatistik des zweithöchsten Werts, scipy quad (andere Zerlegung als der Kern: Preis = max(r, V(2)), ausgewertet über P(V(2) > t))."""
    cdf = sp_cdf(kind)
    hi = sp_hi(kind)
    # E[Preis · 1{V(1) ≥ r}] = ∫_0^∞ P(Preis > t, V(1) ≥ r) dt; für t < r ist das P(V(1) ≥ r), für t ≥ r gilt Preis > t ⇔ V(2) > t
    below = r * (1 - cdf(r) ** n)
    p2 = lambda t: 1 - cdf(t) ** n - n * cdf(t) ** (n - 1) * (1 - cdf(t))        # P(V(2) > t) = P(mindestens zwei > t)
    pts = [p for p in (10.0, 90.0) if p > r] if kind == "mix" else None
    above = integrate.quad(p2, r, hi, points=pts, epsabs=1e-11, epsrel=1e-11, limit=400)[0] if n >= 2 else 0.0
    return below + above


@pytest.mark.parametrize("kind", KINDS)
def test_fpa_bid_function_matches_the_scipy_integral(kind):
    d = get_dist(kind)
    for n, r in ((2, 0.0), (3, 0.0), (5, 0.0), (4, 30.0), (10, 0.0)):
        vs, bs = A.fpa_bid_function(d, n, r)
        values = (20.0, 50.0, 80.0, 100.0) if kind != "mix" else (5.0, 9.5, 92.0, 99.0)
        for v in values:
            if v <= r or d.cdf(v) < 1e-9:
                continue
            assert np.interp(v, vs, bs) == pytest.approx(bid_quad(kind, n, max(r, 0.0), v), abs=2e-4), (kind, n, r, v)


@pytest.mark.parametrize("kind", ("gleich", "beta", "exp"))
def test_fpa_bid_solves_the_first_order_condition(kind):
    """Differentialgleichung des Gleichgewichts: b′(v) = (n − 1) · f(v)/F(v) · (v − b(v)), per Differenzenquotient auf dem Gitter."""
    d = get_dist(kind)
    for n in (2, 3, 6):
        vs, bs = A.fpa_bid_function(d, n)
        i = np.arange(2000, len(vs) - 2000, 997)
        db = (bs[i + 1] - bs[i - 1]) / (vs[i + 1] - vs[i - 1])
        rhs = (n - 1) * d.pdf(vs[i]) / d.cdf(vs[i]) * (vs[i] - bs[i])
        assert np.allclose(db, rhs, rtol=1e-5, atol=1e-7), (kind, n)


@pytest.mark.parametrize("kind", KINDS)
def test_second_price_revenue_two_gridpaths_match_scipy(kind):
    d = get_dist(kind)
    for n in (1, 2, 3, 5):
        for r in (0.0, 20.0, 60.0):
            ref = spa_quad(kind, n, r)
            assert A.spa_revenue_order(d, n, r) == pytest.approx(ref, rel=1e-5, abs=1e-4), ("B", kind, n, r)
            assert A.spa_revenue_virtual(d, n, r) == pytest.approx(ref, rel=1e-5, abs=1e-4), ("A", kind, n, r)


@pytest.mark.parametrize("kind", KINDS)
def test_revenue_equivalence_first_second_allpay(kind):
    d = get_dist(kind)
    for n in (2, 3, 5, 10):
        second = A.spa_revenue_order(d, n)
        assert A.fpa_revenue(d, n) == pytest.approx(second, rel=1e-5), (kind, n)
        assert A.allpay_revenue(d, n) == pytest.approx(second, rel=1e-5), (kind, n)


def test_uniform_revenue_formula_for_all_n():
    d = get_dist("gleich")
    for n in range(2, 11):
        assert A.spa_revenue_order(d, n) == pytest.approx(100.0 * (n - 1) / (n + 1), rel=1e-6)
        assert A.fpa_revenue(d, n) == pytest.approx(100.0 * (n - 1) / (n + 1), rel=1e-6)


def test_first_price_with_reserve_equals_second_price_with_reserve():
    d = get_dist("gleich")
    for n, r in ((2, 50.0), (3, 30.0), (5, 60.0)):
        assert A.fpa_revenue(d, n, r) == pytest.approx(A.spa_revenue_order(d, n, r), rel=1e-6)
        assert A.fpa_revenue(d, n, r) == pytest.approx(spa_quad("gleich", n, r), rel=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_best_response_gap_is_tiny_at_the_equilibrium(kind):
    d = get_dist(kind)
    values = (20.0, 50.0, 80.0, 100.0) if kind != "mix" else (2.0, 5.0, 9.0, 92.0, 95.0)
    for n in (2, 3, 5):
        for v in values:
            assert A.fpa_best_response_gap(d, n, 0.0, v) < 5e-5, (kind, n, v)
            assert A.allpay_best_response_gap(d, n, v) < 5e-5, (kind, n, v)


def test_best_response_checker_catches_a_wrong_opponent_strategy():
    """Negativkontrolle: bieten die Gegner fälschlich w²/100, ist v/2 keine Beste Antwort mehr (Lücken 0,52 / 1,46 / 2,68 Euro bei Wert 30 / 60 / 90 laut Vormessung)."""
    d = get_dist("gleich")
    vs = np.linspace(0.0, 100.0, A.BR_GRID_N)
    wrong = (vs, vs * vs / 100.0)
    gaps = [A.fpa_best_response_gap(d, 2, 0.0, v, opponent=wrong, own_bid=v / 2) for v in (30.0, 60.0, 90.0)]
    assert all(g > 0.4 for g in gaps) and gaps == sorted(gaps)
    assert all(A.fpa_best_response_gap(d, 2, 0.0, v, own_bid=v / 2) < 1e-6 for v in (30.0, 60.0, 90.0))


def test_a_wrong_own_bid_against_the_right_opponents_has_a_gap():
    d = get_dist("gleich")
    assert A.fpa_best_response_gap(d, 3, 0.0, 60.0, own_bid=60.0) > 5.0           # den Wert zu bieten (Zweitpreis-Gebot im Erstpreis) ist keine Beste Antwort


def test_best_response_gap_with_reserve():
    d = get_dist("gleich")
    for n, r in ((2, 50.0), (3, 30.0), (5, 60.0)):
        for v in (r + 5.0, (r + 100.0) / 2, 100.0):
            assert A.fpa_best_response_gap(d, n, r, v) < 5e-5


@pytest.mark.parametrize("n,rho", [(2, 0.5), (2, 0.25), (5, 0.75), (10, 0.5), (3, 1.0)])
def test_risk_averse_bid_factor_is_the_best_reply(n, rho):
    """Gegner bieten λ·w; die Beste Antwort eines Bieters mit Wert v bei Auszahlung (v − x)^ρ ist x = λ·v (scipy `minimize_scalar`)."""
    lam = A.crra_factor(n, rho)
    for v in (30.0, 60.0, 90.0):
        res = optimize.minimize_scalar(lambda x: -((v - x) ** rho * min(x / (lam * 100.0), 1.0) ** (n - 1)), bounds=(0.0, v), method="bounded", options={"xatol": 1e-10})
        assert res.x == pytest.approx(lam * v, rel=1e-5)
        assert A.crra_best_response_gap(n, rho, v) < 1e-7


def test_risk_averse_first_price_revenue_formula_by_integration():
    for n, rho in ((2, 0.5), (5, 0.75), (10, 0.25)):
        lam = A.crra_factor(n, rho)
        value = integrate.quad(lambda v: lam * v * n * (v / 100.0) ** (n - 1) / 100.0, 0, 100)[0]       # E[λ · Höchstwert]
        assert value == pytest.approx(lam * 100.0 * n / (n + 1), rel=1e-9)


@pytest.mark.parametrize("v0", (0.0, 20.0, 40.0))
@pytest.mark.parametrize("n", (1, 2, 3, 5, 10))
def test_reserve_formula_bisection_and_optimizer_agree(v0, n):
    """Myerson-Reserve bei U[0, 100]: (100 + v0)/2, unabhängig von n. Drei Wege: Formel, eigene Bisektion, scipy `minimize_scalar` auf dem Nutzen (Weg B über quad)."""
    d = get_dist("gleich")
    expected = (100.0 + v0) / 2
    assert A.optimal_reserve(d, v0) == pytest.approx(expected, abs=1e-9)
    cdf = sp_cdf("gleich")
    res = optimize.minimize_scalar(lambda r: -(spa_quad("gleich", n, r) + v0 * cdf(r) ** n), bounds=(v0, 100.0), method="bounded", options={"xatol": 1e-7})
    assert res.x == pytest.approx(expected, abs=2e-3)
    assert A.best_reserve_on_grid(d, n, v0)[0] == pytest.approx(expected, abs=0.01)


def test_reserve_for_other_distributions_matches_brentq_on_scipy_virtual_value():
    for kind, v0, expected in (("exp", 0.0, 50.0), ("exp", 20.0, 70.0)):
        d = get_dist(kind)
        phi = lambda v: v - (1 - sp_cdf(kind)(v)) / sp_pdf(kind)(v)
        ref = optimize.brentq(lambda v: phi(v) - v0, 1e-6, 1000.0, xtol=1e-12)
        assert A.optimal_reserve(d, v0) == pytest.approx(ref, abs=1e-6) == pytest.approx(expected, abs=1e-6)
    for v0 in (0.0, 20.0):
        d = get_dist("beta")
        phi = lambda v: v - (1 - sp_cdf("beta")(v)) / sp_pdf("beta")(v)
        ref = optimize.brentq(lambda v: phi(v) - v0, 1e-3, 99.0, xtol=1e-12)
        assert A.optimal_reserve(d, v0) == pytest.approx(ref, abs=1e-6)
        for n in (1, 2, 5):
            assert A.best_reserve_on_grid(d, n, v0)[0] == pytest.approx(ref, abs=0.01)


def test_seller_utility_two_paths_match_scipy():
    d = get_dist("gleich")
    cdf = sp_cdf("gleich")
    for n in (2, 5):
        for r in (50.0, 60.0, 70.0):
            ref = spa_quad("gleich", n, r) + 20.0 * cdf(r) ** n
            assert A.seller_utility(d, n, r, 20.0) == pytest.approx(ref, rel=1e-6)
            assert A.spa_revenue_virtual(d, n, r) + 20.0 * cdf(r) ** n == pytest.approx(ref, rel=1e-6)


def test_welfare_loss_and_efficient_value_match_scipy():
    d = get_dist("beta")
    cdf, pdf = sp_cdf("beta"), sp_pdf("beta")
    for n, r, v0 in ((2, 40.0, 20.0), (5, 60.0, 0.0), (3, 10.0, 30.0)):
        dens = lambda v: n * cdf(v) ** (n - 1) * pdf(v)
        ref = integrate.quad(lambda v: max(v - v0, 0.0) * dens(v), 0, r, epsabs=1e-11)[0] + integrate.quad(lambda v: max(v0 - v, 0.0) * dens(v), r, 100, epsabs=1e-11)[0]
        assert A.welfare_loss(d, n, r, v0) == pytest.approx(ref, rel=1e-6, abs=1e-9)
        eff = integrate.quad(lambda v: max(v, v0) * dens(v), 0, 100, epsabs=1e-11)[0]
        assert A.efficient_value(d, n, v0) == pytest.approx(eff, rel=1e-6)


def test_sale_probability_by_enumeration_of_the_complement():
    d = get_dist("exp")
    assert A.sale_probability(d, 3, 50.0) == pytest.approx(1 - (1 - np.exp(-1)) ** 3)


@pytest.mark.parametrize("fmt", ("zweit", "erst", "allpay"))
@pytest.mark.parametrize("kind", ("gleich", "exp", "beta"))
def test_simulation_agrees_with_the_integrals_within_five_standard_errors(kind, fmt):
    d = get_dist(kind)
    n = 3
    ref = {"zweit": A.spa_revenue_order, "erst": A.fpa_revenue, "allpay": lambda dd, nn: A.allpay_revenue(dd, nn)}[fmt](d, n)
    sims = np.array([A.simulate_revenue(d, n, fmt, 1000 + b, 100_000) for b in range(4)])
    se = sims.std(ddof=1) / 2.0
    assert abs(sims.mean() - ref) < 5 * se + 0.001 * ref


def test_simulation_with_reserve_agrees_with_the_integral():
    d = get_dist("gleich")
    sims = np.array([A.simulate_revenue(d, 2, "zweit", 2000 + b, 100_000, r=50.0) for b in range(4)])
    assert abs(sims.mean() - 100 * 5 / 12) < 5 * sims.std(ddof=1) / 2.0 + 1e-9
    first = np.array([A.simulate_revenue(d, 2, "erst", 3000 + b, 100_000, r=50.0) for b in range(4)])
    assert abs(first.mean() - 100 * 5 / 12) < 5 * first.std(ddof=1) / 2.0 + 0.05


def test_mixture_revenue_by_exact_enumeration_of_the_groups():
    """Mischung, zwei Bieter, Zweitpreis: beide niedrig (0,81) E[min] = 10/3; einer niedrig (0,18) Preis = der niedrige, Mittel 5; beide hoch (0,01) 90 + 10/3."""
    expected = 0.81 * 10 / 3 + 0.18 * 5 + 0.01 * (90 + 10 / 3)
    assert expected == pytest.approx(4.5333333333)
    d = get_dist("mix")
    assert A.spa_revenue_order(d, 2) == pytest.approx(expected, rel=1e-5)
    assert A.spa_revenue_virtual(d, 2) == pytest.approx(expected, rel=1e-5)
    assert A.fpa_revenue(d, 2) == pytest.approx(expected, rel=1e-4)
