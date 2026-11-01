"""Orakel für das Lernen: das Klassenspiel b = α·v in exakten Brüchen gegen scipy-Integration, die Rasterfalle (zwei Gleichgewichte), und die Lernläufe gegen Eigenschaften (Zufall nur als
Bandbreiten, nie exakt)."""

from fractions import Fraction as F

import pytest

import eza_learning as LE

pytest.importorskip("scipy")
from scipy import integrate  # noqa: E402


def payoff_quad(a: float, lam: float) -> float:
    """Erwartete Erstpreis-Auszahlung im Klassenspiel, n = 2, U[0, 1]: ∫₀¹ (1 − a) v · min(1, a v/λ) dv per scipy `quad` (Knick bei v = λ/a)."""
    if a == 0:
        return 0.0
    kink = [lam / a] if lam / a < 1 else None
    return integrate.quad(lambda v: (1 - a) * v * min(1.0, a * v / lam), 0, 1, points=kink, epsabs=1e-13)[0]


def payoff_double(a: float, lam: float) -> float:
    """Dieselbe Größe als Doppelintegral E[(v − a v) 1{a v > λ v′}] über (v, v′) (scipy `dblquad`, Indikator als Integrationsgrenze: v′ < a v/λ)."""
    return integrate.dblquad(lambda vp, v: (1 - a) * v, 0, 1, 0, lambda v: min(1.0, a * v / lam))[0]


@pytest.mark.parametrize("a,lam", [(0.25, 0.5), (0.5, 0.5), (0.45, 0.5), (0.6, 0.5), (0.9, 0.3), (0.45, 0.45), (0.2, 0.9)])
def test_class_game_payoff_matches_numerical_integration(a, lam):
    exact = float(LE.class_game_payoff(F(a).limit_denominator(1000), F(lam).limit_denominator(1000)))
    assert exact == pytest.approx(payoff_quad(a, lam), abs=1e-10)
    assert exact == pytest.approx(payoff_double(a, lam), abs=1e-7)


def test_the_rastered_class_game_has_two_symmetric_equilibria():
    """Auf dem 21er-Raster (Schritt 0,05) sind α = 0,45 und α = 0,50 beide Beste Antwort auf sich selbst; keine andere Zahl des Rasters ist es."""
    assert LE.class_fixed_points() == [F(9, 20), F(1, 2)]


def test_numerical_best_replies_agree_with_the_exact_ones():
    grid = [F(k, 20) for k in range(21)]
    for lam in grid[1:]:
        values = [payoff_quad(float(a), float(lam)) for a in grid]
        best = max(values)
        # Beste Antwort numerisch: alle Raster-α mit Auszahlung innerhalb 1e-12 des Maximums; die exakte Beste Antwort gehört dazu
        assert float(LE.class_game_payoff(LE.class_best_reply(lam, grid), lam)) == pytest.approx(best, abs=1e-12)


def test_the_true_equilibrium_is_the_fixed_point_of_the_continuous_class_game():
    """Auf einem feinen Raster (Schritt 1/200) ist 1/2 weiter ein Gleichgewicht (die Ableitung der Auszahlung verschwindet dort), 99/200 ebenfalls (Rasterrand)."""
    assert F(1, 2) in LE.class_fixed_points(F(1, 200))
    lam = F(1, 2)
    grid = [F(k, 200) for k in range(201)]
    assert LE.class_best_reply(lam, grid) == F(1, 2)
    # Ableitung der Auszahlung (1 − a)(1/2 − λ²/(6a²)) nach a bei a = λ = 1/2: −(1/2 − 1/6) + (1/2)(1/(12 · 1/8)) = 0
    a = F(1, 2)
    derivative = -(F(1, 2) - lam ** 2 / (6 * a ** 2)) + (1 - a) * (lam ** 2 / (3 * a ** 3))
    assert derivative == 0


def test_best_replies_to_the_raster_fixed_point_on_a_finer_grid():
    grid = [F(k, 200) for k in range(201)]
    assert LE.class_best_reply(F(9, 20), grid) == F(47, 100)          # auf 0,45 ist 0,47 besser: 0,45 ist ein Rasterartefakt
    assert LE.class_best_reply(F(1, 2), grid) == F(1, 2)


@pytest.mark.parametrize("rule", LE.RULES)
@pytest.mark.parametrize("fmt", LE.FORMATS)
def test_a_learning_run_is_deterministic_in_the_seed_and_in_range(rule, fmt):
    a = LE.run(rule, 2, fmt, 1500, 7)
    b = LE.run(rule, 2, fmt, 1500, 7)
    c = LE.run(rule, 2, fmt, 1500, 8)
    assert a == b and a != c
    assert 0.0 <= a["alpha_last"] <= 1.0 and 0.0 <= a["alpha_played"] <= 1.0 and 0.0 <= a["revenue"] <= 1.0
    assert len(a["trajectory"]) in (LE.CHECKPOINTS, LE.CHECKPOINTS + 1) and a["alpha_eq"] == LE.equilibrium_alpha(2, fmt)


def test_hedge_with_huge_rate_follows_the_best_reply_in_the_second_price():
    """Zweitpreis: das Gebot α = 1 ist schwach dominant; mit großer Rate sammelt Hedge fast alles Gewicht bei α = 1 (Bandbreite, nie exakt)."""
    r = LE.hedge_run(2, "Zweitpreis", 3000, 5, eta=1.0)
    assert r["alpha_last"] > 0.95


def test_learned_bids_are_not_above_the_value():
    r = LE.q_run(2, "Erstpreis", 4000, 3, evaluation_auctions=5000)
    assert 0.0 <= r["revenue"] <= 1.0
    assert all(0.0 <= x <= 1.0 for x in r["trajectory"])

