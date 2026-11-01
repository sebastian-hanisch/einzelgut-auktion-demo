"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): ein Bieter, genau zwei Bieter, Gleichstand, Mindestgebot am Rand und über dem Träger, Eigenwert über dem Mindestgebot,
Maxima der Regler. Dazu die Entartungen der Verteilungen (Lücke der Mischung) und der Lernläufe."""

import math

import numpy as np
import pytest

import eza_auction as A
import eza_evaluation as V
import eza_hull as HU
import eza_learning as LE
from eza_distributions import KINDS, get_dist
from eza_scenario import Settings


def finite(x):
    return bool(np.all(np.isfinite(x)))


def test_exactly_two_bidders_reproduce_the_hand_values():
    """n = 2 ist der kleinste Fall mit zweithöchstem Gebot (Zweige `n < 2` dürfen nicht greifen): Zweitpreis 100/3, Erstpreis-Gebot v/2."""
    d = get_dist("gleich")
    assert A.spa_revenue_order(d, 2) == pytest.approx(100 / 3, rel=1e-6)
    vs, bs = A.fpa_bid_function(d, 2)
    assert np.interp(70.0, vs, bs) == pytest.approx(35.0)


def test_a_single_bidder_has_no_competition_in_any_format():
    for kind in KINDS:
        d = get_dist(kind)
        assert A.spa_revenue_order(d, 1) == 0.0 and A.spa_revenue_virtual(d, 1) == pytest.approx(0.0, abs=1e-3)
        assert A.fpa_revenue(d, 1) == 0.0 and A.allpay_revenue(d, 1) == 0.0
        vs, bs = A.fpa_bid_function(d, 1)
        assert np.all(bs == 0.0)


def test_single_bidder_views_and_verdicts_do_not_divide_by_zero():
    for kind in KINDS:
        s = Settings(dist=kind, n=1, r=30.0, v0=10.0)
        assert finite(V.analyse(s).utility)
        eq = V.equivalence_view(s, sim_blocks=2, sim_auctions=2000)
        assert V.verdict_equivalence(eq)[0] == "info" and "Wettbewerb" in V.verdict_equivalence(eq)[1]
        rk = V.risk_view(s, sim_blocks=2, sim_auctions=2000)
        assert V.verdict_risk(rk)[0] == "info"
        assert V.verdict_reserve(V.reserve_view(s))[1]
        bk = V.bulow_view(s)
        assert bk.row["n"] == 1 and V.verdict_bulow(kind, s.v0, bk.row)[1]


def test_tied_bids_in_the_second_price_pay_that_bid():
    assert A.second_price_outcome([40, 40, 10], 0) == (0, 40)


def test_reserve_at_the_top_of_the_support_sells_nothing():
    d = get_dist("gleich")
    assert A.sale_probability(d, 3, 100.0) == 0.0
    assert A.spa_revenue_order(d, 3, 100.0) == pytest.approx(0.0, abs=1e-9)
    assert A.fpa_revenue(d, 3, 100.0) == pytest.approx(0.0, abs=1e-9)
    s = Settings(n=3, r=100.0)
    assert finite(V.analyse(s).utility) and finite(V.reserve_view(s).us)


def test_reserve_above_the_support_of_the_mixture_gap_and_exponential_tail():
    for kind, r in (("mix", 50.0), ("mix", 95.0), ("exp", 100.0), ("beta", 100.0)):
        d = get_dist(kind)
        for n in (1, 2, 5):
            assert finite(A.spa_revenue_order(d, n, r)) and finite(A.fpa_revenue(d, n, r))
            assert A.fpa_revenue(d, n, r) == pytest.approx(A.spa_revenue_order(d, n, r), rel=2e-3, abs=2e-3), (kind, r, n)


def test_reserve_below_the_sellers_own_value_is_allowed_and_costs_the_seller():
    d = get_dist("gleich")
    low = A.seller_utility(d, 2, 10.0, 40.0)
    plain = A.seller_utility(d, 2, 40.0, 40.0)
    assert low < plain                                              # unter dem Eigenwert zu verkaufen schadet


def test_efficient_value_and_welfare_loss_at_extreme_reserves():
    d = get_dist("gleich")
    assert A.welfare_loss(d, 2, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)         # Zuschlag immer an den Höchsten, Eigenwert 0
    assert A.welfare_loss(d, 2, 100.0, 0.0) == pytest.approx(A.efficient_value(d, 2, 0.0), rel=1e-6)     # nie verkauft: der ganze Wert geht verloren


def test_maximum_number_of_bidders_for_every_distribution():
    for kind in KINDS:
        s = Settings(dist=kind, n=20, r=40.0, v0=20.0)
        view = V.analyse(s)
        assert finite([view.utility, view.spa, view.fpa, view.spa_virtual]) and 0.0 <= view.p_sale <= 1.0
        assert view.spa == pytest.approx(view.spa_virtual, rel=1e-3, abs=2e-3)
        eq = V.equivalence_view(s, sim_blocks=2, sim_auctions=5000)
        assert finite([eq.rev_second, eq.rev_first, eq.rev_allpay, eq.gap_first, eq.gap_allpay])
        row = V.bulow_view(s).row
        assert row["n"] == 20 and finite(row["best"])


def test_exponential_with_many_bidders_has_no_nan_in_the_bid_functions():
    d = get_dist("exp")
    vs, bs = A.fpa_bid_function(d, 20)
    assert finite(bs) and np.all(bs <= vs + 1e-9) and np.all(np.diff(bs) >= -1e-9)


def test_mixture_bid_function_is_flat_in_the_gap_and_the_best_response_check_works():
    d = get_dist("mix")
    vs, bs = A.fpa_bid_function(d, 3)
    inside = (vs > 10.0) & (vs < 90.0)
    assert np.ptp(bs[inside]) < 1e-9                                 # in der Lücke ändert sich F nicht, das Gebot auch nicht
    assert np.all(np.diff(bs) >= -1e-9)
    assert A.fpa_best_response_gap(d, 3, 0.0, 95.0) < 1e-4


def test_asymmetry_at_the_slider_bounds():
    for w in (0.25, 1.0):
        view = V.asym_view(Settings(w=w))
        assert finite([view.rev_first, view.rev_second, view.rev_myerson, view.misallocation, view.loss_first, view.loss_myerson])
        assert V.verdict_asym(view)[1]
    assert V.asym_view(Settings(w=1.0)).misallocation == 0.0


def test_hull_with_a_single_segment_and_a_flat_curve():
    q = np.array([0.0, 1.0])
    assert HU.optimal_utility_from_hull(q, np.array([0.0, 0.0]), 3, 0.0) == 0.0
    assert HU.optimal_utility_from_hull(q, np.array([0.0, 2.0]), 1, 0.0) == pytest.approx(2.0)    # Steigung 2: Preis 2 verkauft immer


def test_extra_bidders_can_be_unreachable():
    d = get_dist("mix")
    assert HU.extra_bidders_needed(d, 1, 0.0, 1e9, limit=5) is None


def test_participation_threshold_without_gain_from_a_further_bidder_is_infinite():
    d = get_dist("gleich")
    assert math.isinf(HU.participation_threshold(d, 1, 100.0, 1.0))        # Eigenwert am Rand: ein weiterer Bieter ändert den Nutzen nicht


@pytest.mark.parametrize("rule", LE.RULES)
def test_shortest_learning_run_with_four_bidders(rule):
    res = LE.run(rule, 4, "Zweitpreis", 1000, 3, 30.0)                  # kürzester Lauf mit der größten Rate
    assert finite([res["alpha_last"], res["alpha_played"], res["revenue"]] + res["trajectory"])


def test_learning_run_with_the_smallest_rate():
    res = LE.run("Hedge", 2, "Erstpreis", 1000, 3, 0.25)
    assert 0.0 <= res["alpha_last"] <= 1.0


def test_pdf_export_with_unusual_settings():
    pytest.importorskip("fpdf")
    from eza_pdf_export import build_pdf

    s = Settings(dist="mix", n=1, r=100.0, v0=60.0)
    pdf = build_pdf(s, V.analyse(s), [("info", "Erlös 33,33 € ± 0,1 – α ρ φ λ ≥ ≤")])
    assert pdf[:4] == b"%PDF"
