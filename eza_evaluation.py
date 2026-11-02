"""Kennzahlen, Zeilenbauer der Tabellen und Verdikte (reine Funktionen: `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`).

Zahlen in Euro je Slot, deutsches Zahlenformat (`de`). Die Zeilenbauer der vorgerechneten Reihen (`*_table`) erzeugen die Tabellen der App und der README
(tests/test_claims.py prüft, dass jede README-Zeile aus ihnen stammt)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import eza_asymmetry as AS
import eza_auction as A
import eza_constants as C
import eza_hull as HU
import eza_learning as LE
from eza_distributions import LABELS, Dist, get_dist
from eza_experiments import find
from eza_scenario import Settings

DIST_SHORT = {"gleich": "Gleichverteilung", "exp": "Exponentiell", "beta": "Beta(2, 2)", "mix": "Mischung"}


def de(x: float, nd: int = 1) -> str:
    """Deutsches Zahlenformat: Dezimalkomma, Tausenderpunkt."""
    if x is None:
        return "–"
    s = f"{x:,.{nd}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def sci(x: float, nd: int = 1) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 1,3e-10."""
    return f"{x:.{nd}e}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


def pm(cell: dict, nd: int = 2) -> str:
    """Mittel ± Standardfehler eines Messwerts {'mean', 'se'}."""
    return f"{de(cell['mean'], nd)} ± {de(cell['se'], nd)}"


# ----------------------------------------------------------------------------------------------- Hauptansicht
@dataclass(frozen=True)
class MainView:
    kind: str
    n: int
    r: float
    v0: float
    spa: float            # Erlös des Zweitpreises bei Mindestgebot r (Weg B)
    spa_virtual: float    # derselbe Erlös über die virtuellen Werte (Weg A)
    fpa: float            # Erlös des Erstpreises bei Mindestgebot r
    p_sale: float
    utility: float        # Erlös + v0 · P(unverkauft)
    utility_plain: float  # Mindestgebot = Eigenwert (nicht darunter verkaufen)
    r_star: float | None  # Reserve aus φ(r) = v0 (nur bei regulärer Verteilung)
    u_star: float | None
    r_grid: float         # Gittersuche über alle Mindestgebote
    u_grid: float
    loss: float           # Wohlfahrtsverlust bei Mindestgebot r
    efficient: float
    vs: np.ndarray        # Erstpreis-Gebotsfunktion bei Mindestgebot r: Werte und Gebote
    bs: np.ndarray

    @property
    def dist(self) -> Dist:
        return get_dist(self.kind)


def analyse(s: Settings) -> MainView:
    d = get_dist(s.dist)
    spa = A.spa_revenue_order(d, s.n, s.r)
    r_grid, u_grid = A.best_reserve_on_grid(d, s.n, s.v0)
    r_star = A.optimal_reserve(d, s.v0) if d.regular else None
    vs, bs = A.fpa_bid_function(d, s.n, s.r)
    return MainView(
        s.dist, s.n, s.r, s.v0, spa, A.spa_revenue_virtual(d, s.n, s.r), A.fpa_revenue(d, s.n, s.r), A.sale_probability(d, s.n, s.r),
        spa + s.v0 * (1.0 - A.sale_probability(d, s.n, s.r)), A.seller_utility(d, s.n, s.v0, s.v0), r_star,
        A.seller_utility(d, s.n, r_star, s.v0) if r_star is not None else None, r_grid, u_grid, A.welfare_loss(d, s.n, s.r, s.v0), A.efficient_value(d, s.n, s.v0), vs, bs)


def verdict_main(v: MainView) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: was bringt der Slot mit dem gewählten Mindestgebot, was ohne, was wäre das Beste."""
    u, u0 = v.utility, v.utility_plain
    best_r, best_u = (v.r_star, v.u_star) if v.r_star is not None else (v.r_grid, v.u_grid)
    best_name = "Reserve aus φ(r) = Eigenwert" if v.r_star is not None else "bestes Mindestgebot auf dem Gitter (die Verteilung ist unregulär)"
    if v.n == 1:
        head = "Mit einem einzigen Bieter gibt es keinen Wettbewerb: der Slot bringt das Mindestgebot oder nichts."
    else:
        head = f"Erst- und Zweitpreis bringen bei Mindestgebot {de(v.r, 0)} Euro im Mittel dasselbe: {de(v.spa, 2)} Euro (Erstpreis {de(v.fpa, 2)} Euro)."
    text = (f"{head} Mit dem Eigenwert {de(v.v0, 0)} Euro für unverkaufte Slots ergibt sich ein Nutzen von **{de(u, 2)} Euro**; verkauft wird mit Wahrscheinlichkeit {pct(v.p_sale)}. "
            f"Ohne Mindestgebot über dem Eigenwert wären es {de(u0, 2)} Euro; {best_name}: {de(best_r, 1)} Euro mit {de(best_u, 2)} Euro.")
    if abs(u - best_u) < 0.01 * max(best_u, 1.0):
        return "success", text
    return "info", text


# ----------------------------------------------------------------------------------------------- Tab 1: Erstpreis, Zweitpreis, All-pay
@dataclass(frozen=True)
class EquivalenceView:
    kind: str
    n: int
    values: tuple[float, ...]
    first_bids: tuple[float, ...]
    allpay_bids: tuple[float, ...]
    rev_second: float
    rev_second_virtual: float
    rev_first: float
    rev_allpay: float
    gap_first: float
    gap_allpay: float
    sim: dict            # {'second': (Mittel, SE), ...} aus 4 kurzen Blöcken
    vs: np.ndarray
    bs: np.ndarray
    vs_allpay: np.ndarray
    bs_allpay: np.ndarray


LIVE_VALUES = (20.0, 50.0, 80.0)


def equivalence_view(s: Settings, sim_blocks: int = 4, sim_auctions: int = 50_000) -> EquivalenceView:
    from eza_experiments import seed_of          # Seeds aus Bezeichnungen (FNV-1a), nicht Pythons hash()
    d = get_dist(s.dist)
    vs, bs = A.fpa_bid_function(d, s.n)
    vsa, bsa = A.allpay_bid_function(d, s.n)
    values = LIVE_VALUES if s.dist != "mix" else (5.0, 9.0, 95.0)
    sim = {}
    for fmt in ("zweit", "erst", "allpay"):
        runs = np.array([A.simulate_revenue(d, s.n, fmt, seed_of(f"live|{s.seed}|{fmt}|{b}"), sim_auctions) for b in range(sim_blocks)])
        sim[fmt] = (float(runs.mean()), float(runs.std(ddof=1) / np.sqrt(sim_blocks)))
    return EquivalenceView(
        s.dist, s.n, values, tuple(float(np.interp(v, vs, bs)) for v in values), tuple(float(np.interp(v, vsa, bsa)) for v in values),
        A.spa_revenue_order(d, s.n), A.spa_revenue_virtual(d, s.n), A.fpa_revenue(d, s.n), A.allpay_revenue(d, s.n),
        max(A.fpa_best_response_gap(d, s.n, 0.0, v) for v in values), max(A.allpay_best_response_gap(d, s.n, v) for v in values), sim, vs, bs, vsa, bsa)


def verdict_equivalence(v: EquivalenceView) -> tuple[str, str]:
    """Sind die drei Regeln bei diesen Einstellungen erlösgleich, und liegen die Gebote im Gleichgewicht (Beste-Antwort-Lücke)?"""
    revs = (v.rev_second, v.rev_first, v.rev_allpay)
    spread = max(revs) - min(revs)
    if v.n == 1:
        return "info", "Mit einem Bieter gibt es keinen Wettbewerb: in allen drei Regeln ist der Erlös 0 (kein Mindestgebot, kein Konkurrent, der den Preis setzt)."
    same = spread < 1e-3 * max(max(revs), 1.0)
    shade = tuple(1 - b / x for b, x in zip(v.first_bids, v.values))
    return ("success" if same else "warning"), (
        f"Erlös: Zweitpreis {de(v.rev_second, 2)}, Erstpreis {de(v.rev_first, 2)}, All-pay {de(v.rev_allpay, 2)} Euro; größte Abweichung {de(spread, 4)} Euro"
        f"{' (Erlösäquivalenz auf dem Gitter)' if same else ''}. Im Erstpreis schattiert der Spediteur sein Gebot: "
        + ", ".join(f"beim Wert {de(x, 0)} Euro bietet er {de(b, 2)} Euro ({pct(sh)} darunter)" for x, b, sh in zip(v.values, v.first_bids, shade))
        + f". Größte Beste-Antwort-Lücke: Erstpreis {sci(v.gap_first)} Euro, All-pay {sci(v.gap_allpay)} Euro.")


@dataclass(frozen=True)
class RiskView:
    n: int
    rho: float
    factor: float
    rev_first: float
    rev_second: float
    gap: float
    sim: tuple[float, float]


def risk_view(s: Settings, sim_blocks: int = 4, sim_auctions: int = 50_000) -> RiskView:
    from eza_experiments import seed_of
    lam = A.crra_factor(s.n, s.rho)
    runs = np.array([A.simulate_crra_revenue(s.n, s.rho, seed_of(f"liverisk|{s.seed}|{b}"), sim_auctions) for b in range(sim_blocks)])
    return RiskView(s.n, s.rho, lam, lam * 100.0 * s.n / (s.n + 1), 100.0 * (s.n - 1) / (s.n + 1), max(A.crra_best_response_gap(s.n, s.rho, v) for v in (30.0, 60.0, 90.0)),
                    (float(runs.mean()), float(runs.std(ddof=1) / np.sqrt(sim_blocks))))


def verdict_risk(v: RiskView) -> tuple[str, str]:
    if v.n == 1:
        return "info", "Mit einem Bieter gibt es keinen Wettbewerb: beide Regeln bringen 0."
    ratio = v.rev_first / v.rev_second
    if v.rho >= 1.0:
        return "success", (f"Risikoneutral (Exponent 1): Gebotsfaktor {de(v.factor, 3)}, Erstpreis {de(v.rev_first, 2)} Euro gleich Zweitpreis {de(v.rev_second, 2)} Euro: die Erlösäquivalenz gilt. "
                           f"Beste-Antwort-Lücke {sci(v.gap)} Euro.")
    return "warning", (f"Risikoscheu (Exponent {de(v.rho, 2)}): Gebotsfaktor {de(v.factor, 3)} statt {de((v.n - 1) / v.n, 3)}; der Erstpreis bringt {de(v.rev_first, 2)} Euro, der Zweitpreis bleibt bei "
                       f"{de(v.rev_second, 2)} Euro (wahrheitsgetreue Gebote): Verhältnis {de(ratio, 4)}. Die Erlösäquivalenz bricht, der Erstpreis gewinnt. Beste-Antwort-Lücke {sci(v.gap)} Euro.")


# ----------------------------------------------------------------------------------------------- Tab 2: Mindestgebot
@dataclass(frozen=True)
class ReserveView:
    kind: str
    n: int
    v0: float
    rs: np.ndarray
    us: np.ndarray
    r_grid: float
    u_grid: float
    r_star: float | None
    u_star: float | None
    u_plain: float
    p_sale_star: float
    loss_star: float
    loss_star_pct: float
    first_at_reserve: float      # Erstpreis-Erlös beim Mindestgebot r der Seitenleiste
    second_at_reserve: float
    bid_at_reserve: float
    gap_first: float
    r: float


def reserve_view(s: Settings, curve_points: int = 400) -> ReserveView:
    d = get_dist(s.dist)
    rs, us = A.seller_utility_curve(d, s.n, s.v0)
    k = int(np.argmax(us))
    r_star = A.optimal_reserve(d, s.v0) if d.regular else None
    r_best = r_star if r_star is not None else float(rs[k])
    eff = A.efficient_value(d, s.n, s.v0)
    loss = A.welfare_loss(d, s.n, r_best, s.v0)
    step = max(1, len(rs) // curve_points)
    vs, bs = A.fpa_bid_function(d, s.n, s.r)
    gap = max(A.fpa_best_response_gap(d, s.n, s.r, v) for v in (s.r + 5.0, (s.r + max(100.0, s.r + 10.0)) / 2, 100.0) if v > s.r and d.cdf(v) > 1e-9) if s.n >= 2 and s.r < d.hi - 10 else 0.0
    return ReserveView(
        s.dist, s.n, s.v0, rs[::step], us[::step], float(rs[k]), float(us[k]), r_star, A.seller_utility(d, s.n, r_star, s.v0) if r_star is not None else None,
        A.seller_utility(d, s.n, s.v0, s.v0), A.sale_probability(d, s.n, r_best), loss, 100.0 * loss / eff, A.fpa_revenue(d, s.n, s.r), A.spa_revenue_order(d, s.n, s.r), float(bs[0]), gap, s.r)


def verdict_reserve(v: ReserveView) -> tuple[str, str]:
    best_r = v.r_star if v.r_star is not None else v.r_grid
    best_u = v.u_star if v.u_star is not None else v.u_grid
    gain = best_u - v.u_plain
    rel = f" ({pct(gain / v.u_plain, 2)})" if v.u_plain > 1e-9 else ""
    head = (f"Bestes Mindestgebot {de(best_r, 1)} Euro (Eigenwert {de(v.v0, 0)} Euro, {v.n} {'Bieter' if v.n == 1 else 'Bietern'}): Nutzen {de(best_u, 2)} Euro gegenüber {de(v.u_plain, 2)} Euro "
            f"ohne Mindestgebot über dem Eigenwert, Gewinn {de(gain, 2)} Euro{rel}. Der Preis ist Wohlfahrt: der Slot bleibt mit Wahrscheinlichkeit {pct(1 - v.p_sale_star)} unverkauft, "
            f"Verlust {de(v.loss_star, 2)} Euro ({de(v.loss_star_pct, 2)} % des effizienten Werts).")
    if v.r_star is None:
        return "warning", head + " Die Verteilung ist unregulär: die Reserve aus φ(r) = Eigenwert gilt nicht, die Gittersuche zeigt das beste einfache Mindestgebot (nicht das beste Verfahren)."
    return "success", head


# ----------------------------------------------------------------------------------------------- Tab 3: Ein Bieter mehr
@dataclass(frozen=True)
class BulowView:
    kind: str
    n: int
    v0: float
    rows: list[dict]
    row: dict


def bulow_rows(kind: str, v0: float, ns=tuple(range(1, 11))) -> list[dict]:
    d = get_dist(kind)
    q, r = HU.revenue_curve(d, 40001)
    hull = HU.upper_hull(q, r)
    return [HU.bulow_klemperer_row(d, n, v0, q, r, hull) for n in ns]


def bulow_view(s: Settings) -> BulowView:
    rows = bulow_rows(s.dist, s.v0, tuple(range(1, 11)) if s.n <= 10 else tuple(range(1, 11)) + (s.n,))
    return BulowView(s.dist, s.n, s.v0, rows, next(r for r in rows if r["n"] == s.n))


def verdict_bulow(kind: str, v0: float, row: dict) -> tuple[str, str]:
    """Gilt „ein Bieter mehr schlägt das beste Verfahren“ bei diesen Einstellungen? Antwort mit Zahlen und der verletzten Voraussetzung."""
    n = row["n"]
    base = (f"Das beste Verfahren mit {n} {'Bieter' if n == 1 else 'Bietern'} bringt {de(row['best'], 2)} Euro, eine einfache Auktion mit {n + 1} Bietern {de(row['plain_next'], 2)} Euro "
            f"(Verhältnis {de(row['ratio'], 3)}).")
    if row["holds"]:
        return "success", (f"{base} Der zusätzliche Bieter schlägt das beste Verfahren (bei einem Bieter ein Festangebot); er muss mit Wahrscheinlichkeit mindestens {pct(row['p_star'], 1)} teilnehmen "
                           f"(Schwelle p* = {de(row['p_star'], 4)}).")
    reason = ("die Verteilung ist unregulär (der virtuelle Wert steigt nicht)" if not get_dist(kind).regular
              else f"der Eigenwert des Verkäufers ({de(v0, 0)} Euro) liegt über manchen Bieterwerten (Voraussetzung „Bieterwert ≥ Eigenwert“ verletzt)")
    need = f"{row['extra']} zusätzliche Bieter" if row["extra"] is not None else f"mehr als {HU.MAX_EXTRA} zusätzliche Bieter"
    return "warning", f"{base} Die Aussage gilt hier nicht: {reason}. Es bräuchte {need}."


def verdict_bulow_flat(kind: str, rows: list[dict], v0: float) -> tuple[str, str]:
    """Zählung über die Zeilen n = 1 bis 10 bei den Einstellungen der Seitenleiste."""
    ok = sum(1 for r in rows if r["holds"])
    first_fail = next((r["n"] for r in rows if not r["holds"]), None)
    text = f"Bei {LABELS[kind]} und Eigenwert {de(v0, 0)} Euro gilt „einfache Auktion mit n + 1 Bietern ≥ bestes Verfahren mit n Bietern“ in {ok} von {len(rows)} Zeilen (n = {rows[0]['n']} bis {rows[-1]['n']})"
    if first_fail is not None:
        text += f"; die erste Zeile ohne die Aussage ist n = {first_fail}"
    return ("success" if ok == len(rows) else "warning"), text + "."


# ----------------------------------------------------------------------------------------------- Tab 4: Asymmetrie
@dataclass(frozen=True)
class AsymView:
    w: float
    v1: np.ndarray
    b_strong: np.ndarray
    v2: np.ndarray
    b_weak: np.ndarray
    rev_second: float
    rev_first: float
    rev_myerson: float
    misallocation: float
    loss_first: float
    loss_myerson: float
    top_bid: float


def asym_view(s: Settings) -> AsymView:
    w = s.w
    v1 = np.linspace(0.0, 1.0, 201)
    v2 = np.linspace(0.0, w, 201)
    mis, loss = AS.first_price_misallocation(w, 4001)
    return AsymView(w, AS.H * v1, AS.H * AS.bid_strong(v1, w), AS.H * v2, AS.H * AS.bid_weak(v2, w), AS.second_price_revenue(w), AS.first_price_revenue(w, 4001), AS.myerson_revenue(w), mis, loss,
                    AS.myerson_welfare_loss(w, 4001), AS.H * AS.top_bid(w))


def verdict_asym(v: AsymView) -> tuple[str, str]:
    if v.w >= 1.0:
        return "info", (f"Gleich starke Bieter (w = 1): beide bieten die Hälfte ihres Werts, Erstpreis {de(v.rev_first, 2)} und Zweitpreis {de(v.rev_second, 2)} Euro sind gleich, "
                        f"keine falsche Zuteilung. Das Myerson-optimale Verfahren bringt {de(v.rev_myerson, 2)} Euro (Mindestgebot 50 Euro, Wohlfahrtsverlust {pct(v.loss_myerson)}).")
    return "warning", (f"Der zweite Bieter hat Werte nur bis {de(100 * v.w, 0)} Euro (w = {de(v.w, 2)}). Erstpreis {de(v.rev_first, 2)} Euro gegen Zweitpreis {de(v.rev_second, 2)} Euro: Verhältnis "
                       f"{de(v.rev_first / v.rev_second, 4)}, die Äquivalenz bricht. Der Erstpreis vergibt den Slot in {pct(v.misallocation)} der Auktionen nicht an den höheren Wert "
                       f"(Wohlfahrtsverlust {pct(v.loss_first, 2)}), der Zweitpreis mit Geboten gleich den Werten nie. Myerson-optimal (je Bieter ein Mindestgebot): {de(v.rev_myerson, 2)} Euro, "
                       f"Wohlfahrtsverlust {pct(v.loss_myerson)}.")


# ----------------------------------------------------------------------------------------------- Tab 5: Lernen
def run_learning(s: Settings) -> dict:
    """Ein Lernlauf mit den Einstellungen der Seitenleiste."""
    from eza_experiments import seed_of
    return LE.run(s.rule, s.n_learn, s.fmt, s.rounds, seed_of(f"livelearn|{s.seed}|{s.rule}|{s.n_learn}|{s.fmt}|{s.rounds}|{s.rate}"), s.rate)


def verdict_learning(res: dict) -> tuple[str, str]:
    eq, last = res["alpha_eq"], res["alpha_last"]
    rev, rev_eq = res["revenue"], res["revenue_eq"]
    side = "unter" if last < eq else "über"
    return "warning", (f"{res['rule']} im {res['fmt']} mit {res['n']} Bietern: letzter Gebotsfaktor {de(last, 3)} gegen {de(eq, 3)} im Gleichgewicht ({side} dem Gleichgewicht, Abstand {de(abs(last - eq), 3)}). "
                       f"Erlös im letzten Viertel {de(rev, 4)} gegen {de(rev_eq, 4)} im Gleichgewicht (Einheit 100 Euro). **Ein Lauf, ein Zufalls-Seed:** ob diese Richtung bei anderem Seed, anderer Rate "
                       f"und anderem Raster bleibt, zeigt die vorgerechnete Tabelle (4 Läufe je Zelle).")


# ----------------------------------------------------------------------------------------------- Zeilenbauer der vorgerechneten Reihen
def e1_table(tables: dict) -> list[dict]:
    """Gleichverteilung, n = 2 bis 10: Gebot beim Wert 50, Erlöse auf allen Wegen (Euro)."""
    rows = []
    for c in (r for r in tables["e1"] if r["kind"] == "gleich"):
        rows.append({"n": c["n"], "Gebot bei Wert 50": de(c["bid_at_50"], 2), "Formel 100(n−1)/(n+1)": de(c["formula"], 4), "Zweitpreis Weg A": de(c["rev_virtual"], 4),
                     "Zweitpreis Weg B": de(c["rev_order"], 4), "Erstpreis": de(c["rev_first"], 4), "All-pay": de(c["rev_allpay"], 4)})
    return rows


def e1_sim_table(tables: dict) -> list[dict]:
    rows = []
    for c in (r for r in tables["e1"] if r["kind"] == "gleich"):
        rows.append({"n": c["n"], "Sim. Zweitpreis": pm(c["sim_second"]), "Sim. Erstpreis": pm(c["sim_first"]), "Sim. All-pay": pm(c["sim_allpay"]),
                     "Lücke Erstpreis (Euro)": sci(c["gap_first"]), "Lücke All-pay (Euro)": sci(c["gap_allpay"])})
    return rows


def e1_other_table(tables: dict) -> list[dict]:
    rows = []
    for c in (r for r in tables["e1"] if r["kind"] != "gleich"):
        rows.append({"Verteilung": DIST_SHORT[c["kind"]], "n": c["n"], "Zweitpreis Weg A": de(c["rev_virtual"], 4), "Zweitpreis Weg B": de(c["rev_order"], 4), "Erstpreis": de(c["rev_first"], 4),
                     "All-pay": de(c["rev_allpay"], 4), "Sim. Erstpreis": pm(c["sim_first"]), "Lücke Erstpreis (Euro)": sci(c["gap_first"])})
    return rows


def e1_control_table(tables: dict) -> list[dict]:
    return [{"Wert (Euro)": de(c["value"], 0), "Lücke, Gegner bieten w²/100": de(c["gap_wrong_opponent"], 2), "Lücke, Gegner bieten v/2": sci(c["gap_right_opponent"])} for c in tables["e1control"]]


def e2_table(tables: dict) -> list[dict]:
    rows = []
    for c in tables["e2"]:
        rows.append({"n": c["n"], "Exponent ρ": de(c["rho"], 2), "Gebotsfaktor": de(c["factor"], 4), "Erstpreis": de(c["rev_first"], 2), "Zweitpreis": de(c["rev_second"], 2),
                     "Verhältnis": de(c["ratio"], 4), "Sim. Erstpreis": pm(c["sim_first"]), "Lücke (Euro)": sci(c["gap"])})
    return rows


def e3_table(tables: dict) -> list[dict]:
    rows = []
    for c in (r for r in tables["e3"] if r["kind"] == "gleich"):
        rows.append({"Eigenwert": de(c["v0"], 0), "n": c["n"], "r* Formel": de(c["r_formula"], 1), "r* Bisektion": de(c["r_star"], 3), "r* Gitter": de(c["r_grid"], 2),
                     "Nutzen Weg A": de(c["u_virtual"], 3), "Nutzen Weg B": de(c["u_order"], 3), "Nutzen ohne Mindestgebot": de(c["u_plain"], 3), "Gewinn": "–" if c["gain_pct"] is None else de(c["gain_pct"], 3) + " %",
                     "Verkauf": pct(c["p_sale"]), "Wohlfahrtsverlust": de(c["loss_pct"], 3) + " %"})
    return rows


def e3_other_table(tables: dict) -> list[dict]:
    rows = []
    for c in (r for r in tables["e3"] if r["kind"] != "gleich"):
        rows.append({"Verteilung": DIST_SHORT[c["kind"]], "Eigenwert": de(c["v0"], 0), "n": c["n"], "r* Bisektion": de(c["r_star"], 3), "r* Gitter": de(c["r_grid"], 2),
                     "Nutzen Weg A": de(c["u_virtual"], 3), "Nutzen Weg B": de(c["u_order"], 3)})
    return rows


def e3_sim_table(tables: dict) -> list[dict]:
    return [{"n": c["n"], "Eigenwert": de(c["v0"], 0), "Mindestgebot": de(c["r"], 0), "Nutzen Weg B": de(c["u_order"], 3), "Nutzen Weg A": de(c["u_virtual"], 3), "Simulation": pm(c["sim"], 3)} for c in tables["e3sim"]]


def e3_fpa_table(tables: dict) -> list[dict]:
    return [{"n": c["n"], "Mindestgebot": de(c["r"], 0), "Erstpreis": de(c["rev_first"], 4), "Zweitpreis": de(c["rev_second"], 4), "Sim. Erstpreis": pm(c["sim_first"], 3),
             "Gebot beim Wert r": de(c["bid_at_r"], 3), "Lücke (Euro)": sci(c["gap"])} for c in tables["e3fpa"]]


def e4_table(tables: dict, kind: str, v0: float, ns=None) -> list[dict]:
    block = tables["e4"][f"{kind}|{v0}"]
    rows = block["rows"] if isinstance(block, dict) else block
    out = []
    for r in rows:
        if ns is not None and r["n"] not in ns:
            continue
        out.append({"n": r["n"], "Bestes Verfahren (n Bieter)": de(r["best"], 3), "Einfach (n + 1 Bieter)": de(r["plain_next"], 3), "Verhältnis": de(r["ratio"], 4),
                    "Aussage gilt": "ja" if r["holds"] else "nein", "Zusätzliche Bieter": "–" if r["extra"] is None else str(r["extra"]), "Schwelle p*": de(r["p_star"], 4)})
    return out


def e4_counts(tables: dict, kind: str, v0: float) -> tuple[int, int]:
    """(Zeilen mit gültiger Aussage, Zahl der Zeilen) der Reihe."""
    block = tables["e4"][f"{kind}|{v0}"]
    rows = block["rows"] if isinstance(block, dict) else block
    return sum(1 for r in rows if r["holds"]), len(rows)


def e5_table(tables: dict) -> list[dict]:
    return [{"w": de(c["w"], 2), "Zweitpreis": de(c["rev_second"], 2), "Erstpreis": de(c["rev_first"], 2), "Myerson": de(c["rev_myerson"], 2), "Erst / Zweit": de(c["ratio_first"], 4),
             "Myerson / Zweit": de(c["ratio_myerson"], 3), "falsche Zuteilung": pct(c["misallocation"], 2), "Verlust Erstpreis": pct(c["loss_first"], 2), "Verlust Myerson": pct(c["loss_myerson"], 1)}
            for c in tables["e5"]]


def e5_check_table(tables: dict) -> list[dict]:
    return [{"w": de(c["w"], 2), "Sim. Zweitpreis": pm(c["sim"]["second"]), "Sim. Erstpreis": pm(c["sim"]["first"]), "Sim. Myerson": pm(c["sim"]["myerson"]),
             "Lücke stark (Euro)": sci(c["gap_strong"]), "Lücke schwach (Euro)": sci(c["gap_weak"]), "höchstes Gebot": de(c["top_bid"], 2)} for c in tables["e5"]]


def e6_table(lernen: dict) -> list[dict]:
    rows = []
    for c in lernen["cells"]:
        if c["eta"] is not None or c["rounds"] != C.LEARN_T[c["rule"]]:
            continue
        rows.append({"Regel": c["rule"], "n": c["n"], "Format": c["fmt"], "Runden": de(c["rounds"], 0), "α Gleichgewicht": de(c["alpha_eq"], 3), "α letzter Iterierter": pm(c["alpha_last"], 3),
                     "Erlös": pm(c["revenue"], 4), "Erlös Gleichgewicht": de(c["revenue_eq"], 4)})
    return rows


def e6_horizon_table(lernen: dict) -> list[dict]:
    rows = []
    for c in lernen["cells"]:
        if c["rule"] != "Hedge" or c["n"] != 2:
            continue
        rows.append({"Format": c["fmt"], "Runden": de(c["rounds"], 0), "Rate": "Standardrate" if c["eta"] is None else f"fest {de(c['eta'], 1)}", "α Gleichgewicht": de(c["alpha_eq"], 3),
                     "α letzter Iterierter": pm(c["alpha_last"], 3), "Blöcke": ", ".join(de(x, 3) for x in c["alpha_last_blocks"])})
    rows.sort(key=lambda r: (r["Format"], r["Rate"] != "Standardrate", int(r["Runden"].replace(".", ""))))
    return rows


def learn_cell(lernen: dict, rule: str, n: int, fmt: str, rounds: int | None = None, eta=None) -> dict:
    rounds = C.LEARN_T[rule] if rounds is None else rounds
    return find(lernen["cells"], rule=rule, n=n, fmt=fmt, rounds=rounds, eta=eta)


# ----------------------------------------------------------------------------------------------- Verdikte über die vorgerechneten Reihen
def verdict_e1_tables(tables: dict) -> tuple[str, str]:
    rows = [r for r in tables["e1"] if r["kind"] == "gleich"]
    worst = max(max(abs(r[k] - r["formula"]) for k in ("rev_virtual", "rev_order", "rev_first", "rev_allpay")) for r in rows)
    worst_gap = max(max(r["gap_first"], r["gap_allpay"]) for r in rows)
    sim_ok = sum(1 for r in rows for k in ("sim_second", "sim_first", "sim_allpay") if abs(r[k]["mean"] - r["formula"]) < 5 * r[k]["se"] + 0.02)
    return "success", (f"Gleichverteilung, n = {rows[0]['n']} bis {rows[-1]['n']}: in {len(rows)} von {len(rows)} Zeilen sind Zweitpreis (zwei Wege), Erstpreis und All-pay gleich der Formel 100(n−1)/(n+1) "
                       f"(größte Abweichung {sci(worst)} Euro); die Simulation liegt in {sim_ok} von {3 * len(rows)} Zellen innerhalb von 5 Standardfehlern (plus 0,02 Euro); "
                       f"größte Beste-Antwort-Lücke {sci(worst_gap)} Euro.")


def verdict_e2_tables(tables: dict) -> tuple[str, str]:
    rows = [r for r in tables["e2"] if r["rho"] < 1.0]
    higher = sum(1 for r in rows if r["ratio"] > 1.0)
    by_n = {n: [r["ratio"] for r in rows if r["n"] == n] for n in sorted({r["n"] for r in rows})}
    shrink = all(max(by_n[a]) > max(by_n[b]) for a, b in zip(sorted(by_n), sorted(by_n)[1:]))
    return "warning", (f"Risikoscheu (ρ < 1): der Erstpreis bringt mehr als der Zweitpreis in {higher} von {len(rows)} Zellen; Verhältnisse {de(min(r['ratio'] for r in rows), 4)} bis "
                       f"{de(max(r['ratio'] for r in rows), 4)}; der größte Wert je n sinkt mit n: {'ja' if shrink else 'nein'} "
                       f"({', '.join(f'n = {n}: {de(max(v), 4)}' for n, v in by_n.items())}).")


def verdict_e3_tables(tables: dict) -> tuple[str, str]:
    rows = [r for r in tables["e3"] if r["kind"] == "gleich"]
    same_n = sum(1 for r in rows if abs(r["r_star"] - (100 + r["v0"]) / 2) < 1e-6 and abs(r["r_grid"] - r["r_star"]) < 0.011)
    return "success", (f"Gleichverteilung: Reserve aus φ(r) = Eigenwert, Formel (100 + v₀)/2 und Gittersuche stimmen in {same_n} von {len(rows)} Zellen überein "
                       f"(r* = {', '.join(sorted({de(r['r_star'], 0) for r in rows}))} Euro bei Eigenwert {', '.join(de(v, 0) for v in C.E3_V0S)}); r* hängt in diesen Zellen nicht von n ab (n = {rows[0]['n']} bis {rows[-1]['n']}).")


def verdict_e4_tables(tables: dict) -> tuple[str, str]:
    parts = []
    for kind in ("gleich", "exp", "beta"):
        ok, total = e4_counts(tables, kind, 0.0)
        parts.append(f"{DIST_SHORT[kind]} {ok}/{total}")
    ok20 = []
    for kind in ("gleich", "exp", "beta"):
        ok, total = e4_counts(tables, kind, C.E4_V0)
        ok20.append(f"{DIST_SHORT[kind]} {ok}/{total}")
    okm, totm = e4_counts(tables, "mix", 0.0)
    return "warning", (f"Aussage „einfache Auktion mit n + 1 Bietern ≥ bestes Verfahren mit n Bietern“ ohne Eigenwert: {', '.join(parts)}; mit Eigenwert {de(C.E4_V0, 0)} Euro: {', '.join(ok20)}; "
                       f"Mischung (unregulär): {okm}/{totm}.")


def verdict_e5_tables(tables: dict) -> tuple[str, str]:
    asym = [r for r in tables["e5"] if r["w"] < 1.0]
    return "warning", (f"Erstpreis über Zweitpreis in {sum(1 for r in asym if r['ratio_first'] > 1.0)} von {len(asym)} asymmetrischen Zeilen (Verhältnis {de(min(r['ratio_first'] for r in asym), 4)} bis "
                       f"{de(max(r['ratio_first'] for r in asym), 4)}); falsche Zuteilung im Erstpreis {pct(min(r['misallocation'] for r in asym), 2)} bis {pct(max(r['misallocation'] for r in asym), 2)}, "
                       f"Myerson-optimal {de(min(r['ratio_myerson'] for r in asym), 2)}- bis {de(max(r['ratio_myerson'] for r in asym), 2)}-fach der Zweitpreis-Erlös.")


def verdict_e6_tables(lernen: dict) -> tuple[str, str]:
    cells = [c for c in lernen["cells"] if c["eta"] is None and c["rounds"] == C.LEARN_T[c["rule"]]]
    off = [c for c in cells if abs(c["alpha_last"]["mean"] - c["alpha_eq"]) > 3 * max(c["alpha_last"]["se"], 1e-9) + 0.01]
    return "warning", (f"Mit Standardrate landet kein Lernverfahren im Gleichgewicht: in {len(off)} von {len(cells)} Zellen (Regel, n, Format) weicht der letzte Gebotsfaktor um mehr als 0,01 plus 3 Standardfehler "
                       f"vom Gleichgewicht ab. Die Richtung hängt von der Regel ab.")
