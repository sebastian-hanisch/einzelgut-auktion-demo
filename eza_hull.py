"""Bulow–Klemperer: das beste Verfahren mit n Bietern gegen eine einfache Auktion mit einem Bieter mehr (Festangebot im Quantilenraum).

Quantilenraum: q = 1 − F(v) ist die Wahrscheinlichkeit, dass ein Wert mindestens v ist; die **Erlöskurve** ist R(q) = q · v(q) (Festpreis v(q) verkauft
mit Wahrscheinlichkeit q). Der Verkäufer mit Eigenwert v0 und n Bietern erreicht im besten Verfahren E[max(MR(q₁), v0)], wo MR die Steigung der **oberen
konkaven Hülle** von R ist (bei regulärer Verteilung ist R schon konkav, die Hülle ändert nichts) und q₁ = 1 − F(Höchstwert) die Dichte n(1 − q)^(n−1) hat.
Die einfache Auktion mit n + 1 Bietern ist der Zweitpreis mit Mindestgebot v0 (kein weiteres Mindestgebot). Rechnung ohne scipy."""

from __future__ import annotations

import numpy as np

from eza_auction import GRID_N, optimal_reserve, seller_utility, spa_revenue_virtual, trapz
from eza_distributions import Dist

HULL_N = 100001            # Stützstellen der Erlöskurve im Quantilenraum
MAX_EXTRA = 200            # so viele zusätzliche Bieter werden höchstens probiert


def revenue_curve(d: Dist, n_points: int = HULL_N) -> tuple[np.ndarray, np.ndarray]:
    """Erlöskurve R(q) = q · v(q), v(q) = F⁻¹(1 − q), auf n_points Stützstellen q ∈ [0, 1] (R(0) = 0 und R(1) = lo)."""
    q = np.linspace(0.0, 1.0, n_points)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = d.ppf(1.0 - q)
    with np.errstate(invalid="ignore"):
        r = q * v
    r = np.where(np.isfinite(r), r, 0.0)
    r[0] = 0.0
    return q, r


def upper_hull(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Indizes der Ecken der oberen konkaven Hülle (monotone Kette, ein Durchgang)."""
    hull: list[int] = []
    for i in range(len(q)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (r[b] - r[a]) * (q[i] - q[a]) <= (r[i] - r[a]) * (q[b] - q[a]):     # b liegt auf oder unter der Strecke a–i
                hull.pop()
            else:
                break
        hull.append(i)
    return np.array(hull)


def optimal_utility_from_hull(q: np.ndarray, r: np.ndarray, n: int, v0: float, hull: np.ndarray | None = None) -> float:
    """Bester erwarteter Nutzen mit n Bietern und Eigenwert v0: Summe über die Segmente der Hülle von max(Steigung, v0) · [(1 − qa)^n − (1 − qb)^n]."""
    h = upper_hull(q, r) if hull is None else hull
    qs, rs = q[h], r[h]
    slopes = np.diff(rs) / np.diff(qs)
    weight = (1.0 - qs[:-1]) ** n - (1.0 - qs[1:]) ** n
    return float(np.sum(np.maximum(slopes, v0) * weight))


def plain_revenue_from_curve(q: np.ndarray, r: np.ndarray, n: int) -> float:
    """Zweitpreis ohne Mindestgebot im Quantilenraum: n (n − 1) ∫ R(q) (1 − q)^(n−2) dq (n ≥ 2; für n = 1 gibt es keinen zweiten Bieter: 0)."""
    if n < 2:
        return 0.0
    return float(n * (n - 1) * trapz(r * (1.0 - q) ** (n - 2), q))


def plain_utility(d: Dist, n: int, v0: float) -> float:
    """Einfache Auktion: Zweitpreis mit Mindestgebot v0 (nicht darunter verkaufen), Nutzen = Erlös + v0 · P(unverkauft)."""
    return seller_utility(d, n, v0, v0, GRID_N)


def optimal_utility_regular(d: Dist, n: int, v0: float) -> float:
    """Bestes Verfahren bei regulärer Verteilung über Weg A: Zweitpreis mit der Myerson-Reserve r* (φ(r*) = v0); Nutzen = Erlös + v0 · F(r*)^n."""
    r_star = optimal_reserve(d, v0)
    return spa_revenue_virtual(d, n, r_star) + v0 * float(d.cdf(r_star)) ** n


def extra_bidders_needed(d: Dist, n: int, v0: float, target: float, limit: int = MAX_EXTRA) -> int | None:
    """Kleinste Zahl k ≥ 1 zusätzlicher Bieter, mit der die einfache Auktion (n + k Bieter) das beste Verfahren mit n Bietern (Nutzen `target`) erreicht;
    None, wenn bis `limit` nicht."""
    for k in range(1, limit + 1):
        if plain_utility(d, n + k, v0) >= target:
            return k
    return None


def participation_threshold(d: Dist, n: int, v0: float, target: float) -> float:
    """Teilnahmewahrscheinlichkeit p* des zusätzlichen Bieters, ab der (1 − p)·Nutzen(n Bieter) + p·Nutzen(n + 1 Bieter) das beste Verfahren erreicht:
    p* = (target − U(n)) / (U(n + 1) − U(n)), U = einfache Auktion. Ist p* > 1, reicht ein zusätzlicher Bieter auch bei sicherer Teilnahme nicht."""
    base, nxt = plain_utility(d, n, v0), plain_utility(d, n + 1, v0)
    if nxt <= base:
        return float("inf")
    return (target - base) / (nxt - base)


def bulow_klemperer_row(d: Dist, n: int, v0: float, q: np.ndarray | None = None, r: np.ndarray | None = None, hull: np.ndarray | None = None) -> dict:
    """Eine Zeile der Bulow–Klemperer-Tabelle: bestes Verfahren mit n Bietern, einfache Auktion mit n + 1, Verhältnis, zusätzliche Bieter, Schwelle p*.
    Das beste Verfahren kommt aus der Hülle im Quantilenraum (bei regulärer Verteilung gegen Weg A geprüft, `optimal_utility_regular`)."""
    if q is None or r is None:
        q, r = revenue_curve(d)
    best = optimal_utility_from_hull(q, r, n, v0, hull)
    plain_next = plain_utility(d, n + 1, v0)
    k = extra_bidders_needed(d, n, v0, best)
    return {"n": n, "best": best, "plain_next": plain_next, "ratio": plain_next / best if best > 0 else float("inf"),
            "holds": plain_next >= best, "extra": k, "p_star": participation_threshold(d, n, v0, best)}
