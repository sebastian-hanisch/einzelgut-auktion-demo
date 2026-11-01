"""Kern der Einzelgut-Auktion: ein Slot, n Spediteure mit unabhängigen privaten Werten, Verkäufer mit Eigenwert v0 und Mindestgebot r.

Jede Regel ist eine kleine Einheit mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK 1b). Rechnung ohne scipy: Integrale per Trapez auf einem
Wertegitter von `GRID_N` Punkten (Fehler O(h²), im Test gegen scipy `quad` und gegen Brüche), Nullstellen per Bisektion.

Gleichgewichte (symmetrisch, risikoneutral, Werte unabhängig aus F, Mindestgebot r, F(v)^(n−1) = Gewinnwahrscheinlichkeit eines Bieters mit Wert v):
* Zweitpreis (= englische Auktion): jeder bietet seinen Wert (dominant); Preis = max(r, zweithöchstes Gebot), Zuschlag nur bei Höchstgebot ≥ r.
* Erstpreis: b(v) = v − ∫_r^v (F(x)/F(v))^(n−1) dx (Zuschlag zahlt das eigene Gebot); b(r) = r.
* All-pay (jeder zahlt sein Gebot, ohne Mindestgebot): b(v) = ∫_lo^v x (n−1) F(x)^(n−2) f(x) dx (Bedingung erster Ordnung).
Erlös des Zweitpreises auf zwei Wegen: Weg A über die virtuellen Werte φ(v) = v − (1 − F)/f, Weg B über die Ordnungsstatistik des zweithöchsten Werts."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np

from eza_distributions import Dist, cumtrapz, suffix_trapz
from eza_rng import uniform_block

GRID_N = 20001               # Gitterpunkte für Integrale und Gebotsfunktionen (Fehler der Trapezregel ~ h², h = 0,005 Euro bei [0, 100])
BR_GRID_N = 4001             # gröberes Gitter für das Beste-Antwort-Gitter
BR_BIDS = 6001               # Zahl der geprüften Gebote je Wert
TINY = 1e-290                # darunter ist F^(n−1) zu klein für einen Quotienten


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezregel ∫ y dx (eigene Fassung, unabhängig von der numpy-Version)."""
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


# ----------------------------------------------------------------------------------------------- Einheiten für Aufzählung (Handrechnung)
def second_price_outcome(bids, r):
    """Zweitpreis mit Mindestgebot r: (Gewinner, Preis). Gewinner ist der erste mit dem Höchstgebot (Gleichstand: kleinster Index);
    kein Zuschlag (None, 0), wenn das Höchstgebot unter r liegt. Preis = max(r, zweithöchstes Gebot)."""
    if not bids:
        return None, 0
    top = max(bids)
    if top < r:
        return None, 0
    winner = bids.index(top)
    rest = sorted(bids[:winner] + bids[winner + 1:], reverse=True)
    second = rest[0] if rest else r
    return winner, max(r, second)


def spa_revenue_enum(support, n, r):
    """Zweitpreis-Erlös bei n Bietern, deren Werte unabhängig und gleichverteilt aus der endlichen Menge `support` stammen: Aufzählung aller
    |support|^n Werteprofile, exakt in Brüchen (die Handrechnung der Mini-Instanz)."""
    support = [Fraction(s) for s in support]
    r = Fraction(r)
    total = Fraction(0)
    profiles = list(product(support, repeat=n))
    for prof in profiles:
        _, price = second_price_outcome(list(prof), r)
        total += price
    return total / len(profiles)


def first_price_utility(bid, value, win_probability):
    """Erwartete Auszahlung im Erstpreis: (Wert − eigenes Gebot) mal Gewinnwahrscheinlichkeit."""
    return (value - bid) * win_probability


# ----------------------------------------------------------------------------------------------- Gitter
def value_grid(d: Dist, r: float = 0.0, n_points: int = GRID_N) -> np.ndarray:
    """Wertegitter von max(r, lo) bis hi (die Wertebereiche, in denen ein Bieter mit Wert ≥ r teilnimmt)."""
    r0 = max(float(r), d.lo)
    return np.linspace(r0, d.hi, n_points) if r0 < d.hi else np.array([d.hi, d.hi])


def win_probability_power(d: Dist, n: int, vs: np.ndarray) -> np.ndarray:
    """F(v)^(n−1): Wahrscheinlichkeit, dass alle n − 1 Konkurrenten einen kleineren Wert haben."""
    return d.cdf(vs) ** (n - 1)


# ----------------------------------------------------------------------------------------------- Gebotsfunktionen
def fpa_bid_function(d: Dist, n: int, r: float = 0.0, n_points: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
    """Gleichgewichtsgebot des Erstpreises auf dem Gitter: b(v) = v − ∫_r^v (F(x)/F(v))^(n−1) dx. Rückgabe (Werte, Gebote)."""
    vs = value_grid(d, r, n_points)
    fp = win_probability_power(d, n, vs)
    integral = cumtrapz(fp, vs)
    with np.errstate(divide="ignore", invalid="ignore"):
        shade = np.where(fp > TINY, integral / fp, 0.0)
    return vs, vs - shade


def allpay_bid_function(d: Dist, n: int, n_points: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
    """Gleichgewichtsgebot der All-pay-Auktion ohne Mindestgebot: b(v) = ∫_lo^v x (n−1) F(x)^(n−2) f(x) dx (n ≥ 2; für n = 1 ist es 0)."""
    vs = value_grid(d, 0.0, n_points)
    if n < 2:
        return vs, np.zeros_like(vs)
    integrand = vs * (n - 1) * d.cdf(vs) ** (n - 2) * d.pdf(vs)
    return vs, cumtrapz(integrand, vs)


def crra_factor(n: int, rho: float) -> float:
    """Erstpreis-Gebotsfaktor bei Risikoscheu u(x) = x^ρ und Gleichverteilung: b(v) = (n−1)/(n−1+ρ) · v. Bei ρ = 1 der risikoneutrale (n−1)/n."""
    return (n - 1) / (n - 1 + rho)


# ----------------------------------------------------------------------------------------------- Erlöse
def sale_probability(d: Dist, n: int, r: float) -> float:
    """Wahrscheinlichkeit, dass der Slot verkauft wird: mindestens ein Wert ≥ r."""
    return float(1.0 - d.cdf(max(float(r), d.lo)) ** n)


def spa_revenue_virtual(d: Dist, n: int, r: float = 0.0, n_points: int = GRID_N) -> float:
    """Weg A: Zweitpreis-Erlös = n ∫_r^hi F^(n−1) (v f − (1 − F)) dv, also der erwartete virtuelle Wert des Höchsten, wenn er ≥ r ist."""
    vs = value_grid(d, r, n_points)
    return trapz(n * d.cdf(vs) ** (n - 1) * d.phi_times_density(vs), vs)


def spa_revenue_order(d: Dist, n: int, r: float = 0.0, n_points: int = GRID_N) -> float:
    """Weg B: Preis = max(r, zweithöchster Wert), nur wenn der Höchste ≥ r: r·P(genau einer ≥ r) + E[V(2) · 1{V(2) ≥ r}]."""
    r0 = max(float(r), d.lo)
    fr = float(d.cdf(r0))
    only_one = n * fr ** (n - 1) * (1.0 - fr)
    if n < 2 or r0 >= d.hi:
        return r0 * only_one
    vs = value_grid(d, r0, n_points)
    second = trapz(vs * n * (n - 1) * d.cdf(vs) ** (n - 2) * (1.0 - d.cdf(vs)) * d.pdf(vs), vs)
    return r0 * only_one + second


def fpa_revenue(d: Dist, n: int, r: float = 0.0, n_points: int = GRID_N) -> float:
    """Erstpreis-Erlös = n ∫_r^hi b(v) F(v)^(n−1) f(v) dv (erwartetes Höchstgebot, Bieter unter r bieten nicht)."""
    vs, bs = fpa_bid_function(d, n, r, n_points)
    return trapz(n * bs * d.cdf(vs) ** (n - 1) * d.pdf(vs), vs)


def allpay_revenue(d: Dist, n: int, n_points: int = GRID_N) -> float:
    """All-pay-Erlös = n ∫ b(v) f(v) dv (jeder zahlt sein Gebot)."""
    vs, bs = allpay_bid_function(d, n, n_points)
    return trapz(n * bs * d.pdf(vs), vs)


def seller_utility(d: Dist, n: int, r: float, v0: float, n_points: int = GRID_N) -> float:
    """Nutzen des Verkäufers = Zweitpreis-Erlös (= Erstpreis-Erlös) bei Mindestgebot r + v0 · P(unverkauft)."""
    return spa_revenue_order(d, n, r, n_points) + v0 * (1.0 - sale_probability(d, n, r))


def spa_revenue_curve(d: Dist, n: int, n_points: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
    """Erlös des Zweitpreises (Weg B) für jedes Mindestgebot r auf dem Gitter in einem Durchgang (Suffix-Integral). Rückgabe (r, Erlös)."""
    vs = value_grid(d, 0.0, n_points)
    fr = d.cdf(vs)
    only_one = n * fr ** (n - 1) * (1.0 - fr)
    if n < 2:
        return vs, vs * only_one
    second = suffix_trapz(vs * n * (n - 1) * fr ** (n - 2) * (1.0 - fr) * d.pdf(vs), vs)
    return vs, vs * only_one + second


def seller_utility_curve(d: Dist, n: int, v0: float, n_points: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
    """U(r) für jedes Mindestgebot r auf dem Gitter: Erlös + v0 · F(r)^n."""
    rs, rev = spa_revenue_curve(d, n, n_points)
    return rs, rev + v0 * d.cdf(rs) ** n


def best_reserve_on_grid(d: Dist, n: int, v0: float, n_points: int = GRID_N) -> tuple[float, float]:
    """Gittersuche: das Mindestgebot mit größtem Verkäufernutzen und dieser Nutzen (bei Gleichstand das kleinste r)."""
    rs, u = seller_utility_curve(d, n, v0, n_points)
    k = int(np.argmax(u))
    return float(rs[k]), float(u[k])


def optimal_reserve(d: Dist, v0: float) -> float:
    """Myerson-Reserve bei regulärer Verteilung: das r mit φ(r) = v0 (Bisektion; bei nicht definiertem φ, f = 0, gilt φ = −∞).
    Liegt φ am unteren Rand schon über v0, ist die Reserve der untere Rand des Trägers."""
    if not d.regular:
        raise ValueError("Die Reserve aus φ(r) = v0 gilt nur bei regulärer Verteilung (φ steigt).")

    def ok(r):
        phi = d.virtual_value(r)
        return bool(np.isfinite(phi) and phi >= v0)

    lo, hi = d.lo, d.hi
    if ok(lo + 1e-12):
        return lo
    for _ in range(200):
        mid = (lo + hi) / 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def efficient_value(d: Dist, n: int, v0: float, n_points: int = GRID_N) -> float:
    """E[max(V(1), v0)]: Wert, wenn der Slot immer an den höchsten Wert geht, sofern dieser über dem Eigenwert des Verkäufers liegt."""
    vs = value_grid(d, 0.0, n_points)
    dens = n * d.cdf(vs) ** (n - 1) * d.pdf(vs)
    return trapz(np.maximum(vs, v0) * dens, vs)


def welfare_loss(d: Dist, n: int, r: float, v0: float, n_points: int = GRID_N) -> float:
    """Wohlfahrtsverlust des Mindestgebots r: E[max(V(1), v0)] minus erreichter Wert (Zuschlag an den Höchsten, wenn V(1) ≥ r, sonst bleibt v0)."""
    def part(gain, a, b):
        if b <= a:
            return 0.0
        vs = np.linspace(a, b, n_points)
        return trapz(gain(vs) * n * d.cdf(vs) ** (n - 1) * d.pdf(vs), vs)

    # unter r bleibt der Slot beim Verkäufer (Verlust, wenn der Höchstwert über v0 liegt); ab r geht er an den Höchsten (Verlust, wenn dessen Wert unter v0 liegt)
    r_in = min(max(float(r), d.lo), d.hi)
    return part(lambda v: np.maximum(v - v0, 0.0), d.lo, r_in) + part(lambda v: np.maximum(v0 - v, 0.0), r_in, d.hi)


# ----------------------------------------------------------------------------------------------- Beste-Antwort-Gitter (Orakel des Gleichgewichts)
def fpa_best_response_gap(d: Dist, n: int, r: float, v: float, opponent=None, own_bid: float | None = None) -> float:
    """Gegner bieten die Gleichgewichtsgebote (oder `opponent = (Werte, Gebote)`); Gewinnwahrscheinlichkeit eines Gebots x ist F(b⁻¹(x))^(n−1).
    Geprüft wird das Gebot `own_bid` (Standard: das Gleichgewichtsgebot b(v)). Rückgabe: beste Auszahlung über `BR_BIDS` Gebote minus Auszahlung
    des geprüften Gebots, mindestens 0 (ideal ≈ 0; bei einer falschen Gegnerstrategie deutlich größer: Negativkontrolle)."""
    vs, bs = fpa_bid_function(d, n, r, BR_GRID_N) if opponent is None else opponent
    fp = d.cdf(vs) ** (n - 1)
    x_eq = float(np.interp(v, vs, bs)) if own_bid is None else float(own_bid)
    u_eq = first_price_utility(x_eq, v, float(np.interp(x_eq, bs, fp)))
    xs = np.linspace(bs[0], min(v, bs[-1]), BR_BIDS)
    u = (v - xs) * np.interp(xs, bs, fp)
    return float(max(u.max() - u_eq, 0.0))


def allpay_best_response_gap(d: Dist, n: int, v: float) -> float:
    """All-pay: Auszahlung eines Gebots x ist v·F(b⁻¹(x))^(n−1) − x; Lücke wie beim Erstpreis."""
    vs, bs = allpay_bid_function(d, n, BR_GRID_N)
    fp = d.cdf(vs) ** (n - 1)
    x_eq = float(np.interp(v, vs, bs))
    u_eq = v * float(np.interp(x_eq, bs, fp)) - x_eq
    xs = np.linspace(0.0, bs[-1], BR_BIDS)
    u = v * np.interp(xs, bs, fp) - xs
    return float(max(u.max() - u_eq, 0.0))


def crra_best_response_gap(n: int, rho: float, v: float, h: float = 100.0, n_bids: int = 20001) -> float:
    """Risikoscheu u(x) = x^ρ, Gleichverteilung U[0, h]: Gegner bieten λ·w. Auszahlung des Gebots x: (v − x)^ρ · (x/(λ h))^(n−1)."""
    if n < 2:
        return 0.0                  # ohne Konkurrenten gewinnt jedes Gebot; das Gleichgewichtsgebot 0 ist die Beste Antwort
    lam = crra_factor(n, rho)
    xs = np.linspace(0.0, v, n_bids)
    u = (v - xs) ** rho * np.minimum(xs / (lam * h), 1.0) ** (n - 1)
    x_f = lam * v
    u_f = (v - x_f) ** rho * min(x_f / (lam * h), 1.0) ** (n - 1)
    return float(max(u.max() - u_f, 0.0))


# ----------------------------------------------------------------------------------------------- Simulation (SplitMix64)
def simulate_revenue(d: Dist, n: int, fmt: str, seed: int, m: int, r: float = 0.0, n_points: int = BR_GRID_N) -> float:
    """Mittlerer Erlös über `m` simulierte Auktionen; Werte aus dem SplitMix64-Strom `seed` über die Quantilfunktion. fmt: 'zweit', 'erst', 'allpay'."""
    values = d.ppf(uniform_block(seed, m * n).reshape(m, n))
    if fmt == "zweit":
        srt = np.sort(values, axis=1)
        top = srt[:, -1]
        second = srt[:, -2] if n >= 2 else np.full(m, -np.inf)
        return float(np.mean(np.where(top >= r, np.maximum(r, second), 0.0)))
    if fmt == "erst":
        vs, bs = fpa_bid_function(d, n, r, n_points)
        bids = np.where(values >= max(r, d.lo), np.interp(values, vs, bs), -np.inf)
        top = bids.max(axis=1)
        return float(np.mean(np.where(np.isfinite(top), top, 0.0)))
    if fmt == "allpay":
        vs, bs = allpay_bid_function(d, n, n_points)
        return float(np.mean(np.interp(values, vs, bs).sum(axis=1)))
    raise ValueError(f"unbekanntes Format: {fmt}")


def simulate_crra_revenue(n: int, rho: float, seed: int, m: int, h: float = 100.0) -> float:
    """Erstpreis bei Risikoscheu, U[0, h]: Erlös = λ · Höchstwert (jeder bietet λ·v)."""
    values = h * uniform_block(seed, m * n).reshape(m, n)
    return float(np.mean(crra_factor(n, rho) * values.max(axis=1)))
