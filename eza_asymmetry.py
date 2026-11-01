"""Asymmetrische Bieter im Erstpreis: zwei Spediteure, Wert 1 ~ U[0, 100] (stark), Wert 2 ~ U[0, 100·w] (schwach), 0 < w ≤ 1.

Geschlossenes Gleichgewicht (Fall zweier Gleichverteilter mit Untergrenze 0, Ansatz; gerechnet in Einheiten von 100 Euro, x = Wert des starken, y = Wert des schwachen Bieters
als Funktion des Gebots b): x(b) = 2b/(1 + c·b²), y(b) = 2b/(1 − c·b²), c = (w² − 1)/w², höchstes Gebot b̄ = w/(1 + w). Beide lösen
x′(y − b) = x und y′(x − b) = y mit x(b̄) = 1, y(b̄) = w (die Bedingungen erster Ordnung beider Bieter). Für w = 1 ist c = 0 und beide bieten v/2.
Das Zweitpreis-Gebot bleibt der Wert. Erlöse in Euro (mal 100)."""

from __future__ import annotations

import numpy as np

H = 100.0
ASYM_N = 20001


def asym_constant(w: float) -> float:
    """c = (w² − 1)/w²; 0 bei w = 1, sonst negativ."""
    return (w * w - 1.0) / (w * w)


def top_bid(w: float) -> float:
    """Höchstes Gebot beider Bieter b̄ = w/(1 + w), in Einheiten von 100 Euro."""
    return w / (1.0 + w)


def inverse_strong(b, w: float):
    """Wert des starken Bieters (Einheiten 100 Euro) zum Gebot b: x(b) = 2b/(1 + c b²)."""
    b = np.asarray(b, dtype=float)
    return 2 * b / (1 + asym_constant(w) * b * b)


def inverse_weak(b, w: float):
    """Wert des schwachen Bieters (Einheiten 100 Euro) zum Gebot b: y(b) = 2b/(1 − c b²)."""
    b = np.asarray(b, dtype=float)
    return 2 * b / (1 - asym_constant(w) * b * b)


def bid_strong(v, w: float):
    """Gleichgewichtsgebot des starken Bieters bei Wert v (Einheiten 100 Euro): (1 − √(1 − c v²))/(c v); für c = 0 oder v = 0 ist es v/2."""
    v = np.asarray(v, dtype=float)
    c = asym_constant(w)
    if abs(c) < 1e-14:
        return v / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, (1 - np.sqrt(1 - c * v * v)) / (c * v), 0.0)


def bid_weak(v, w: float):
    """Gleichgewichtsgebot des schwachen Bieters bei Wert v (Einheiten 100 Euro): (−1 + √(1 + c v²))/(c v); für c = 0 oder v = 0 ist es v/2."""
    v = np.asarray(v, dtype=float)
    c = asym_constant(w)
    if abs(c) < 1e-14:
        return v / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, (-1 + np.sqrt(1 + c * v * v)) / (c * v), 0.0)


def _trapz(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def second_price_revenue(w: float) -> float:
    """Zweitpreis-Erlös E[min(v1, v2)] = w/2 − w²/6 (Einheiten 100 Euro), mal 100 in Euro. Herleitung: ∫ P(min > t) dt = ∫₀^w (1 − t)(1 − t/w) dt."""
    return H * (w / 2 - w * w / 6)


def first_price_revenue(w: float, n_points: int = ASYM_N) -> float:
    """Erstpreis-Erlös E[max Gebot] = b̄ − ∫₀^b̄ P(b₁ ≤ b)·P(b₂ ≤ b) db = b̄ − ∫₀^b̄ x(b)·y(b)/w db (Euro)."""
    bb = top_bid(w)
    bs = np.linspace(0.0, bb, n_points)
    return H * (bb - _trapz(inverse_strong(bs, w) * inverse_weak(bs, w) / w, bs))


def myerson_revenue(w: float) -> float:
    """Erlös des Myerson-optimalen Verfahrens (Mindestgebote je Bieter, φ₁ = 2v − 1, φ₂ = 2v − w, Zuschlag an den größten positiven virtuellen Wert,
    Schwellenzahlung): ∫₀¹ (1 − P₁(z)·P₂(z)) dz mit P₁ = min(1, (1 + z)/2), P₂ = min(1, (w + z)/(2w)); geschlossen 5w/8 − 5w²/24 + (1 − w)²/4 (Euro)."""
    return H * (5 * w / 8 - 5 * w * w / 24 + (1 - w) ** 2 / 4)


def first_price_misallocation(w: float, n_points: int = ASYM_N) -> tuple[float, float]:
    """Erstpreis: (Wahrscheinlichkeit, dass der Zuschlag nicht an den höheren Wert geht; Wohlfahrtsverlust in Anteilen des effizienten Werts E[max(v1, v2)]).
    Der schwache Bieter gewinnt mit Wert v2 genau dann, wenn v1 < x(b₂(v2)); falsch ist es für v1 ∈ (v2, x(b₂(v2))) (der starke hätte mehr), Verlust (x − v2)²/2."""
    v2 = np.linspace(0.0, w, n_points)
    x_at = inverse_strong(bid_weak(v2, w), w)
    gap = np.maximum(x_at - v2, 0.0)
    prob = _trapz(gap, v2) / w
    loss = _trapz(gap * gap / 2, v2) / w
    return prob, loss / (0.5 + w * w / 6)


def myerson_welfare_loss(w: float, n_points: int = ASYM_N) -> float:
    """Wohlfahrtsverlust des Myerson-optimalen Verfahrens in Anteilen von E[max(v1, v2)] = ½ + w²/6 (Slot bleibt bei kleinen Werten unverkauft,
    der Zuschlag folgt den virtuellen Werten). Erreichter Wert: Bieter 1 gewinnt für v₁ ≥ max(½, v₂ + (1 − w)/2), Bieter 2 für v₂ > w/2 und v₁ < v₂ + (1 − w)/2."""
    v2 = np.linspace(0.0, w, n_points)
    t = np.maximum(0.5, v2 + (1 - w) / 2)
    one = _trapz((1 - t * t) / 2, v2) / w
    s = np.linspace(w / 2, w, n_points)
    two = _trapz(s * (s + (1 - w) / 2), s) / w
    efficient = 0.5 + w * w / 6
    return 1.0 - (one + two) / efficient
