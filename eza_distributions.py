"""Verteilungen der privaten Werte (Euro je Slot): Dichte, Verteilungsfunktion, Quantilfunktion, virtueller Wert.

Vier Verteilungen, alle ohne scipy: Gleichverteilung U[0, 100], Exponentiell (Mittel 50), Beta(2, 2) auf [0, 100] und eine **Mischung**
(90 % U[0, 10], 10 % U[90, 100]), deren virtueller Wert nicht monoton ist (unregulär). Für die Gitterrechnung hat jede einen endlichen
Träger [lo, hi]; bei der Exponentialverteilung ist hi das Quantil 1 − 1e-10 (Abschneiden, Rest 1e-10 der Wahrscheinlichkeit)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

H = 100.0                          # Obergrenze der begrenzten Verteilungen (Euro je Slot)
EXP_MEAN = 50.0
EXP_TAIL = 1e-10                   # abgeschnittene Wahrscheinlichkeit der Exponentialverteilung
MIX_SPLIT = 0.9                    # Anteil der niedrigen Gruppe der Mischung
MIX_LOW_TOP, MIX_HIGH_BOTTOM = 10.0, 90.0

KINDS = ("gleich", "exp", "beta", "mix")
LABELS = {
    "gleich": "Gleichverteilung U[0, 100]",
    "exp": "Exponentiell (Mittel 50)",
    "beta": "Beta(2, 2) auf [0, 100]",
    "mix": "Mischung (unregulär): 90 % U[0, 10], 10 % U[90, 100]",
}
REGULAR = {"gleich": True, "exp": True, "beta": True, "mix": False}


@dataclass(frozen=True)
class Dist:
    kind: str
    lo: float
    hi: float

    @property
    def name(self) -> str:
        return LABELS[self.kind]

    @property
    def regular(self) -> bool:
        """Regulär: der virtuelle Wert v − (1 − F)/f steigt (Voraussetzung von Bulow–Klemperer und Myerson ohne Glättung)."""
        return REGULAR[self.kind]

    def cdf(self, v):
        v = np.asarray(v, dtype=float)
        if self.kind == "gleich":
            return np.clip(v / H, 0.0, 1.0)
        if self.kind == "exp":
            return np.where(v > 0, -np.expm1(-np.maximum(v, 0.0) / EXP_MEAN), 0.0)
        if self.kind == "beta":
            x = np.clip(v / H, 0.0, 1.0)
            return 3 * x * x - 2 * x ** 3
        low = MIX_SPLIT * np.clip(v / MIX_LOW_TOP, 0.0, 1.0)
        high = (1 - MIX_SPLIT) * np.clip((v - MIX_HIGH_BOTTOM) / (H - MIX_HIGH_BOTTOM), 0.0, 1.0)
        return low + high

    def pdf(self, v):
        v = np.asarray(v, dtype=float)
        if self.kind == "gleich":
            return np.where((v >= 0) & (v <= H), 1.0 / H, 0.0)
        if self.kind == "exp":
            return np.where(v >= 0, np.exp(-np.maximum(v, 0.0) / EXP_MEAN) / EXP_MEAN, 0.0)
        if self.kind == "beta":
            x = v / H
            return np.where((x >= 0) & (x <= 1), 6 * x * (1 - x) / H, 0.0)
        # An den Sprungstellen 10 und 90 gilt der Mittelwert beider Seiten: die Trapezregel auf dem Gitter integriert den Sprung dann
        # ohne Fehler erster Ordnung (die Verteilung selbst ändert das nicht, eine Dichte ist nur fast überall festgelegt).
        low_level, high_level = MIX_SPLIT / MIX_LOW_TOP, (1 - MIX_SPLIT) / (H - MIX_HIGH_BOTTOM)
        low = np.where((v >= 0) & (v < MIX_LOW_TOP), low_level, np.where(v == MIX_LOW_TOP, low_level / 2, 0.0))
        high = np.where((v > MIX_HIGH_BOTTOM) & (v <= H), high_level, np.where(v == MIX_HIGH_BOTTOM, high_level / 2, 0.0))
        return low + high

    def ppf(self, p):
        """Quantilfunktion; für die Mischung am Sprung bei p = 0,9 gilt der untere Ast (Wert 10)."""
        p = np.asarray(p, dtype=float)
        if self.kind == "gleich":
            return H * p
        if self.kind == "exp":
            return -EXP_MEAN * np.log1p(-p)
        if self.kind == "beta":
            # 3x² − 2x³ = p wird mit x = ½ + y zu y³ − ¾ y + (p − ½)/2 = 0; sin 3θ = 3 sin θ − 4 sin³θ löst es: y = sin(arcsin(2p − 1)/3)
            return H * (0.5 + np.sin(np.arcsin(np.clip(2 * p - 1, -1.0, 1.0)) / 3.0))
        return np.where(p <= MIX_SPLIT, MIX_LOW_TOP * p / MIX_SPLIT,
                        MIX_HIGH_BOTTOM + (H - MIX_HIGH_BOTTOM) * (p - MIX_SPLIT) / (1 - MIX_SPLIT))

    def virtual_value(self, v):
        """φ(v) = v − (1 − F(v))/f(v); wo die Dichte 0 ist, ist φ nicht definiert (nan)."""
        v = np.asarray(v, dtype=float)
        f = self.pdf(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0, v - (1.0 - self.cdf(v)) / f, np.nan)

    def phi_times_density(self, v):
        """φ(v)·f(v) = v·f(v) − (1 − F(v)): überall definiert, auch dort, wo f = 0 (Lücke der Mischung)."""
        v = np.asarray(v, dtype=float)
        return v * self.pdf(v) - (1.0 - self.cdf(v))

    def mean(self) -> float:
        return {"gleich": H / 2, "exp": EXP_MEAN, "beta": H / 2, "mix": MIX_SPLIT * MIX_LOW_TOP / 2 + (1 - MIX_SPLIT) * (MIX_HIGH_BOTTOM + H) / 2}[self.kind]


def get_dist(kind: str) -> Dist:
    if kind not in KINDS:
        raise ValueError(f"unbekannte Verteilung: {kind}")
    if kind == "exp":
        return Dist(kind, 0.0, EXP_MEAN * -math.log(EXP_TAIL))
    return Dist(kind, 0.0, H)


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Kumulatives Trapezintegral ∫_{x0}^{x_k} y; gleiche Länge wie y, erster Wert 0."""
    return np.concatenate(([0.0], np.cumsum((y[1:] + y[:-1]) * np.diff(x) / 2.0)))


def suffix_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Kumulatives Trapezintegral von x_k bis zum Ende; letzter Wert 0."""
    c = cumtrapz(y, x)
    return c[-1] - c
