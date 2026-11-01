"""Szenario: die Einstellungen der Seitenleiste als unveränderliches Objekt (Schlüssel für den Zwischenspeicher)."""

from __future__ import annotations

from dataclasses import dataclass

import eza_constants as C


@dataclass(frozen=True)
class Settings:
    dist: str = C.DIST_DEFAULT
    n: int = C.DEFAULT_N
    r: float = float(C.DEFAULT_R)
    v0: float = float(C.DEFAULT_V0)
    rho: float = C.DEFAULT_RHO
    w: float = C.DEFAULT_W
    rule: str = "Hedge"
    fmt: str = "Erstpreis"
    n_learn: int = 2
    rounds: int = C.DEFAULT_ROUNDS
    rate: float = C.DEFAULT_RATE
    seed: int = C.DEFAULT_SEED
