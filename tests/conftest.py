"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte sind von Hand gerechnet (Rechnung im Docstring)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def values_123():
    """Werte 1, 2, 3 Euro, jeder mit Wahrscheinlichkeit 1/3, zwei Bieter: 9 Werteprofile.

    Zweitpreis ohne Mindestgebot: Preise der 9 Paare (1,1)1 (1,2)1 (1,3)1 (2,1)1 (2,2)2 (2,3)2 (3,1)1 (3,2)2 (3,3)3, Summe 14, also 14/9.
    Mit Mindestgebot 2: (1,1) bleibt unverkauft (Höchstgebot 1 < 2), alle anderen Preise sind mindestens 2: (1,2)2 (1,3)2 (2,1)2 (2,2)2 (2,3)2 (3,1)2 (3,2)2 (3,3)3,
    Summe 17, also 17/9; unverkauft 1 von 9 Paaren."""
    return {"support": [1, 2, 3], "n": 2, "revenue_plain": F(14, 9), "reserve": 2, "revenue_reserve": F(17, 9), "unsold": F(1, 9)}


@pytest.fixture
def uniform_hand():
    """U[0, 100] Euro, zwei Bieter (Einheit 100 Euro, Handrechnung mit Brüchen).

    Zweitpreis ohne Mindestgebot: E[min] = 1/3 (33,33 Euro). Erstpreis: Gebot v/2, Erlös E[max]/2 = (2/3)/2 = 1/3. All-pay: Gebot v²/2, Erlös 2 E[v²/2] = 1/3.
    Mindestgebot 1/2: (n−1)/(n+1) + r^n − 2n r^(n+1)/(n+1) = 1/3 + 1/4 − 1/6 = 5/12 (41,67 Euro). Reserve φ(r) = 2r − 1 = v0: r* = (1 + v0)/2, also 1/2 bei v0 = 0, 3/4 bei v0 = 1/2.
    Nutzen bei v0 = 20 Euro, r = 50: 5/12 + 0,2 · P(unverkauft) = 5/12 + 0,2 · 1/4 = 5/12 + 1/20 = 14/30 (46,67 Euro).
    Bulow–Klemperer, n = 1: bester Festpreis 1/2 mit Erlös 1/4 (25 Euro); zwei Bieter ohne Mindestgebot 1/3; Schwelle p* = (1/4)/(1/3) = 3/4."""
    return {"revenue_plain": F(1, 3), "revenue_reserve": F(5, 12), "reserve_v0_0": F(1, 2), "reserve_v0_half": F(3, 4), "utility_20": F(14, 30),
            "bk_best": F(1, 4), "bk_next": F(1, 3), "bk_threshold": F(3, 4)}


@pytest.fixture
def asymmetry_half():
    """w = 1/2 (zweiter Bieter mit Werten bis 50 Euro): c = (1/4 − 1)/(1/4) = −3, höchstes Gebot b̄ = (1/2)/(3/2) = 1/3, x(b̄) = (2/3)/(1 − 3/9) = 1, y(b̄) = (2/3)/(1 + 3/9) = 1/2.
    Zweitpreis-Erlös w/2 − w²/6 = 1/4 − 1/24 = 5/24 (20,83 Euro). Myerson: 5w/8 − 5w²/24 + (1 − w)²/4 = 5/16 − 5/96 + 1/16 = 31/96 (32,29 Euro)."""
    return {"c": F(-3), "top": F(1, 3), "x_top": F(1), "y_top": F(1, 2), "second": F(5, 24), "myerson": F(31, 96)}


@pytest.fixture
def risk_half():
    """Risikoscheu ρ = 1/2, n = 2: Gebotsfaktor (n−1)/(n−1+ρ) = 1/(3/2) = 2/3; Erstpreis-Erlös 2/3 · 2/3 = 4/9 gegen Zweitpreis 1/3, Verhältnis 4/3."""
    return {"factor": F(2, 3), "first": F(4, 9), "second": F(1, 3), "ratio": F(4, 3)}
