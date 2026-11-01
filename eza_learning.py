"""Lernende Bieter: n Spediteure lernen Gebote b = α·v (Wert v gleichverteilt auf [0, 1], 1 = 100 Euro) über wiederholte Auktionen.

Drei Lernregeln, jede eine Einheit: **Hedge** (volle Rückmeldung: nach jeder Runde sieht der Bieter, was jedes α gebracht hätte), **Exp3** (nur die eigene
Auszahlung wird beobachtet) und **tabellarisches Q-Lernen** je Wertklasse (nur eigene Auszahlung, ε-greedy). Das Gleichgewicht ist im Erstpreis α = (n − 1)/n, im
Zweitpreis α = 1 (schwach dominant). Zufall nur über SplitMix64 (`eza_rng`), vorab als Block gezogen. Die Auszahlungen liegen in [0, 1]."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from eza_rng import uniform_block

K_ALPHAS = 21
ALPHAS = np.linspace(0.0, 1.0, K_ALPHAS)            # Raster der Gebotsfaktoren: Schritt 0,05
Q_ACTIONS = np.linspace(0.0, 1.0, 11)               # Q-Lernen: Schritt 0,1
Q_CLASSES = 10                                      # Wertklassen [k/10, (k+1)/10)
Q_FIRST_CLASS = 3                                   # ausgewertet werden die Klassen ab Wert 0,3 (kleine Werte tragen kaum zur Auszahlung bei)
Q_BASE_RATE = 0.05
RULES = ("Hedge", "Exp3", "Q-Lernen")
FORMATS = ("Erstpreis", "Zweitpreis")
CHECKPOINTS = 100


def equilibrium_alpha(n: int, fmt: str) -> float:
    """Gebotsfaktor im Gleichgewicht: Erstpreis (n − 1)/n, Zweitpreis 1."""
    return (n - 1) / n if fmt == "Erstpreis" else 1.0


def equilibrium_revenue(n: int) -> float:
    """Gleichgewichtserlös beider Formate bei U[0, 1] (Erlösäquivalenz): (n − 1)/(n + 1)."""
    return (n - 1) / (n + 1)


def auction_outcome(bids: np.ndarray, fmt: str) -> tuple[int, float, float, float]:
    """Eine Auktion: (Gewinner, höchstes Gebot, zweithöchstes Gebot, Erlös). Gleichstand: der kleinste Index gewinnt."""
    win = int(np.argmax(bids))
    top = float(bids[win])
    second = float(np.partition(bids, -2)[-2])
    return win, top, second, top if fmt == "Erstpreis" else second


def payoff_all_alphas(value: float, rival_top: float, fmt: str) -> np.ndarray:
    """Auszahlung jedes Rasterwerts α gegen das höchste Gegnergebot (Hedge-Rückmeldung): gewonnen, wenn α·v > rival_top (Gleichstand verloren)."""
    bid = ALPHAS * value
    win = bid > rival_top
    price = bid if fmt == "Erstpreis" else rival_top
    return np.where(win, value - price, 0.0)


def hedge_probabilities(losses: np.ndarray, eta: float) -> np.ndarray:
    """Hedge-Gewichte exp(−η · Verlust), je Bieter normiert (Verluste um das Minimum verschoben, damit nichts überläuft)."""
    w = np.exp(-eta * (losses - losses.min(axis=1, keepdims=True)))
    return w / w.sum(axis=1, keepdims=True)


def exp3_probabilities(scores: np.ndarray, gamma: float) -> np.ndarray:
    """Exp3-Wahrscheinlichkeiten (1 − γ)·softmax(Summe) + γ/K."""
    w = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (1 - gamma) * w / w.sum(axis=1, keepdims=True) + gamma / scores.shape[1]


def draw_index(p_row: np.ndarray, u: float) -> int:
    """Index aus der Verteilung p_row per Inversion mit der Gleichverteilten u."""
    return int(min(np.searchsorted(np.cumsum(p_row), u, side="right"), len(p_row) - 1))


def q_update(q: float, payoff: float, rate: float) -> float:
    """Q-Lernschritt: Q ← Q + Rate · (Auszahlung − Q)."""
    return q + rate * (payoff - q)


def _result(n, fmt, last_alpha, played, revenue, trajectory, extra=None):
    out = {"n": n, "fmt": fmt, "alpha_last": float(last_alpha), "alpha_played": float(played), "revenue": float(revenue),
           "alpha_eq": equilibrium_alpha(n, fmt), "revenue_eq": equilibrium_revenue(n), "trajectory": [float(x) for x in trajectory]}
    if extra:
        out.update(extra)
    return out


def hedge_run(n: int, fmt: str, rounds: int, seed: int, rate_factor: float = 1.0, eta: float | None = None) -> dict:
    """Hedge mit voller Rückmeldung. Rate η = rate_factor · √(8 ln K / T) (Standardrate), oder fest `eta`. Ergebnis: letzter Iterierter (mittleres α der
    Mischstrategien), gespielte α und Erlös im letzten Viertel, Verlauf des mittleren α an 100 Stützstellen."""
    u = uniform_block(seed, rounds * 2 * n).reshape(rounds, 2, n)
    eta = rate_factor * float(np.sqrt(8 * np.log(K_ALPHAS) / rounds)) if eta is None else eta
    losses = np.zeros((n, K_ALPHAS))
    played, revenue = np.zeros(rounds), np.zeros(rounds)
    step = max(1, rounds // CHECKPOINTS)
    traj = []
    for t in range(rounds):
        p = hedge_probabilities(losses, eta)
        if t % step == 0:
            traj.append(float((p @ ALPHAS).mean()))
        values = u[t, 0]
        idx = np.minimum((np.cumsum(p, axis=1) <= u[t, 1][:, None]).sum(axis=1), K_ALPHAS - 1)
        bids = ALPHAS[idx] * values
        win, top, second, rev = auction_outcome(bids, fmt)
        revenue[t], played[t] = rev, ALPHAS[idx].mean()
        for i in range(n):
            rival = second if i == win else top
            losses[i] += 1.0 - payoff_all_alphas(float(values[i]), rival, fmt)
    last = (hedge_probabilities(losses, eta) @ ALPHAS).mean()
    quarter = slice(-max(1, rounds // 4), None)
    return _result(n, fmt, last, played[quarter].mean(), revenue[quarter].mean(), traj, {"rule": "Hedge", "eta": eta})


def exp3_run(n: int, fmt: str, rounds: int, seed: int, rate_factor: float = 1.0) -> dict:
    """Exp3: nur die eigene Auszahlung des gespielten α wird beobachtet (importance-gewichtet). γ = min(1, rate_factor · √(K ln K / ((e − 1) T)))."""
    u = uniform_block(seed, rounds * 2 * n).reshape(rounds, 2, n)
    gamma = min(1.0, rate_factor * float(np.sqrt(K_ALPHAS * np.log(K_ALPHAS) / ((np.e - 1) * rounds))))
    scores = np.zeros((n, K_ALPHAS))
    played, revenue = np.zeros(rounds), np.zeros(rounds)
    step = max(1, rounds // CHECKPOINTS)
    traj = []
    for t in range(rounds):
        p = exp3_probabilities(scores, gamma)
        if t % step == 0:
            traj.append(float((p @ ALPHAS).mean()))
        values = u[t, 0]
        idx = np.minimum((np.cumsum(p, axis=1) <= u[t, 1][:, None]).sum(axis=1), K_ALPHAS - 1)
        bids = ALPHAS[idx] * values
        win, top, second, rev = auction_outcome(bids, fmt)
        revenue[t], played[t] = rev, ALPHAS[idx].mean()
        payoff = float(values[win]) - (top if fmt == "Erstpreis" else second)
        scores[win, idx[win]] += gamma / K_ALPHAS * (payoff / p[win, idx[win]])
    last = (exp3_probabilities(scores, gamma) @ ALPHAS).mean()
    quarter = slice(-max(1, rounds // 4), None)
    return _result(n, fmt, last, played[quarter].mean(), revenue[quarter].mean(), traj, {"rule": "Exp3", "gamma": gamma})


def q_run(n: int, fmt: str, rounds: int, seed: int, rate_factor: float = 1.0, evaluation_auctions: int = 100_000) -> dict:
    """Tabellarisches Q-Lernen je Wertklasse (10 Klassen, 11 Aktionen), ε-greedy mit ε = max(0,01; exp(−5 t / T)), Lernrate min(1, 0,05 · rate_factor).
    Ergebnis: mittleres α der gierigen Strategie in den Klassen ab Wert 0,3, und der Erlös dieser Strategie in `evaluation_auctions` frischen Auktionen."""
    rate = min(1.0, Q_BASE_RATE * rate_factor)
    k_act = len(Q_ACTIONS)
    u = uniform_block(seed, rounds * 3 * n).reshape(rounds, 3, n)
    q = [[[0.0] * k_act for _ in range(Q_CLASSES)] for _ in range(n)]
    step = max(1, rounds // CHECKPOINTS)
    traj = []
    decay = 5.0 / rounds
    for t in range(rounds):
        if t % step == 0:
            traj.append(float(np.mean([Q_ACTIONS[int(np.argmax(q[i][k]))] for i in range(n) for k in range(Q_FIRST_CLASS, Q_CLASSES)])))
        eps = max(0.01, float(np.exp(-decay * t)))
        values = u[t, 0]
        classes = np.minimum((values * Q_CLASSES).astype(int), Q_CLASSES - 1)
        acts = []
        for i in range(n):
            if u[t, 1, i] < eps:
                acts.append(min(int(u[t, 2, i] * k_act), k_act - 1))
            else:
                acts.append(int(np.argmax(q[i][classes[i]])))
        bids = np.array([Q_ACTIONS[a] * values[i] for i, a in enumerate(acts)])
        win, top, second, _ = auction_outcome(bids, fmt)
        for i in range(n):
            pay = (float(values[i]) - (top if fmt == "Erstpreis" else second)) if i == win else 0.0
            cell = q[i][int(classes[i])]
            cell[acts[i]] = q_update(cell[acts[i]], pay, rate)
    greedy = np.array([[Q_ACTIONS[int(np.argmax(q[i][k]))] for k in range(Q_CLASSES)] for i in range(n)])
    values = uniform_block(seed + 7777, evaluation_auctions * n).reshape(evaluation_auctions, n)
    cls = np.minimum((values * Q_CLASSES).astype(int), Q_CLASSES - 1)
    alpha = np.stack([greedy[i][cls[:, i]] for i in range(n)], axis=1)
    bids = np.sort(alpha * values, axis=1)
    revenue = float(np.mean(bids[:, -1] if fmt == "Erstpreis" else bids[:, -2]))
    last = greedy[:, Q_FIRST_CLASS:].mean()
    return _result(n, fmt, last, last, revenue, traj, {"rule": "Q-Lernen", "rate": rate})


def run(rule: str, n: int, fmt: str, rounds: int, seed: int, rate_factor: float = 1.0) -> dict:
    """Ein Lernlauf mit der gewählten Regel."""
    if rule == "Hedge":
        return hedge_run(n, fmt, rounds, seed, rate_factor)
    if rule == "Exp3":
        return exp3_run(n, fmt, rounds, seed, rate_factor)
    if rule == "Q-Lernen":
        return q_run(n, fmt, rounds, seed, rate_factor)
    raise ValueError(f"unbekannte Lernregel: {rule}")


# ----------------------------------------------------------------------------------------------- Klassenspiel b = α·v (exakt in Brüchen)
def class_game_payoff(a: Fraction, lam: Fraction) -> Fraction:
    """Erwartete Erstpreis-Auszahlung bei n = 2, U[0, 1]: Bieter spielt b = a·v, Gegner b = λ·v'. Für a < λ ist sie (1 − a)·a/(3λ), für a ≥ λ ist sie
    (1 − a)·(1/2 − λ²/(6a²)); a = 0 zahlt 0. (Integral über v der Gewinnwahrscheinlichkeit min(1, a v/λ) mal (1 − a) v.)"""
    if a == 0:
        return Fraction(0)
    if a < lam:
        return (1 - a) * a / (3 * lam)
    return (1 - a) * (Fraction(1, 2) - lam * lam / (6 * a * a))


def class_best_reply(lam: Fraction, grid: list[Fraction]) -> Fraction:
    """Beste Antwort im Klassenspiel auf Gegner-λ über das Raster `grid` (bei Gleichstand der kleinere Faktor)."""
    return max(grid, key=lambda a: (class_game_payoff(a, lam), -a))


def class_fixed_points(step: Fraction = Fraction(1, 20)) -> list[Fraction]:
    """Symmetrische Gleichgewichte des Klassenspiels auf dem Raster mit Schritt `step`: Faktoren λ > 0, die Beste Antwort auf sich selbst sind."""
    grid = [step * k for k in range(int(1 / step) + 1)]
    return [lam for lam in grid if lam > 0 and class_best_reply(lam, grid) == lam]
