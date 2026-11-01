"""Messreihen der Experimente E1 bis E6, vorgerechnet in data/tables.json (E1 bis E5) und data/lernen.json (E6).

Jede Zelle ist eine Rechnung (exakte Gitterintegrale, Beste-Antwort-Lücken) und, wo ein Zufall im Spiel ist, `BLOCKS` unabhängige Blöcke zu je `SIM_AUCTIONS` Auktionen;
jeder Block hat einen eigenen SplitMix64-Strom aus seiner Bezeichnung (`seed_of`, FNV-1a: Python-`hash` ist je Prozess gesalzen und taugt nicht). Ausgewiesen werden Mittel und
Standardfehler der Blockwerte. Die Dateien tragen den Hash der Quellen, aus denen sie stammen (`generator_hash`): ändert sich Code oder Messparameter, passt der Hash nicht mehr
und die App zeigt die Reihen nicht, bis `tools/make_tables.py` bzw. `tools/make_lernen.py` neu gelaufen ist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import eza_asymmetry as AS
import eza_auction as A
import eza_constants as C
import eza_hull as HU
import eza_learning as LE
from eza_distributions import get_dist
from eza_rng import uniform_block

ROOT = Path(__file__).resolve().parent
TABLES_FILE = ROOT / "data" / "tables.json"
LEARN_FILE = ROOT / "data" / "lernen.json"
TABLE_SOURCES = ("eza_distributions.py", "eza_auction.py", "eza_hull.py", "eza_asymmetry.py", "eza_rng.py", "eza_experiments.py", "eza_constants.py")
LEARN_SOURCES = ("eza_learning.py", "eza_rng.py", "eza_experiments.py", "eza_constants.py")
HASH_END = "# --- ab hier nicht mehr im Erzeuger-Hash"
NEGATIVE_CONTROL_VALUES = (30.0, 60.0, 90.0)
MIX_GAP_VALUES = (2.0, 5.0, 9.0, 92.0, 95.0, 100.0)


def seed_of(label: str) -> int:
    """64-Bit-Seed aus einer Bezeichnung (FNV-1a über die UTF-8-Bytes)."""
    h = 0xCBF29CE484222325
    for byte in label.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _hash(names) -> str:
    h = hashlib.sha256()
    for name in names:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        if name == "eza_constants.py":
            text = text.split(HASH_END)[0]
        h.update(name.encode() + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def tables_hash() -> str:
    return _hash(TABLE_SOURCES)


def learn_hash() -> str:
    return _hash(LEARN_SOURCES)


def mean_se(values) -> dict:
    """Mittel und Standardfehler (ddof = 1) der Blockwerte."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}


def _sim(label: str, fn) -> dict:
    """`BLOCKS` Blöcke: fn(seed) -> Erlös des Blocks."""
    return mean_se([fn(seed_of(f"{label}|{b}")) for b in range(C.BLOCKS)])


# ----------------------------------------------------------------------------------------------- Zellen E1 bis E5
def e1_cell(kind: str, n: int) -> dict:
    """Erst-/Zweitpreis/All-pay ohne Mindestgebot: Gebot beim Wert 50, Erlöse auf allen Wegen, Simulation, Beste-Antwort-Lücken."""
    d = get_dist(kind)
    vs, bs = A.fpa_bid_function(d, n)
    gap_values = C.E1_GAP_VALUES if kind != "mix" else MIX_GAP_VALUES
    out = {"kind": kind, "n": n, "bid_at_50": float(np.interp(50.0, vs, bs)) if kind != "mix" else None,
           "rev_virtual": A.spa_revenue_virtual(d, n), "rev_order": A.spa_revenue_order(d, n), "rev_first": A.fpa_revenue(d, n), "rev_allpay": A.allpay_revenue(d, n),
           "formula": 100.0 * (n - 1) / (n + 1) if kind == "gleich" else None,
           "gap_first": max(A.fpa_best_response_gap(d, n, 0.0, v) for v in gap_values), "gap_allpay": max(A.allpay_best_response_gap(d, n, v) for v in gap_values)}
    for fmt, key in (("zweit", "sim_second"), ("erst", "sim_first"), ("allpay", "sim_allpay")):
        out[key] = _sim(f"e1|{kind}|{n}|{fmt}", lambda s, fmt=fmt: A.simulate_revenue(d, n, fmt, s, C.SIM_AUCTIONS))
    return out


def e1_control() -> list[dict]:
    """Negativkontrolle des Beste-Antwort-Prüfers: n = 2, die Gegner bieten fälschlich w²/100; das Gleichgewichtsgebot v/2 muss eine Lücke > 0 haben."""
    d = get_dist("gleich")
    vs = np.linspace(0.0, 100.0, A.BR_GRID_N)
    wrong = (vs, vs * vs / 100.0)
    return [{"value": v, "gap_wrong_opponent": A.fpa_best_response_gap(d, 2, 0.0, v, opponent=wrong, own_bid=v / 2),
             "gap_right_opponent": A.fpa_best_response_gap(d, 2, 0.0, v, own_bid=v / 2)} for v in NEGATIVE_CONTROL_VALUES]


def e2_cell(n: int, rho: float) -> dict:
    lam = A.crra_factor(n, rho)
    spa = 100.0 * (n - 1) / (n + 1)
    first = lam * 100.0 * n / (n + 1)
    return {"n": n, "rho": rho, "factor": lam, "rev_first": first, "rev_second": spa, "ratio": first / spa,
            "gap": max(A.crra_best_response_gap(n, rho, v) for v in (30.0, 60.0, 90.0)),
            "sim_first": _sim(f"e2|{n}|{rho}", lambda s: A.simulate_crra_revenue(n, rho, s, C.SIM_AUCTIONS))}


def e3_cell(kind: str, v0: float, n: int) -> dict:
    """Mindestgebot: Reserve aus φ(r) = v0, Gittersuche, Nutzen auf zwei Wegen, Gewinn gegenüber r = v0, Verkaufswahrscheinlichkeit, Wohlfahrtsverlust."""
    d = get_dist(kind)
    r_star = A.optimal_reserve(d, v0)
    r_grid, u_grid = A.best_reserve_on_grid(d, n, v0)
    u_b = A.seller_utility(d, n, r_star, v0)
    u_a = A.spa_revenue_virtual(d, n, r_star) + v0 * float(d.cdf(r_star)) ** n
    u_plain = A.seller_utility(d, n, v0, v0)
    eff = A.efficient_value(d, n, v0)
    loss = A.welfare_loss(d, n, r_star, v0)
    return {"kind": kind, "v0": v0, "n": n, "r_formula": (100.0 + v0) / 2 if kind == "gleich" else (v0 + 50.0 if kind == "exp" else None), "r_star": r_star, "r_grid": r_grid,
            "u_virtual": u_a, "u_order": u_b, "u_grid": u_grid, "u_plain": u_plain, "gain_pct": 100.0 * (u_b / u_plain - 1.0) if u_plain > 1e-9 else None,
            "p_sale": A.sale_probability(d, n, r_star), "loss": loss, "loss_pct": 100.0 * loss / eff}


def e3_sim_cell(n: int, v0: float, r: float) -> dict:
    """Nutzen des Verkäufers bei Mindestgebot r: Weg B, Weg A und Simulation (Zweitpreis, Gleichverteilung)."""
    d = get_dist("gleich")
    p_unsold = float(d.cdf(r)) ** n
    return {"n": n, "v0": v0, "r": r, "u_order": A.seller_utility(d, n, r, v0), "u_virtual": A.spa_revenue_virtual(d, n, r) + v0 * p_unsold,
            "sim": mean_se([_e3_sim_block(d, n, v0, r, b) for b in range(C.BLOCKS)])}


def _e3_sim_block(d, n, v0, r, b) -> float:
    """Nutzen des Verkäufers in einem Block: Erlös des Zweitpreises mit Mindestgebot r plus v0 mal Anteil unverkaufter Auktionen (aus denselben Werten)."""
    values = d.ppf(uniform_block(seed_of(f"e3sim|{n}|{v0}|{r}|{b}"), C.SIM_AUCTIONS * n).reshape(C.SIM_AUCTIONS, n))
    srt = np.sort(values, axis=1)
    top = srt[:, -1]
    second = srt[:, -2] if n >= 2 else np.full(len(top), -np.inf)
    sold = top >= r
    return float(np.mean(np.where(sold, np.maximum(r, second), v0)))


def e3_fpa_cell(n: int, r: float) -> dict:
    """Erstpreis mit Mindestgebot gleich Zweitpreis mit Mindestgebot (Gleichverteilung): Erlöse, Lücke, Gebot beim Wert r."""
    d = get_dist("gleich")
    vs, bs = A.fpa_bid_function(d, n, r)
    return {"n": n, "r": r, "rev_first": A.fpa_revenue(d, n, r), "rev_second": A.spa_revenue_order(d, n, r),
            "gap": max(A.fpa_best_response_gap(d, n, r, v) for v in (r + 5.0, (r + 100.0) / 2, 100.0)), "bid_at_r": float(bs[0]), "bid_at_100": float(bs[-1]),
            "sim_first": _sim(f"e3fpa|{n}|{r}", lambda s: A.simulate_revenue(d, n, "erst", s, C.SIM_AUCTIONS, r=r))}


def e4_rows(kind: str, v0: float, ns) -> list[dict]:
    """Bulow–Klemperer: bestes Verfahren mit n Bietern gegen die einfache Auktion mit n + 1 Bietern (Zweitpreis mit Mindestgebot v0)."""
    d = get_dist(kind)
    q, r = HU.revenue_curve(d)
    hull = HU.upper_hull(q, r)
    rows = []
    for n in ns:
        row = HU.bulow_klemperer_row(d, n, v0, q, r, hull)
        row["best_virtual"] = HU.optimal_utility_regular(d, n, v0) if d.regular else None
        row["plain_quantile"] = HU.plain_revenue_from_curve(q, r, n + 1) if v0 == 0.0 else None
        rows.append(row)
    return rows


def e5_cell(w: float) -> dict:
    """Asymmetrie: Erlöse (Zweitpreis, Erstpreis, Myerson), falsche Zuteilung, Wohlfahrtsverluste, Randwerte und Beste-Antwort-Lücken."""
    spa, fpa, my = AS.second_price_revenue(w), AS.first_price_revenue(w), AS.myerson_revenue(w)
    mis, loss = AS.first_price_misallocation(w)
    bb = AS.top_bid(w)
    return {"w": w, "rev_second": spa, "rev_first": fpa, "rev_myerson": my, "ratio_first": fpa / spa, "ratio_myerson": my / spa,
            "misallocation": mis, "loss_first": loss, "loss_myerson": AS.myerson_welfare_loss(w), "top_bid": AS.H * bb,
            "x_top": float(AS.inverse_strong(bb, w)), "y_top": float(AS.inverse_weak(bb, w)),
            "sim": {"second": _sim(f"e5|{w}|zweit", lambda s: _asym_sim(w, s, "zweit")), "first": _sim(f"e5|{w}|erst", lambda s: _asym_sim(w, s, "erst")),
                    "myerson": _sim(f"e5|{w}|myerson", lambda s: _asym_sim(w, s, "myerson"))}}


def _asym_sim(w: float, seed: int, mechanism: str) -> float:
    """Simulation zweier Bieter (Einheiten 100 Euro, Ergebnis in Euro): Wert 1 ~ U[0, 1], Wert 2 ~ U[0, w]."""
    m = C.ASYM_SIM_AUCTIONS
    u = uniform_block(seed, 2 * m).reshape(2, m)
    v1, v2 = u[0], u[1] * w
    if mechanism == "zweit":
        return AS.H * float(np.mean(np.minimum(v1, v2)))
    if mechanism == "erst":
        return AS.H * float(np.mean(np.maximum(AS.bid_strong(v1, w), AS.bid_weak(v2, w))))
    p1, p2 = 2 * v1 - 1, 2 * v2 - w
    one = (p1 >= p2) & (p1 > 0)
    two = (p2 > p1) & (p2 > 0)
    pay = np.where(one, (1 + np.maximum(p2, 0)) / 2, np.where(two, (w + np.maximum(p1, 0)) / 2, 0.0))
    return AS.H * float(np.mean(pay))


def asym_best_response_gap(w: float) -> dict:
    """Beste-Antwort-Gitter beider Bieter im asymmetrischen Erstpreis-Gleichgewicht (Einheiten 100 Euro, Ergebnis in Euro) und der Rest der beiden Differentialgleichungen."""
    bb = AS.top_bid(w)
    grid = np.linspace(0.0, bb, 20001)
    gap_strong = gap_weak = 0.0
    for v1 in (0.3, 0.6, 0.9, 1.0):
        u = (v1 - grid) * np.minimum(AS.inverse_weak(grid, w) / w, 1.0)
        b_eq = float(AS.bid_strong(v1, w))
        u_eq = (v1 - b_eq) * min(float(AS.inverse_weak(b_eq, w)) / w, 1.0)
        gap_strong = max(gap_strong, float(u.max() - u_eq))
    for v2 in (0.3 * w, 0.6 * w, 0.9 * w, w):
        u = (v2 - grid) * np.minimum(AS.inverse_strong(grid, w), 1.0)
        b_eq = float(AS.bid_weak(v2, w))
        u_eq = (v2 - b_eq) * min(float(AS.inverse_strong(b_eq, w)), 1.0)
        gap_weak = max(gap_weak, float(u.max() - u_eq))
    return {"gap_strong": AS.H * gap_strong, "gap_weak": AS.H * gap_weak}


def tables_jobs() -> list[tuple]:
    jobs = [("e1", "gleich", n) for n in C.E1_NS] + [("e1", kind, n) for kind, ns in C.E1_OTHER for n in ns]
    jobs += [("e1control",)] + [("e2", n, rho) for n in C.E2_NS for rho in C.E2_RHOS]
    jobs += [("e3", "gleich", v0, n) for v0 in C.E3_V0S for n in C.E3_NS] + [("e3", kind, v0, n) for kind in ("exp", "beta") for v0 in C.E3_OTHER_V0S for n in C.E3_OTHER_NS]
    jobs += [("e3sim", n, v0, r) for n, v0, rs in C.E3_SIM_CELLS for r in rs] + [("e3fpa", n, r) for n, r in C.E3_FPA_CELLS]
    jobs += [("e4", "gleich", 0.0), ("e4", "exp", 0.0), ("e4", "beta", 0.0), ("e4", "gleich", C.E4_V0), ("e4", "exp", C.E4_V0), ("e4", "beta", C.E4_V0), ("e4", "mix", 0.0)]
    jobs += [("e5", w) for w in C.E5_WS]
    return jobs


def run_table_job(job: tuple):
    """Ein Auftrag; Rückgabe (Auftrag, Ergebnis). Läuft in Unterprozessen (`tools/make_tables.py`)."""
    kind = job[0]
    if kind == "e1":
        res = e1_cell(job[1], job[2])
    elif kind == "e1control":
        res = e1_control()
    elif kind == "e2":
        res = e2_cell(job[1], job[2])
    elif kind == "e3":
        res = e3_cell(job[1], job[2], job[3])
    elif kind == "e3sim":
        res = e3_sim_cell(job[1], job[2], job[3])
    elif kind == "e3fpa":
        res = e3_fpa_cell(job[1], job[2])
    elif kind == "e4":
        ns = C.E4_MIX_NS if job[1] == "mix" else C.E4_NS
        res = e4_rows(job[1], job[2], ns)
        if job[1] == "mix":
            q, r = HU.revenue_curve(get_dist("mix"))
            res = {"rows": res, "max_curve": float(r.max())}
    elif kind == "e5":
        res = {**e5_cell(job[1]), **asym_best_response_gap(job[1])}
    else:
        raise ValueError(job)
    return job, res


def assemble_tables(results: dict) -> dict:
    g = lambda *k: results[tuple(k)]
    return {
        "generator_hash": tables_hash(), "blocks": C.BLOCKS, "auctions_per_block": C.SIM_AUCTIONS, "asym_auctions_per_block": C.ASYM_SIM_AUCTIONS,
        "e1": [g("e1", "gleich", n) for n in C.E1_NS] + [g("e1", kind, n) for kind, ns in C.E1_OTHER for n in ns],
        "e1control": g("e1control"),
        "e2": [g("e2", n, rho) for n in C.E2_NS for rho in C.E2_RHOS],
        "e3": [g("e3", "gleich", v0, n) for v0 in C.E3_V0S for n in C.E3_NS] + [g("e3", kind, v0, n) for kind in ("exp", "beta") for v0 in C.E3_OTHER_V0S for n in C.E3_OTHER_NS],
        "e3sim": [g("e3sim", n, v0, r) for n, v0, rs in C.E3_SIM_CELLS for r in rs],
        "e3fpa": [g("e3fpa", n, r) for n, r in C.E3_FPA_CELLS],
        "e4": {f"{kind}|{v0}": g("e4", kind, v0) for kind, v0 in (("gleich", 0.0), ("exp", 0.0), ("beta", 0.0), ("gleich", C.E4_V0), ("exp", C.E4_V0), ("beta", C.E4_V0), ("mix", 0.0))},
        "e5": [g("e5", w) for w in C.E5_WS],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def _load(path: Path, current_hash: str):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if payload.get("generator_hash") == current_hash else None


def load_tables():
    """Die vorgerechneten Reihen E1 bis E5, oder None, wenn die Datei fehlt oder zu einem anderen Rechencode gehört."""
    return _load(TABLES_FILE, tables_hash())


def load_lernen():
    """Die vorgerechneten Lernläufe E6, oder None (wie `load_tables`)."""
    return _load(LEARN_FILE, learn_hash())


def find(rows, **conds):
    """Die erste Zeile, deren Felder den Bedingungen entsprechen (genau eine muss es geben)."""
    hit = [r for r in rows if all(r[k] == v for k, v in conds.items())]
    if len(hit) != 1:
        raise KeyError(f"{len(hit)} Zeilen für {conds}")
    return hit[0]


# ----------------------------------------------------------------------------------------------- E6 Lernen
def learn_jobs() -> list[tuple]:
    """(Regel, Bieter, Format, Runden, feste Rate oder None, Block). Hedge und Exp3: n = 2 und 4; Q-Lernen: n = 2; Hedge zusätzlich über den Horizont und mit fester Rate."""
    jobs = []
    for rule in ("Hedge", "Exp3"):
        for n in C.LEARN_NS_TABLE:
            for fmt in LE.FORMATS:
                jobs += [(rule, n, fmt, C.LEARN_T[rule], None, b) for b in range(C.LEARN_BLOCKS)]
    jobs += [("Q-Lernen", n, fmt, C.LEARN_T["Q-Lernen"], None, b) for n in C.LEARN_Q_NS for fmt in LE.FORMATS for b in range(C.LEARN_BLOCKS)]
    for fmt in LE.FORMATS:
        jobs += [("Hedge", 2, fmt, t, None, b) for t in C.LEARN_HORIZONS if t != C.LEARN_T["Hedge"] for b in range(C.LEARN_BLOCKS)]
        jobs += [("Hedge", 2, fmt, t, C.LEARN_FIXED_ETA, b) for t in C.LEARN_FIXED_ETA_HORIZONS for b in range(C.LEARN_BLOCKS)]
    return jobs


def run_learn_job(job: tuple):
    rule, n, fmt, rounds, eta, block = job
    seed = seed_of(f"learn|{rule}|{n}|{fmt}|{rounds}|{eta}|{block}")
    res = LE.hedge_run(n, fmt, rounds, seed, eta=eta) if eta is not None else LE.run(rule, n, fmt, rounds, seed)
    return job, {k: res[k] for k in ("alpha_last", "alpha_played", "revenue")}


def assemble_learn(results: dict) -> dict:
    cells = {}
    for (rule, n, fmt, rounds, eta, block), res in results.items():
        cells.setdefault((rule, n, fmt, rounds, eta), []).append((block, res))
    out = []
    for (rule, n, fmt, rounds, eta), items in cells.items():
        items.sort()
        out.append({"rule": rule, "n": n, "fmt": fmt, "rounds": rounds, "eta": eta, "alpha_eq": LE.equilibrium_alpha(n, fmt), "revenue_eq": LE.equilibrium_revenue(n),
                    "alpha_last": mean_se([r["alpha_last"] for _, r in items]), "alpha_played": mean_se([r["alpha_played"] for _, r in items]),
                    "revenue": mean_se([r["revenue"] for _, r in items]), "alpha_last_blocks": [r["alpha_last"] for _, r in items]})
    return {"generator_hash": learn_hash(), "blocks": C.LEARN_BLOCKS, "cells": out}
