"""Die vorgerechneten Messreihen (data/tables.json, data/lernen.json) gehören zum aktuellen Rechencode und werden von der Rechnung reproduziert.

Eigene Datei, weil jede Änderung an den Rechenmodulen oder den Messparametern in eza_constants den Erzeuger-Hash ändert. Deterministische Zellen werden neu gerechnet und mit der gespeicherten
Zelle verglichen; Zufallszellen (Simulation, Lernen) hängen nur vom Seed der Blockbezeichnung ab und werden mit einer Marge wegen unterschiedlicher Summationsreihenfolge verglichen."""

import json

import pytest

import eza_constants as C
import eza_experiments as XP
from eza_distributions import get_dist


@pytest.fixture(scope="module")
def tables():
    return json.loads(XP.TABLES_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lernen():
    return json.loads(XP.LEARN_FILE.read_text(encoding="utf-8"))


def test_files_match_the_generator_hashes(tables, lernen):
    assert tables["generator_hash"] == XP.tables_hash(), "Quellcode geändert: python tools/make_tables.py ausführen"
    assert lernen["generator_hash"] == XP.learn_hash(), "Quellcode geändert: python tools/make_lernen.py ausführen"
    assert XP.load_tables() is not None and XP.load_lernen() is not None


def test_load_returns_none_for_a_stale_hash_a_missing_file_and_a_broken_file(monkeypatch, tmp_path):
    monkeypatch.setattr(XP, "tables_hash", lambda: "0000000000000000")
    monkeypatch.setattr(XP, "learn_hash", lambda: "0000000000000000")
    assert XP.load_tables() is None and XP.load_lernen() is None
    monkeypatch.undo()
    monkeypatch.setattr(XP, "TABLES_FILE", tmp_path / "fehlt.json")
    assert XP.load_tables() is None
    broken = tmp_path / "kaputt.json"
    broken.write_text("{nicht json", encoding="utf-8")
    monkeypatch.setattr(XP, "LEARN_FILE", broken)
    assert XP.load_lernen() is None


def test_hash_ignores_the_slider_and_preset_part_of_the_constants():
    text = (XP.ROOT / "eza_constants.py").read_text(encoding="utf-8")
    assert text.count(XP.HASH_END) == 1                       # die Marke steht genau einmal, nicht schon im Kopf-Docstring
    before, after = text.split(XP.HASH_END)
    assert "E1_NS" in before and "LEARN_T" in before and "SIM_AUCTIONS" in before
    assert "PRESETS = {" in after and "N_MAX" in after and "PRESETS = {" not in before and "N_MAX" not in before


def test_changing_the_slider_part_does_not_change_the_hash(monkeypatch):
    base = XP.tables_hash()
    original = XP.ROOT / "eza_constants.py"
    text = original.read_text(encoding="utf-8")
    patched = text.replace("N_MIN, N_MAX, DEFAULT_N = 1, 20, 2", "N_MIN, N_MAX, DEFAULT_N = 1, 20, 3")
    assert patched != text
    monkeypatch.setattr(type(original), "read_text", lambda self, encoding=None: patched if self.name == "eza_constants.py" else open(self, encoding=encoding).read())
    assert XP.tables_hash() == base


def test_cells_match_the_code(tables, lernen):
    assert tables["blocks"] == C.BLOCKS and tables["auctions_per_block"] == C.SIM_AUCTIONS and tables["asym_auctions_per_block"] == C.ASYM_SIM_AUCTIONS
    assert [(c["kind"], c["n"]) for c in tables["e1"]] == [("gleich", n) for n in C.E1_NS] + [(k, n) for k, ns in C.E1_OTHER for n in ns]
    assert [(c["n"], c["rho"]) for c in tables["e2"]] == [(n, r) for n in C.E2_NS for r in C.E2_RHOS]
    assert len([c for c in tables["e3"] if c["kind"] == "gleich"]) == len(C.E3_V0S) * len(C.E3_NS)
    assert [(c["n"], c["v0"], c["r"]) for c in tables["e3sim"]] == [(n, v0, r) for n, v0, rs in C.E3_SIM_CELLS for r in rs]
    assert [c["w"] for c in tables["e5"]] == list(C.E5_WS)
    assert sorted(tables["e4"]) == sorted(["gleich|0.0", "exp|0.0", "beta|0.0", "gleich|20.0", "exp|20.0", "beta|20.0", "mix|0.0"])
    assert [r["n"] for r in tables["e4"]["gleich|0.0"]] == list(C.E4_NS) and [r["n"] for r in tables["e4"]["mix|0.0"]["rows"]] == list(C.E4_MIX_NS)
    assert len(XP.tables_jobs()) == len(set(XP.tables_jobs())) == 76
    assert len(XP.learn_jobs()) == len(set(XP.learn_jobs())) == 72
    assert lernen["blocks"] == C.LEARN_BLOCKS and all(len(c["alpha_last_blocks"]) == C.LEARN_BLOCKS for c in lernen["cells"])
    assert len(lernen["cells"]) == 18


def test_a_deterministic_cell_is_reproduced_exactly(tables):
    fresh = XP.e3_cell("gleich", 20.0, 2)
    stored = XP.find([c for c in tables["e3"] if c["kind"] == "gleich"], v0=20.0, n=2)
    for key in ("r_star", "r_grid", "u_virtual", "u_order", "u_grid", "u_plain", "p_sale", "loss", "loss_pct"):
        assert fresh[key] == pytest.approx(stored[key], rel=1e-9), key
    fresh5 = XP.e5_cell(0.5)
    stored5 = XP.find(tables["e5"], w=0.5)
    for key in ("rev_second", "rev_first", "rev_myerson", "misallocation", "loss_first", "loss_myerson"):
        assert fresh5[key] == pytest.approx(stored5[key], rel=1e-9), key


def test_a_simulated_cell_is_reproduced_by_its_seeds(tables):
    fresh = XP.e1_cell("gleich", 3)
    stored = XP.find([c for c in tables["e1"] if c["kind"] == "gleich"], n=3)
    for key in ("sim_second", "sim_first", "sim_allpay"):
        assert fresh[key]["mean"] == pytest.approx(stored[key]["mean"], abs=1e-6), key       # gleicher Strom; Marge nur wegen der Summationsreihenfolge
    assert fresh["rev_first"] == pytest.approx(stored["rev_first"], rel=1e-9)


def test_a_bulow_klemperer_series_is_reproduced(tables):
    fresh = XP.e4_rows("exp", 20.0, (1, 2))
    stored = tables["e4"]["exp|20.0"][:2]
    for a, b in zip(fresh, stored):
        for key in ("best", "plain_next", "ratio", "p_star"):
            assert a[key] == pytest.approx(b[key], rel=1e-9), key
        assert a["holds"] == b["holds"] and a["extra"] == b["extra"]


def test_a_short_learning_job_is_reproduced_within_a_margin(lernen):
    _, res = XP.run_learn_job(("Hedge", 2, "Zweitpreis", 2000, None, 0))
    cell = XP.find(lernen["cells"], rule="Hedge", n=2, fmt="Zweitpreis", rounds=2000, eta=None)
    assert res["alpha_last"] == pytest.approx(cell["alpha_last_blocks"][0], abs=0.02)         # ein Lauf mit gleichem Seed: Marge nur wegen unterschiedlicher libm-Rundung


def test_learning_cells_carry_the_equilibrium_and_standard_errors(lernen):
    for c in lernen["cells"]:
        assert c["alpha_eq"] == pytest.approx((c["n"] - 1) / c["n"] if c["fmt"] == "Erstpreis" else 1.0)
        assert c["revenue_eq"] == pytest.approx((c["n"] - 1) / (c["n"] + 1)) and c["alpha_last"]["se"] >= 0.0
        assert c["rounds"] in set(C.LEARN_T.values()) | set(C.LEARN_HORIZONS)


def test_the_stored_tables_use_the_documented_distributions(tables):
    for c in tables["e1"]:
        assert get_dist(c["kind"]).kind == c["kind"]
