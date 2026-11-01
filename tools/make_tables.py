"""Rechnet die Messreihen E1 bis E5 (Erst-/Zweitpreis/All-pay, Risikoscheu, Mindestgebot, Bulow–Klemperer, Asymmetrie) neu und schreibt data/tables.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_tables.py [--workers 8]
Ändert sich eza_distributions.py, eza_auction.py, eza_hull.py, eza_asymmetry.py, eza_rng.py, eza_experiments.py oder der Messteil von eza_constants.py, stimmt der Hash nicht mehr
(tests/test_measurements.py schlägt an, die App zeigt die Reihen dann nicht); danach diesen Aufruf wiederholen. Jeder Block hat einen festen Zufalls-Seed aus seiner Bezeichnung
(eza_experiments.seed_of): die Reihen sind bei gleichem Code Zahl für Zahl dieselben, auch bei anderer Zahl von Prozessen."""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eza_experiments as XP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    start = time.time()
    jobs = XP.tables_jobs()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for job, res in ex.map(XP.run_table_job, jobs, chunksize=1):
            results[job] = res
    XP.write_json(XP.TABLES_FILE, XP.assemble_tables(results))
    print(f"geschrieben: {XP.TABLES_FILE}  ({time.time() - start:.0f} s, Hash {XP.tables_hash()}, {len(jobs)} Aufträge)")


if __name__ == "__main__":
    main()
