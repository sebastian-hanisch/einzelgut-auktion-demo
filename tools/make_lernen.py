"""Rechnet die Lernläufe E6 (Hedge, Exp3, Q-Lernen im Erst- und Zweitpreis) neu und schreibt data/lernen.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_lernen.py [--workers 8]
Ändert sich eza_learning.py, eza_rng.py, eza_experiments.py oder der Messteil von eza_constants.py, stimmt der Hash nicht mehr; danach diesen Aufruf wiederholen.
Dauer: einige Minuten (Python-Schleife über bis zu 300.000 Runden je Lauf). Die Läufe sind bei gleichem Code Zahl für Zahl dieselben, auch bei anderer Zahl von Prozessen."""

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
    jobs = XP.learn_jobs()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for job, res in ex.map(XP.run_learn_job, jobs, chunksize=1):
            results[job] = res
            print(f"{len(results)}/{len(jobs)} {job} ({time.time() - start:.0f} s)", flush=True)
    XP.write_json(XP.LEARN_FILE, XP.assemble_learn(results))
    print(f"geschrieben: {XP.LEARN_FILE}  ({time.time() - start:.0f} s, Hash {XP.learn_hash()}, {len(jobs)} Aufträge)")


if __name__ == "__main__":
    main()
