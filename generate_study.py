"""Vorgerechnete Studie: sechs Verfahren auf Hartmann6 (eingebettet in 6, 12, 24, 48 Eingaben) und Ackley (2, 6, 12, 24 Eingaben), je 10 feste Läufe, Budget 100.
Rund 10 bis 20 Minuten auf 14 Kernen. CMA-ES braucht das Paket `cma` (requirements-dev).

    python generate_study.py            # schreibt precomputed_study.json
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import hdb_constants as C
import hdb_evaluation as ev


def job(args):
    problem, dim, method, seed = args
    r = ev.run_one(problem, dim, method, seed, C.STUDY_BUDGET)
    return problem, dim, method, seed, {"best": r.best_so_far, "lengthscales": r.lengthscales, "seconds": r.seconds}


def main(workers=14):
    t0 = time.time()
    jobs = [(p, d, m, s) for p in C.PROBLEMS for d in C.DIMENSIONS[p] for m in C.METHODS for s in C.SWEEP_SEEDS]
    jobs.sort(key=lambda j: -j[1])                       # die langsamen (hohe Dimension) zuerst
    runs = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for problem, dim, method, seed, rec in ex.map(job, jobs, chunksize=1):
            slot = runs.setdefault(ev.study_key(problem, dim), {}).setdefault(method, {"best": [], "lengthscales": [], "seconds": [], "seeds": []})
            slot["best"].append(rec["best"])
            slot["seconds"].append(rec["seconds"])
            slot["seeds"].append(seed)
            if rec["lengthscales"]:
                slot["lengthscales"].append(rec["lengthscales"])
    out = {"runs": runs, "meta": {"seeds": len(C.SWEEP_SEEDS), "budget": C.STUDY_BUDGET, "jobs": len(jobs), "seconds": round(time.time() - t0)}}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_study.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("fertig", len(jobs), "Läufe in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
