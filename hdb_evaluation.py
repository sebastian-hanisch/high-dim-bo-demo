"""Auswertung: ein Lauf je Verfahren auf einer Testfunktion, Verlaufsspuren, gelernte Längenmaße, vorgerechnete Studie mit gepaarten Läufen und Vorzeichentest."""

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import hdb_algorithm as A
import hdb_constants as C
import hdb_scenario as S

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFSET = {m: 1000 * (k + 1) for k, m in enumerate(C.METHODS)}


@dataclass(frozen=True)
class Settings:
    problem: str = "hartmann"
    dim: int = 10
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


@dataclass
class Run:
    method: str
    best_so_far: list         # bester Wert nach t Auswertungen
    lengthscales: list        # gelernte Längenmaße des letzten Modells je Eingabe (leer bei Zufall und CMA-ES)
    seconds: float
    X: np.ndarray             # ausgewertete Stellen in Reihenfolge


def run_one(problem, dim, method, seed, budget):
    """Ein Lauf; Zufallsstrom aus Seed und Verfahren. 'cma' braucht das Paket cma (nur für die Studie, requirements-dev)."""
    f = S.objective(problem)
    rng = np.random.default_rng(seed + _OFFSET[method])
    t0 = time.perf_counter()
    ls = []
    if method == "random":
        X, y = A.random_search(f, dim, budget, rng)
    elif method == "cma":
        import cma
        es = cma.CMAEvolutionStrategy(rng.random(dim), 0.3, {"bounds": [0, 1], "seed": int(seed + 1) % (2 ** 31), "verbose": -9, "popsize": 4 + int(3 * np.log(dim))})
        xs, ys = [], []
        while len(ys) < budget:
            cand = es.ask()
            vals = []
            for x in cand:
                if len(ys) >= budget:
                    break
                x = np.clip(np.asarray(x), 0.0, 1.0)
                v = f(x)
                xs.append(x)
                ys.append(v)
                vals.append(v)
            if len(vals) == len(cand):
                es.tell(cand, vals)
        X, y = np.array(xs), np.array(ys)
    else:
        X, y, ls_arr = A.bo_search(f, dim, budget, rng, mode=method.split("_", 1)[1])
        ls = [float(v) for v in ls_arr]
    best = np.minimum.accumulate(y)
    return Run(method, [float(v) for v in best], ls, time.perf_counter() - t0, X)


def regret(problem, values):
    return np.asarray(values, float) - C.GLOBAL_MIN[problem]


# --- Vorgerechnete Studie ------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def study():
    with open(os.path.join(_HERE, "precomputed_study.json"), encoding="utf-8") as fh:
        return json.load(fh)


def study_key(problem, dim):
    return f"{problem}_{int(dim)}"


def table(problem, dim):
    """Median und 90-%-Quantil des Regrets je Verfahren und Prüfpunkt (10 feste Läufe), dazu Mediane der Rechenzeit je Lauf (Sekunden) und der gelernten Längenmaße."""
    raw = study()["runs"][study_key(problem, dim)]
    out = {}
    for m, rec in raw.items():
        arr = np.array(rec["best"])                                   # Läufe x Budget
        reg = arr - C.GLOBAL_MIN[problem]
        out[m] = {cp: {"median": float(np.median(reg[:, cp - 1])), "q90": float(np.quantile(reg[:, cp - 1], 0.9))} for cp in C.CHECKPOINTS}
        out[m]["seconds"] = float(np.median(rec["seconds"]))
    return out


def curves(problem, dim):
    """Median und 25-/75-%-Quantil des Regrets über das Budget je Verfahren (10 feste Läufe)."""
    raw = study()["runs"][study_key(problem, dim)]
    out = {}
    for m, rec in raw.items():
        reg = np.array(rec["best"]) - C.GLOBAL_MIN[problem]
        out[m] = {"median": np.median(reg, axis=0), "q25": np.quantile(reg, 0.25, axis=0), "q75": np.quantile(reg, 0.75, axis=0)}
    return out


def paired_stats(problem, dim, key, base, checkpoint, tol=1e-9):
    """Gepaarter Vergleich der Variante `key` gegen `base` über die 10 Läufe (Regret; negativ = Variante besser): besser, schlechter, gleich, Median der Differenz, p (Vorzeichentest)."""
    from scipy.stats import binomtest
    raw = study()["runs"][study_key(problem, dim)]
    a = np.array(raw[key]["best"])[:, checkpoint - 1]
    b = np.array(raw[base]["best"])[:, checkpoint - 1]
    diff = a - b
    better, worse = int((diff < -tol).sum()), int((diff > tol).sum())
    p = float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0
    return {"better": better, "worse": worse, "equal": int(len(diff) - better - worse), "median_diff": float(np.median(diff)), "p": p, "n": int(len(diff))}


def verdict_of(stats, alpha=0.05):
    if stats["p"] < alpha:
        return "besser" if stats["better"] > stats["worse"] else "schlechter"
    return "kein Unterschied nachweisbar"


def length_scales_summary(problem, dim):
    """Mittleres gelerntes Längenmaß der wirksamen und der unwirksamen Eingaben (nur Hartmann6; BO mit ARD): Mittel über die Läufe."""
    raw = study()["runs"][study_key(problem, dim)]["bo_ard"]["lengthscales"]
    arr = np.array(raw)                                                # Läufe x d
    if problem != "hartmann":
        return {"active": float(arr.mean()), "inactive": None}
    act = C.ACTIVE_DIMS["hartmann"]
    return {"active": float(arr[:, :act].mean()), "inactive": float(arr[:, act:].mean()) if arr.shape[1] > act else None}
