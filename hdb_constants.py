"""Konstanten der Demo „Hohe Dimensionen“: Testfunktionen, Dimensionen, Verfahren, Studie, Regler, Presets, Farben."""

# --- Testfunktionen ------------------------------------------------------------------------------------------------------------------------
PROBLEMS = ("hartmann", "ackley")
PROBLEM_LABELS = {"hartmann": "Hartmann6, eingebettet (6 wirksame Eingaben, der Rest ohne Wirkung)", "ackley": "Ackley (alle Eingaben wirksam)"}
PROBLEM_SHORT = {"hartmann": "Hartmann6 eingebettet", "ackley": "Ackley"}
ACTIVE_DIMS = {"hartmann": 6, "ackley": None}                  # None: alle Eingaben wirksam
GLOBAL_MIN = {"hartmann": -3.32237, "ackley": 0.0}
DIMENSIONS = {"hartmann": (6, 12, 24, 48), "ackley": (2, 6, 12, 24)}
LIVE_DIMENSIONS = {"hartmann": (6, 8, 10, 12), "ackley": (2, 4, 6, 8)}
LIVE_MAX_N = 40

# --- Verfahren --------------------------------------------------------------------------------------------------------------------------
METHODS = ("random", "cma", "bo_ard", "bo_iso", "bo_scaled", "bo_tr")
METHOD_LABELS = {
    "random": "Zufall",
    "cma": "CMA-ES",
    "bo_ard": "BO, je Eingabe ein Längenmaß (ARD)",
    "bo_iso": "BO, ein gemeinsames Längenmaß",
    "bo_scaled": "BO, ARD mit größeren Start-Längenmaßen (√d)",
    "bo_tr": "BO mit Vertrauensbereich (TuRBO-artig)",
}
METHOD_COLORS = {"random": "#8c8c8c", "cma": "#D68A2E", "bo_ard": "#2F6B65", "bo_iso": "#7B5EA7", "bo_scaled": "#14233B", "bo_tr": "#B3463A"}
BO_METHODS = ("bo_ard", "bo_iso", "bo_scaled", "bo_tr")
LIVE_METHODS = ("random", "bo_ard", "bo_iso", "bo_tr")

# --- Studie ------------------------------------------------------------------------------------------------------------------------------
STUDY_BUDGET = 100
CHECKPOINTS = (25, 50, 100)
SWEEP_SEEDS = tuple(range(700000, 700010))
N_START_MAX = 30                 # Startdesign: min(d + 2, 30) Läufe
LS_BOUNDS = (0.05, 10.0)

# --- Regler -----------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 15, 40, 30
SEED_MAX, DEFAULT_SEED = 999999, 35

PRESETS = {
    "Standardfall (Voreinstellung)": {"dim": 10, "n": DEFAULT_N, "method": "bo_ard", "seed": DEFAULT_SEED},
    "Viele Eingaben ohne Wirkung": {"dim": 12, "n": 40, "method": "bo_ard", "seed": DEFAULT_SEED},
    "Ein gemeinsames Längenmaß": {"dim": 10, "n": DEFAULT_N, "method": "bo_iso", "seed": DEFAULT_SEED},
    "Vertrauensbereich": {"dim": 10, "n": DEFAULT_N, "method": "bo_tr", "seed": DEFAULT_SEED},
    "Wenige Auswertungen": {"dim": 12, "n": 15, "method": "bo_ard", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Hartmann6 mit 4 zusätzlichen Eingaben ohne Wirkung (insgesamt 10), 30 Auswertungen, BO mit je einem Längenmaß je Eingabe.",
    "Viele Eingaben ohne Wirkung": "Hartmann6 mit 6 zusätzlichen Eingaben ohne Wirkung (insgesamt 12) und 40 Auswertungen: das Modell muss die unwichtigen Eingaben erkennen.",
    "Ein gemeinsames Längenmaß": "Das Modell darf nur ein Längenmaß für alle Eingaben lernen und kann unwichtige Eingaben nicht ausblenden.",
    "Vertrauensbereich": "BO sucht nur in einem Kasten um die beste Stelle, der sich bei Erfolg weitet und bei Misserfolg verkleinert.",
    "Wenige Auswertungen": "Nur 15 Auswertungen bei 12 Eingaben: das Startdesign (14 Läufe) frisst fast das ganze Budget.",
}

COLOR_OBS = "#2F6B65"
COLOR_BAD = "#B3463A"
COLOR_NEXT = "#D68A2E"
