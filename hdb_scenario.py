"""Testfunktionen auf dem Einheitswürfel [0, 1]^d (Minimierung, rauschfrei): Hartmann6 mit unwirksamen Zusatzeingaben und Ackley mit allen Eingaben wirksam."""

import numpy as np

_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann6(x):
    """Hartmann6 (Dixon und Szegö) auf den ersten sechs Eingaben; weitere Eingaben haben **keine Wirkung**. Minimum −3,32237."""
    z = np.asarray(x, float)[:6]
    return -float((_ALPHA * np.exp(-(_A * (z - _P) ** 2).sum(1))).sum())


def ackley(x):
    """Ackley auf allen Eingaben, z = 8x − 4 in [−4, 4]^d; Minimum 0 bei x = 0,5 in allen Eingaben."""
    z = 8.0 * np.asarray(x, float) - 4.0
    return float(-20.0 * np.exp(-0.2 * np.sqrt((z ** 2).mean())) - np.exp(np.cos(2.0 * np.pi * z).mean()) + 20.0 + np.e)


def objective(problem):
    return hartmann6 if problem == "hartmann" else ackley
