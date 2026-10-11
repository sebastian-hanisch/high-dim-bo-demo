"""Gemeinsame Fixtures: das Studienergebnis und eine kleine Beobachtungsmenge, die von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Vier Beobachtungen einer Funktion von zwei Eingaben (nur die erste wirkt): f(x) = x_1."""
    X = np.array([[0.1, 0.9], [0.4, 0.2], [0.7, 0.6], [0.9, 0.1]])
    return X, X[:, 0].copy()
