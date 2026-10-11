"""Kern der Demo: Gauß-Prozess (Matérn-5/2) mit je Eingabe einem Längenmaß (ARD) oder einem gemeinsamen, erwartete Verbesserung, Suche mit optionalem Vertrauensbereich.

Alles auf dem Einheitswürfel, Minimierung, nur numpy/scipy. Eine Einheit = eine Regel (DEMO-PLAYBOOK §1b): `matern52`, `GaussianProcess.fit/predict`,
`expected_improvement`, `candidate_set` (Kandidaten der Akquisitionsmaximierung), `update_trust_region`, `bo_search`, `random_search`. Zufall nur über übergebene Generatoren.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

import hdb_constants as C


def matern52(A, B, lengthscales, variance):
    """Matérn-5/2-Kern k(a, b) = σ² (1 + √5 r + 5r²/3) exp(−√5 r), r = Abstand mit Längenmaßen je Eingabe (oder einem gemeinsamen)."""
    r2 = np.maximum((((A[:, None, :] - B[None, :, :]) / lengthscales) ** 2).sum(-1), 0.0)
    s = np.sqrt(5.0) * np.sqrt(r2)
    return variance * (1.0 + s + s * s / 3.0) * np.exp(-s)


class GaussianProcess:
    """Gauß-Prozess-Regression, Ausgabe standardisiert, Rauschvarianz fest klein (rauschfreie Funktionen). `isotropic`: ein gemeinsames Längenmaß.
    Die Hyperparameter maximieren die Randwahrscheinlichkeit (L-BFGS-B); `start_ls` ist der Startwert der Längenmaße, `warm` ein früherer Parameterstand."""

    def __init__(self, isotropic=False, start_ls=0.4, noise=1e-6):
        self.isotropic = isotropic
        self.start_ls = start_ls
        self.noise = noise
        self.theta_ = None

    def _unpack(self, theta, d):
        if self.isotropic:
            return np.full(d, np.exp(theta[0])), np.exp(theta[1])
        return np.exp(theta[:d]), np.exp(theta[d])

    def _neg_log_marginal(self, theta, X, z):
        ls, var = self._unpack(theta, X.shape[1])
        K = matern52(X, X, ls, var) + (self.noise + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, z)
        return 0.5 * z @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(z) * np.log(2 * np.pi)

    def fit(self, X, y, warm=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.X = X
        self.y_mean = float(y.mean())
        self.y_std = max(float(y.std()), 1e-9)
        z = (y - self.y_mean) / self.y_std
        d = X.shape[1]
        k = 1 if self.isotropic else d
        bounds = [(np.log(C.LS_BOUNDS[0]), np.log(C.LS_BOUNDS[1]))] * k + [(np.log(0.05), np.log(10.0))]
        th0 = np.concatenate([np.full(k, np.log(self.start_ls)), [0.0]]) if warm is None else np.asarray(warm, float)
        res = minimize(self._neg_log_marginal, th0, args=(X, z), method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
        self.theta_ = res.x
        self.lengthscales, self.variance = self._unpack(res.x, d)
        K = matern52(X, X, self.lengthscales, self.variance) + (self.noise + 1e-8) * np.eye(len(X))
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, z)
        return self

    def predict(self, Xs):
        """Mittelwert und Standardabweichung der Funktion."""
        Ks = matern52(np.asarray(Xs, float), self.X, self.lengthscales, self.variance)
        mu = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        var = np.maximum(self.variance - (Ks * v.T).sum(1), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std


def expected_improvement(mu, sd, best):
    """Erwartete Verbesserung (Minimierung): E[max(best − f, 0)], f ~ N(mu, sd²)."""
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def candidate_set(X, y, rng, n_random=1500, n_local=450, local_sd=0.05, trust=None):
    """Kandidaten für die Maximierung der Akquisition. Ohne Vertrauensbereich: zufällige Punkte im ganzen Würfel plus Störungen um die drei besten Beobachtungen.
    Mit Vertrauensbereich `trust = (Mitte, halbe Kantenlänge)`: zufällige Punkte im Kasten um die Mitte (auf den Würfel begrenzt)."""
    d = X.shape[1]
    if trust is not None:
        center, half = trust
        return np.clip(center + (rng.random((n_random, d)) - 0.5) * 2.0 * half, 0.0, 1.0)
    cand = rng.random((n_random, d))
    top = X[np.argsort(y)[: min(3, len(y))]]
    local = np.clip(np.repeat(top, n_local // max(len(top), 1), 0) + rng.normal(0.0, local_sd, (len(top) * (n_local // max(len(top), 1)), d)), 0.0, 1.0)
    return np.vstack([cand, local])


def update_trust_region(half, successes, failures, improved, d):
    """TuRBO-artige Regel: nach 3 Erfolgen in Folge weitet sich der halbe Kasten um das 1,6-Fache (höchstens 0,8), nach max(4, d/2) Misserfolgen halbiert er sich (mindestens 0,02).
    Gibt (half, successes, failures) zurück."""
    if improved:
        successes, failures = successes + 1, 0
    else:
        successes, failures = 0, failures + 1
    if successes >= 3:
        half, successes = min(half * 1.6, 0.8), 0
    if failures >= max(4, d // 2):
        half, failures = max(half / 2.0, 0.02), 0
    return half, successes, failures


def random_search(f, d, budget, rng):
    """Gleichverteilt im Würfel. Gibt (X, y) zurück."""
    X = rng.random((budget, d))
    return X, np.array([f(x) for x in X])


def bo_search(f, d, budget, rng, mode="ard", n_start=None, refit_every=None):
    """Bayessche Optimierung. mode: 'ard' (je Eingabe ein Längenmaß), 'iso' (ein gemeinsames), 'scaled' (ARD, Start-Längenmaß 0,25·√d), 'tr' (ARD mit Vertrauensbereich).
    Startdesign min(d + 2, 30) gleichverteilte Läufe. Gibt (X, y, Längenmaße des letzten Modells) zurück."""
    n_start = min(n_start or min(d + 2, C.N_START_MAX), budget)
    X = rng.random((n_start, d))
    y = np.array([f(x) for x in X])
    start_ls = 0.25 * np.sqrt(d) if mode == "scaled" else 0.4
    refit_every = refit_every or (1 if d <= 12 else 3)
    gp = None
    theta = None
    half, succ, fail = 0.4, 0, 0
    step = 0
    while len(y) < budget:
        if gp is None or step % refit_every == 0:
            gp = GaussianProcess(isotropic=(mode == "iso"), start_ls=start_ls).fit(X, y, warm=theta)
            theta = gp.theta_
        else:
            gp.X = X                          # Beobachtungen ergänzt, Hyperparameter behalten: Zerlegung neu
            gp = _refresh(gp, X, y)
        if mode == "tr":
            trust = (X[int(np.argmin(y))], half)
            cand = candidate_set(X, y, rng, trust=trust)
        else:
            cand = candidate_set(X, y, rng)
        mu, sd = gp.predict(cand)
        ei = expected_improvement(mu, sd, float(np.min(y)))
        x_new = cand[int(np.argmax(ei))]
        y_new = f(x_new)
        improved = y_new < float(np.min(y)) - 1e-9
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        if mode == "tr":
            half, succ, fail = update_trust_region(half, succ, fail, improved, d)
        step += 1
    return X, y, (gp.lengthscales if gp is not None else np.full(d, np.nan))


def _refresh(gp, X, y):
    """Hyperparameter behalten, Kernmatrix und Zerlegung für die erweiterten Beobachtungen neu rechnen."""
    gp.y_mean = float(y.mean())
    gp.y_std = max(float(y.std()), 1e-9)
    z = (y - gp.y_mean) / gp.y_std
    K = matern52(X, X, gp.lengthscales, gp.variance) + (gp.noise + 1e-8) * np.eye(len(X))
    gp._chol = cho_factor(K, lower=True)
    gp._alpha = cho_solve(gp._chol, z)
    return gp
