"""
Ajustes polinomiales y bootstrap CIs sobre tiempos de ejecución.

Funciones principales
---------------------
:func:`fit_polynomial`
    np.polyfit con coeficientes en notación "decreciente" (compatible
    con np.poly1d). Reporta R² ajustado.

:func:`bootstrap_ci`
    Repite ``n_resamples`` bootstrap resamples y devuelve intervalos
    de confianza (percentiles) sobre cada coeficiente del ajuste.

:func:`fit_exponential_in_k`
    Ajuste lineal sobre ``log t = a + b·k``; útil para Dreyfus–Wagner
    (que es ~exponencial en k).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolyFit:
    coeffs: np.ndarray           # orden decreciente: c[0]*x^d + c[1]*x^(d-1) + ...
    degree: int
    r2: float
    r2_adj: float
    n_points: int

    def predict(self, x):
        return np.polyval(self.coeffs, x)


@dataclass
class BootstrapResult:
    coeffs_median: np.ndarray
    coeffs_ci_low: np.ndarray
    coeffs_ci_high: np.ndarray
    n_resamples: int
    confidence: float


def _r2(y, y_hat) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_polynomial(x, y, degree: int = 2) -> PolyFit:
    """Ajusta un polinomio de grado ``degree`` por mínimos cuadrados.

    Reporta R² y R² ajustado para penalizar grados altos.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= degree + 1:
        raise ValueError("Faltan puntos para el grado solicitado.")
    coeffs = np.polyfit(x, y, deg=degree)
    y_hat = np.polyval(coeffs, x)
    r2 = _r2(y, y_hat)
    n = len(x)
    p = degree + 1
    if n - p - 1 > 0:
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = float("nan")
    return PolyFit(coeffs=coeffs, degree=degree, r2=r2, r2_adj=r2_adj, n_points=n)


def bootstrap_ci(
    x, y, degree: int = 2,
    n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0,
) -> BootstrapResult:
    """Bootstrap por remuestreo con reemplazo sobre ``n_resamples`` réplicas."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= degree + 1:
        raise ValueError("Faltan puntos para bootstrap al grado solicitado.")

    rng = np.random.default_rng(seed)
    all_coeffs = np.empty((n_resamples, degree + 1))
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xr, yr = x[idx], y[idx]
        try:
            all_coeffs[r] = np.polyfit(xr, yr, deg=degree)
        except (np.linalg.LinAlgError, ValueError):
            all_coeffs[r] = np.nan

    finite = ~np.any(np.isnan(all_coeffs), axis=1)
    all_coeffs = all_coeffs[finite]
    if all_coeffs.shape[0] == 0:
        raise RuntimeError("Todos los resamples fallaron.")

    alpha = (1.0 - confidence) / 2.0
    low = np.quantile(all_coeffs, alpha, axis=0)
    med = np.quantile(all_coeffs, 0.5, axis=0)
    high = np.quantile(all_coeffs, 1.0 - alpha, axis=0)

    return BootstrapResult(
        coeffs_median=med, coeffs_ci_low=low, coeffs_ci_high=high,
        n_resamples=int(all_coeffs.shape[0]), confidence=confidence,
    )


def fit_exponential_in_k(k, t) -> PolyFit:
    """Ajuste ``log t = a + b·k``. Devuelve ``PolyFit`` sobre ``log t``."""
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = t > 0
    k = k[mask]
    log_t = np.log(t[mask])
    return fit_polynomial(k, log_t, degree=1)
