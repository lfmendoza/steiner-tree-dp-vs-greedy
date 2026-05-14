"""
Análisis estadístico de la calidad greedy vs óptimo.

A partir del CSV crudo producido por :mod:`bench.run_experiments`,
calcula resúmenes por ``(instance_family, n, k, algo)`` con mediana,
cuartiles y conteos de bins de histograma del cociente.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HIST_BINS = (0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, float("inf"))


def ratio_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen agregado por ``(instance_family, n, k, algo)``.

    Columnas calculadas:
      ``ratio_median``, ``ratio_q1``, ``ratio_q3``, ``ratio_min``,
      ``ratio_max``, ``count``.
    """
    df = df.dropna(subset=["ratio_vs_dp"]).copy()
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["instance_family", "n", "k", "algo"])["ratio_vs_dp"]
    summary = grouped.agg(
        ratio_median="median",
        ratio_q1=lambda s: float(np.quantile(s, 0.25)),
        ratio_q3=lambda s: float(np.quantile(s, 0.75)),
        ratio_min="min",
        ratio_max="max",
        count="count",
    ).reset_index()
    return summary


def ratio_histogram(df: pd.DataFrame, bins: tuple = HIST_BINS) -> pd.DataFrame:
    """Distribución del cociente por familia × algoritmo (long-form)."""
    df = df.dropna(subset=["ratio_vs_dp"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["bin"] = pd.cut(df["ratio_vs_dp"], bins=list(bins), include_lowest=True)
    return (
        df.groupby(["instance_family", "algo", "bin"], observed=True)
        .size()
        .reset_index(name="count")
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Carga el CSV crudo con los tipos esperados."""
    df = pd.read_csv(path)
    return df
