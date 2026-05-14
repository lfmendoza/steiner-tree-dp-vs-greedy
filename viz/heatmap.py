"""
Heatmap (n, k) → cociente medio greedy/óptimo desde un CSV de bench.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    algo: str = "mst_heuristic",
    family: str | None = None,
    value: str = "ratio_vs_dp",
    aggfunc: str = "median",
    title: str | None = None,
) -> Path:
    """Lee el CSV y dibuja un heatmap ``(n, k) → value``.

    Parameters
    ----------
    csv_path : str | Path
        CSV producido por ``bench/run_experiments.py``.
    out_path : str | Path
        Ruta del PNG de salida.
    algo : str
        Nombre del algoritmo a filtrar.
    family : str, optional
        Familia de instancia para filtrar; si None, agrega todas.
    value : str
        Columna del CSV a agregar.
    aggfunc : str
        Función de agregación para pivot_table (``median``, ``mean``, etc.).
    title : str, optional
        Título de la figura.

    Returns
    -------
    Path
        ``out_path``.
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["algo"] == algo]
    if family is not None:
        df = df[df["instance_family"] == family]
    df = df.dropna(subset=[value])

    if df.empty:
        # Figura placeholder explícito.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path

    pivot = df.pivot_table(index="k", columns="n", values=value, aggfunc=aggfunc)
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n (|V|)")
    ax.set_ylabel("k (|T|)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{aggfunc}({value})")
    # Etiquetas numéricas dentro de cada celda.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black",
                        fontsize=8)

    if title:
        ax.set_title(title)
    else:
        suffix = f", family={family}" if family else ""
        ax.set_title(f"{algo} — {aggfunc}({value}){suffix}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
