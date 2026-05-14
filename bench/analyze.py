"""
CSV crudo → figuras y tablas LaTeX listas para el paper.

Genera, en ``--output`` (por defecto ``docs/figures``):

* ``time_scatter.png`` — tiempo log-log (DP vs greedy).
* ``quality_box.png`` — boxplot del cociente por familia.
* ``spider_ratio.png`` — cociente del spider vs la cota teórica.
* ``heatmap_kmb_euclidean.png`` — heatmap (n, k) -> cociente (KMB sobre euclidean).

Y en ``docs/tables``:

* ``steinlib.tex`` — tabla de instancias SteinLib.
* ``regression_dp.tex`` — coeficientes con CIs del ajuste DP.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .quality import ratio_stats
from .regression import bootstrap_ci, fit_exponential_in_k, fit_polynomial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _algo_color(name: str) -> str:
    return {
        "dreyfus_wagner": "#1f77b4",
        "mst_heuristic": "#d62728",
        "mehlhorn": "#2ca02c",
        "rsph": "#9467bd",
    }.get(name, "#7f7f7f")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def plot_time_scatter(df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = False
    for algo in ("dreyfus_wagner", "mst_heuristic", "mehlhorn", "rsph"):
        sub = df[(df["algo"] == algo) & (df["time_s_median"] > 0) & (~df["timed_out"])]
        if sub.empty:
            continue
        # DP: eje x = k, greedies: eje x = n.
        if algo == "dreyfus_wagner":
            x = sub["k"]
            label = f"{algo} (vs k)"
        else:
            x = sub["n"]
            label = f"{algo} (vs n)"
        ax.scatter(x, sub["time_s_median"], color=_algo_color(algo), label=label, alpha=0.6, s=18)
        plotted = True

    if plotted:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("tamaño (k para DP, n para greedy)")
    ax.set_ylabel("tiempo mediano (s)")
    ax.set_title("Tiempo de ejecución (log-log)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quality_box(df: pd.DataFrame, out_path: Path) -> Path:
    sub = df.dropna(subset=["ratio_vs_dp"])
    sub = sub[sub["algo"] != "dreyfus_wagner"]
    if sub.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    families = sorted(sub["instance_family"].unique())
    algos = ["mst_heuristic", "mehlhorn", "rsph"]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(families)), 5))
    positions = []
    labels = []
    data: list[list[float]] = []
    colors: list[str] = []
    x = 0
    width = 0.8
    for fam in families:
        for algo in algos:
            vals = sub[(sub["instance_family"] == fam) & (sub["algo"] == algo)]["ratio_vs_dp"].tolist()
            if not vals:
                continue
            data.append(vals)
            positions.append(x)
            colors.append(_algo_color(algo))
            labels.append(f"{fam}\n{algo}")
            x += width
        x += width  # gap entre familias

    bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(2.0, color="black", linestyle="--", alpha=0.5, label="cota 2·OPT")
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("cociente greedy / óptimo")
    ax.set_title("Calidad del greedy por familia (boxplot del cociente)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spider_ratio(df: pd.DataFrame, out_path: Path) -> Path:
    sub = df[df["instance_family"].str.startswith("spider")].copy()
    sub = sub.dropna(subset=["ratio_vs_dp"])
    sub = sub[sub["algo"] != "dreyfus_wagner"]

    if sub.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "(sin datos de spider)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    sub["epsilon"] = sub["instance_family"].str.replace("spider_eps", "").astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))

    ks = np.arange(2, max(sub["k"].max() + 1, 20))
    ax.plot(ks, 2.0 * (1.0 - 1.0 / ks), "k--", label="cota teórica 2(1 − 1/k)", linewidth=1.5)

    for algo in sorted(sub["algo"].unique()):
        for eps in sorted(sub["epsilon"].unique()):
            data = sub[(sub["algo"] == algo) & (sub["epsilon"] == eps)].sort_values("k")
            if data.empty:
                continue
            ax.plot(
                data["k"], data["ratio_vs_dp"],
                marker="o", markersize=4, linewidth=1.2,
                color=_algo_color(algo), alpha=0.4 + 0.6 * (eps == sub["epsilon"].min()),
                label=f"{algo}, eps={eps}",
            )

    ax.set_xlabel("k (número de terminales)")
    ax.set_ylabel("cociente greedy / óptimo")
    ax.set_title("Familia spider: cociente empírico se acerca a la cota 2(1 − 1/k)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quality_heatmap(df: pd.DataFrame, out_path: Path, *, algo: str = "mst_heuristic",
                         family: str = "euclidean") -> Path:
    from viz.heatmap import plot_heatmap as _heat

    csv_tmp = out_path.parent / f"_{family}_{algo}_subset.csv"
    df_sub = df[(df["instance_family"] == family) & (df["algo"] == algo)]
    df_sub.to_csv(csv_tmp, index=False)
    p = _heat(csv_tmp, out_path, algo=algo, family=family, value="ratio_vs_dp",
              title=f"{algo} sobre {family} — mediana del cociente")
    csv_tmp.unlink(missing_ok=True)
    return p


# ---------------------------------------------------------------------------
# Tablas LaTeX
# ---------------------------------------------------------------------------


def table_steinlib(df: pd.DataFrame, out_path: Path) -> Path:
    sub = df[df["instance_family"] == "steinlib_B"].copy()
    if sub.empty:
        out_path.write_text("% (sin datos SteinLib)\n", encoding="utf-8")
        return out_path

    pivot = sub.pivot_table(
        index=["n", "k"], columns="algo", values="cost", aggfunc="median"
    ).reset_index()

    cols = ["n", "k"] + [c for c in ("dreyfus_wagner", "mst_heuristic", "mehlhorn", "rsph")
                          if c in pivot.columns]
    pivot = pivot[cols]

    rename = {
        "dreyfus_wagner": "DP",
        "mst_heuristic": "KMB",
        "mehlhorn": "Mehlhorn",
        "rsph": "RSPH",
    }
    pivot = pivot.rename(columns=rename)

    header = " & ".join(pivot.columns) + r" \\"
    lines = [
        r"\begin{tabular}{" + "c" * len(pivot.columns) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        cells = []
        for c in pivot.columns:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, float) and not math.isnan(v):
                cells.append(f"{v:.2f}")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def table_dp_regression(df: pd.DataFrame, out_path: Path) -> Path:
    """Coeficientes con bootstrap CI para log(t_DP) vs k."""
    sub = df[(df["algo"] == "dreyfus_wagner") & (~df["timed_out"]) & (df["time_s_median"] > 0)]
    if len(sub) < 5:
        out_path.write_text("% (datos DP insuficientes para regresión)\n", encoding="utf-8")
        return out_path

    k = sub["k"].to_numpy()
    t = sub["time_s_median"].to_numpy()
    fit = fit_exponential_in_k(k, t)
    boot = bootstrap_ci(k, np.log(t), degree=1, n_resamples=1000, seed=0)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"coef & estimación & CI 95\% bajo & CI 95\% alto \\",
        r"\midrule",
        f"b (pendiente, log t = a + b·k) & {fit.coeffs[0]:.4f} & "
        f"{boot.coeffs_ci_low[0]:.4f} & {boot.coeffs_ci_high[0]:.4f} \\\\",
        f"a (intercepto)               & {fit.coeffs[1]:.4f} & "
        f"{boot.coeffs_ci_low[1]:.4f} & {boot.coeffs_ci_high[1]:.4f} \\\\",
        r"\midrule",
        f"R² ajustado & \\multicolumn{{3}}{{c}}{{{fit.r2_adj:.4f}}} \\\\",
        f"n puntos    & \\multicolumn{{3}}{{c}}{{{fit.n_points}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def analyze(csv_path: Path, fig_dir: Path, tab_dir: Path) -> dict:
    df = pd.read_csv(csv_path)
    _safe_mkdir(fig_dir)
    _safe_mkdir(tab_dir)

    outputs: dict[str, Path] = {}
    outputs["time_scatter"] = plot_time_scatter(df, fig_dir / "time_scatter.png")
    outputs["quality_box"] = plot_quality_box(df, fig_dir / "quality_box.png")
    outputs["spider_ratio"] = plot_spider_ratio(df, fig_dir / "spider_ratio.png")
    outputs["heatmap_kmb_euclidean"] = plot_quality_heatmap(
        df, fig_dir / "heatmap_kmb_euclidean.png",
        algo="mst_heuristic", family="euclidean",
    )
    outputs["table_steinlib"] = table_steinlib(df, tab_dir / "steinlib.tex")
    outputs["table_dp_regression"] = table_dp_regression(df, tab_dir / "regression_dp.tex")

    # Resumen agregado: lo guardamos también como CSV.
    rs = ratio_stats(df)
    summary_path = csv_path.parent / "summary_ratio.csv"
    rs.to_csv(summary_path, index=False)
    outputs["summary_csv"] = summary_path

    return outputs


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analiza el CSV del bench y genera figuras/tablas.")
    p.add_argument("--input", type=Path, default=Path("bench/results/raw.csv"))
    p.add_argument("--figures", type=Path, default=Path("docs/figures"))
    p.add_argument("--tables", type=Path, default=Path("docs/tables"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if not args.input.exists():
        print(f"[analyze] CSV no encontrado: {args.input}", file=sys.stderr)
        return 1
    outputs = analyze(args.input, args.figures, args.tables)
    print("[analyze] generados:")
    for name, p in outputs.items():
        print(f"  - {name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
