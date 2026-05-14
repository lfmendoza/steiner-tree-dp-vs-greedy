"""
Dashboard interactivo Streamlit — DP exacta vs heurísticas greedy.

Uso:
    streamlit run dashboard/app.py

La interfaz expone los cinco generadores de instancias y los cuatro
algoritmos. Para cada corrida renderiza los árboles producidos por
cada algoritmo en paneles lado a lado y muestra una tabla con tiempos
y cocientes vs Dreyfus–Wagner.
"""
from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path

# Streamlit pone primero en sys.path la carpeta del script (dashboard/),
# no la raíz del repo; sin esto falla: ModuleNotFoundError: steiner.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from steiner import Instance, dreyfus_wagner, mst_heuristic
from steiner.instances import (
    double_spider,
    euclidean,
    geometric,
    grid_with_shortcut,
    random_er,
    spider,
)
from steiner.instances.steinlib import list_steinlib, parse_stp
from steiner.mehlhorn import mehlhorn
from steiner.rsph import rsph
from viz.draw_tree import compute_layout, draw_instance, overlay_tree

import matplotlib.pyplot as plt


ALGOS = {
    "Dreyfus–Wagner (DP)": ("dreyfus_wagner", dreyfus_wagner, "#1f77b4"),
    "KMB (1981)": ("mst_heuristic", mst_heuristic, "#d62728"),
    "Mehlhorn (1988)": ("mehlhorn", mehlhorn, "#2ca02c"),
    "RSPH": ("rsph", rsph, "#9467bd"),
}


@st.cache_data(show_spinner=False)
def _build_random_er(n, p, k, seed):
    return random_er(n=n, p=p, k=k, seed=seed)


@st.cache_data(show_spinner=False)
def _build_euclidean(n, k, seed):
    return euclidean(n=n, k=k, seed=seed)


@st.cache_data(show_spinner=False)
def _build_geometric(n, r, k, seed):
    return geometric(n=n, r=r, k=k, seed=seed)


@st.cache_data(show_spinner=False)
def _build_spider(k, eps):
    return spider(k=k, epsilon=eps)


@st.cache_data(show_spinner=False)
def _build_double_spider(k1, k2, eps, bridge):
    return double_spider(k1=k1, k2=k2, epsilon=eps, bridge=bridge)


@st.cache_data(show_spinner=False)
def _build_grid(n, sw):
    return grid_with_shortcut(n=n, shortcut_weight=sw)


def build_instance(family: str) -> Instance | None:
    if family == "Erdős–Rényi":
        n = st.sidebar.slider("n", 5, 80, 20)
        p = st.sidebar.slider("p", 0.05, 1.0, 0.5)
        k = st.sidebar.slider("k", 2, max(2, n), min(5, n))
        seed = st.sidebar.number_input("seed", value=0, step=1)
        return _build_random_er(n, p, k, int(seed))
    if family == "Euclidean":
        n = st.sidebar.slider("n", 5, 60, 15)
        k = st.sidebar.slider("k", 2, n, min(5, n))
        seed = st.sidebar.number_input("seed", value=0, step=1)
        return _build_euclidean(n, k, int(seed))
    if family == "Geometric":
        n = st.sidebar.slider("n", 10, 80, 25)
        r = st.sidebar.slider("r", 0.1, 1.0, 0.35)
        k = st.sidebar.slider("k", 2, n, min(5, n))
        seed = st.sidebar.number_input("seed", value=0, step=1)
        return _build_geometric(n, r, k, int(seed))
    if family == "Spider (tight)":
        k = st.sidebar.slider("k", 2, 30, 6)
        eps = st.sidebar.select_slider("epsilon", [0.5, 0.2, 0.1, 0.05, 0.01], 0.05)
        return _build_spider(k, eps)
    if family == "Double spider":
        k1 = st.sidebar.slider("k1", 2, 12, 4)
        k2 = st.sidebar.slider("k2", 2, 12, 4)
        eps = st.sidebar.select_slider("epsilon", [0.5, 0.2, 0.1, 0.05, 0.01], 0.05)
        bridge = st.sidebar.slider("bridge weight", 0.1, 5.0, 1.0)
        return _build_double_spider(k1, k2, eps, bridge)
    if family == "Grid + shortcut":
        n = st.sidebar.slider("n (lado)", 3, 8, 4)
        sw = st.sidebar.slider("shortcut weight", 0.0, 2.0, 0.5)
        return _build_grid(n, sw)
    if family == "SteinLib (B)":
        files = list_steinlib()
        if not files:
            st.sidebar.warning("No hay archivos en docs/steinlib_data/. Corre bench/fetch_steinlib.py.")
            return None
        chosen = st.sidebar.selectbox("instancia", [p.name for p in files])
        return parse_stp(Path("docs/steinlib_data") / chosen)
    return None


def render_dashboard() -> None:
    st.set_page_config(page_title="Steiner DP vs Greedy", layout="wide")
    st.title("Steiner Tree — DP vs Greedy")
    st.markdown(
        "Comparación interactiva entre la programación dinámica exacta de "
        "**Dreyfus–Wagner (1971)** y tres heurísticas 2-aproximación:"
        " **KMB (1981)**, **Mehlhorn (1988)** y **RSPH** (Takahashi–Matsuyama / Voß)."
    )

    st.sidebar.header("Instancia")
    family = st.sidebar.selectbox(
        "Familia",
        [
            "Spider (tight)",
            "Double spider",
            "Grid + shortcut",
            "Erdős–Rényi",
            "Euclidean",
            "Geometric",
            "SteinLib (B)",
        ],
    )
    instance = build_instance(family)
    if instance is None:
        st.stop()

    st.sidebar.header("Algoritmos")
    chosen = st.sidebar.multiselect(
        "ejecutar",
        list(ALGOS.keys()),
        default=list(ALGOS.keys()),
    )

    run_dp_skip = instance.k > 14 and "Dreyfus–Wagner (DP)" in chosen
    if run_dp_skip:
        st.warning(
            f"k = {instance.k} > 14: DP omitida automáticamente para no bloquear "
            "la UI. Las heurísticas siguen ejecutándose."
        )
        chosen = [c for c in chosen if c != "Dreyfus–Wagner (DP)"]

    st.subheader(f"Instancia: {family} — n={instance.n}, m={instance.m}, k={instance.k}")

    results: dict = {}
    for label in chosen:
        key, fn, color = ALGOS[label]
        t0 = time.perf_counter()
        cost, tree = fn(instance)
        elapsed = time.perf_counter() - t0
        results[label] = (cost, tree, elapsed, color, key)

    if not results:
        st.info("Selecciona al menos un algoritmo.")
        return

    cols = st.columns(len(results))
    layout = compute_layout(instance.graph)
    for ax_col, (label, (cost, tree, elapsed, color, key)) in zip(cols, results.items()):
        with ax_col:
            fig, ax = plt.subplots(figsize=(5, 5))
            draw_instance(ax, instance, layout, show_weights=instance.m <= 25)
            overlay_tree(ax, tree, layout, color=color, label=label)
            ax.set_title(f"{label}\ncost = {cost:.4f}, t = {elapsed*1000:.1f} ms")
            ax.axis("off")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            st.image(buf.getvalue())

    st.subheader("Costos y cocientes")
    ref = results.get("Dreyfus–Wagner (DP)")
    ref_cost = ref[0] if ref is not None else None
    rows = []
    for label, (cost, _, elapsed, _, _) in results.items():
        ratio = (cost / ref_cost) if ref_cost else None
        rows.append({"algo": label, "cost": cost, "tiempo (s)": elapsed, "ratio vs DP": ratio})
    st.table(rows)


if __name__ == "__main__":
    render_dashboard()
