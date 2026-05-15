"""
Dashboard interactivo Steiner Tree — DP vs Greedy.

Tabs:
  1. Comparacion final   — 4 algoritmos lado a lado con Plotly interactivo.
  2. Paso a paso: GSVI  — animacion del nuevo algoritmo con tabla de ahorros.
  3. Paso a paso: RSPH  — animacion del crecimiento incremental.
  4. Tiempos y calidad  — graficos de los resultados del bench.

Uso:
    streamlit run dashboard/app.py

"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Asegurar que la raiz del repo este en sys.path (Streamlit pone
# primero la carpeta del script, no la raiz del proyecto).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from steiner import Instance, dreyfus_wagner, gsvi, mst_heuristic
from steiner.gsvi import gsvi_steps
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
from steiner.rsph import rsph, rsph_steps
from viz.interactive import (
    compute_plotly_layout,
    make_comparison_figure,
    make_network_figure,
)

# ---------------------------------------------------------------------------
# Constantes de color por algoritmo
# ---------------------------------------------------------------------------
ALGO_COLORS = {
    "Dreyfus-Wagner (DP)": "#1f77b4",
    "KMB (1981)": "#d62728",
    "Mehlhorn (1988)": "#2ca02c",
    "RSPH": "#9467bd",
    "GSVI": "#e377c2",
}

ALGO_FNS = {
    "Dreyfus-Wagner (DP)": dreyfus_wagner,
    "KMB (1981)": mst_heuristic,
    "Mehlhorn (1988)": mehlhorn,
    "RSPH": rsph,
    "GSVI": gsvi,
}

# ---------------------------------------------------------------------------
# Generadores de instancias
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _build_instance(family, **kwargs) -> Instance:
    builders = {
        "Spider (tight)": spider,
        "Double spider": double_spider,
        "Grid + shortcut": grid_with_shortcut,
        "Erdos-Renyi": random_er,
        "Euclidean": euclidean,
        "Geometric": geometric,
    }
    return builders[family](**kwargs)


def sidebar_instance() -> Instance | None:
    """Renderiza la barra lateral y devuelve la instancia elegida."""
    st.sidebar.header("Instancia")
    family = st.sidebar.selectbox(
        "Familia",
        ["Spider (tight)", "Double spider", "Grid + shortcut",
         "Erdos-Renyi", "Euclidean", "Geometric", "SteinLib (B)"],
    )

    if family == "Spider (tight)":
        k = st.sidebar.slider("k (terminales)", 2, 20, 5)
        eps = st.sidebar.select_slider("epsilon", [0.5, 0.2, 0.1, 0.05, 0.01], 0.05)
        return _build_instance("Spider (tight)", k=k, epsilon=eps)

    elif family == "Double spider":
        k1 = st.sidebar.slider("k1", 2, 10, 4)
        k2 = st.sidebar.slider("k2", 2, 10, 4)
        eps = st.sidebar.select_slider("epsilon", [0.5, 0.2, 0.1, 0.05, 0.01], 0.05)
        return double_spider(k1=k1, k2=k2, epsilon=eps)

    elif family == "Grid + shortcut":
        n = st.sidebar.slider("n (lado)", 3, 7, 4)
        sw = st.sidebar.slider("atajo diagonal (peso)", 0.0, 2.0, 0.5)
        return grid_with_shortcut(n=n, shortcut_weight=sw)

    elif family == "Erdos-Renyi":
        n = st.sidebar.slider("n", 5, 40, 15)
        p = st.sidebar.slider("p (densidad)", 0.1, 1.0, 0.5)
        k = st.sidebar.slider("k", 2, max(2, n), min(5, n))
        seed = int(st.sidebar.number_input("seed", value=0, step=1))
        return _build_instance("Erdos-Renyi", n=n, p=p, k=k, seed=seed)

    elif family == "Euclidean":
        n = st.sidebar.slider("n", 5, 30, 12)
        k = st.sidebar.slider("k", 2, n, min(5, n))
        seed = int(st.sidebar.number_input("seed", value=0, step=1))
        return _build_instance("Euclidean", n=n, k=k, seed=seed)

    elif family == "Geometric":
        n = st.sidebar.slider("n", 10, 40, 20)
        r = st.sidebar.slider("r (radio)", 0.1, 1.0, 0.35)
        k = st.sidebar.slider("k", 2, n, min(5, n))
        seed = int(st.sidebar.number_input("seed", value=0, step=1))
        return geometric(n=n, r=r, k=k, seed=seed)

    elif family == "SteinLib (B)":
        files = list_steinlib(_ROOT / "docs" / "steinlib_data")
        if not files:
            st.sidebar.warning(
                "No hay archivos en docs/steinlib_data/. "
                "Ejecuta: python -m bench.fetch_steinlib"
            )
            return None
        chosen = st.sidebar.selectbox("Instancia", [p.name for p in files])
        try:
            return parse_stp(_ROOT / "docs" / "steinlib_data" / chosen)
        except Exception as exc:
            st.sidebar.error(f"Error al parsear: {exc}")
            return None

    return None


# ---------------------------------------------------------------------------
# Tab 1: Comparacion final
# ---------------------------------------------------------------------------


def tab_comparison(instance: Instance) -> None:
    st.subheader(
        f"Comparacion de los 4 algoritmos — "
        f"n={instance.n}, m={instance.m}, k={instance.k}"
    )

    skip_dp = instance.k > 14
    if skip_dp:
        st.warning(
            f"k={instance.k} > 14: Dreyfus-Wagner omitido automaticamente "
            "(muy lento). Los greedies siguen ejecutandose."
        )

    chosen_algos = [a for a in ALGO_COLORS if a != "Dreyfus-Wagner (DP)" or not skip_dp]
    selected = st.multiselect("Algoritmos a mostrar", chosen_algos,
                              default=chosen_algos)
    if not selected:
        st.info("Selecciona al menos un algoritmo.")
        return

    results: dict = {}
    opt_cost: float | None = None
    layout = compute_plotly_layout(instance.graph, instance)

    with st.spinner("Calculando..."):
        for name in selected:
            fn = ALGO_FNS[name]
            t0 = time.perf_counter()
            cost, tree = fn(instance)
            elapsed = time.perf_counter() - t0
            results[name] = (cost, tree, ALGO_COLORS[name], elapsed)
            if name == "Dreyfus-Wagner (DP)":
                opt_cost = cost

    # Tabla de resultados
    rows = []
    for name, (cost, _, _, elapsed) in results.items():
        ratio = f"{cost/opt_cost:.4f}" if (opt_cost and opt_cost > 0) else "—"
        rows.append({
            "Algoritmo": name,
            "Costo del arbol": f"{cost:.4f}",
            "Tiempo (ms)": f"{elapsed*1000:.2f}",
            "Ratio vs DP": ratio,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Figura comparativa interactiva
    plot_results = {name: (cost, tree, color)
                   for name, (cost, tree, color, _) in results.items()}
    fig = make_comparison_figure(instance, plot_results, layout, opt_cost)
    st.plotly_chart(fig, use_container_width=True)

    # Leyenda visual
    st.caption(
        "**Como leer el grafico:** "
        "Cuadrados dorados = terminales. "
        "Circulos de color = puntos de Steiner en el arbol. "
        "Circulos grises = vertices no usados. "
        "Pasa el cursor sobre una arista o nodo para ver su peso / informacion. "
        "Usa la rueda del raton o los botones de Plotly para hacer zoom/pan."
    )


# ---------------------------------------------------------------------------
# Tab 2: Paso a paso — GSVI
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def _compute_gsvi_steps(instance_key: str, _instance: Instance) -> list[dict]:
    """Precalcula todos los pasos de GSVI (cacheado)."""
    steps = list(gsvi_steps(_instance))
    # Serializar solo lo necesario para el slider
    return steps


def tab_gsvi_steps(instance: Instance) -> None:
    st.subheader("GSVI — Insercion Codiciosa de Puntos de Steiner (paso a paso)")
    st.markdown(
        "**Diferencia clave vs KMB/Mehlhorn/RSPH:** estos deciden *caminos entre terminales*. "
        "GSVI decide *que vertices de Steiner incluir explicitamente* como hubs, "
        "calculando el ahorro de cada candidato antes de insertarlo."
    )

    layout = compute_plotly_layout(instance.graph, instance)

    with st.spinner("Calculando pasos de GSVI..."):
        inst_key = str(sorted(instance.terminals)) + str(instance.n)
        steps = _compute_gsvi_steps(inst_key, instance)

    if not steps:
        st.error("No se pudo ejecutar GSVI en esta instancia.")
        return

    n_steps = len(steps)
    step_labels = []
    for s in steps:
        if s["type"] == "initial":
            step_labels.append("0: Estado inicial (MST terminales)")
        elif s["type"] == "insert":
            v = s["inserted_vertex"]
            step_labels.append(f"{s['step_num']}: Insertar '{v}' (ahorro={s['best_savings']:.4f})")
        else:
            step_labels.append(f"{s['step_num']}: Resultado final (poda)")

    chosen_label = st.select_slider(
        "Paso del algoritmo", options=step_labels, value=step_labels[0]
    )
    step_idx = step_labels.index(chosen_label)
    step = steps[step_idx]

    # Columna izquierda: grafo interactivo
    col_graph, col_table = st.columns([3, 1])

    with col_graph:
        fig = make_network_figure(
            instance.graph,
            layout,
            instance,
            tree=step["tree"],
            candidate_savings=step["candidate_savings"] if step["type"] != "done" else None,
            inserted_vertex=step["inserted_vertex"],
            step_description=step["description"],
            title=f"GSVI — {chosen_label}",
            show_all_weights=(instance.m <= 40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Ahorros de candidatos**")
        if step["candidate_savings"]:
            cand_df = pd.DataFrame([
                {"Vertice": str(v), "Ahorro": f"{s:.4f}"}
                for v, s in sorted(step["candidate_savings"].items(),
                                   key=lambda x: -x[1])
                if s > 1e-10
            ])
            if not cand_df.empty:
                st.dataframe(cand_df, use_container_width=True, hide_index=True)
            else:
                st.info("Sin candidatos con ahorro > 0.")
        else:
            st.info("Sin candidatos en este paso.")

        st.markdown("**Vertices activos**")
        active = step["active_set"]
        terminals_active = active & instance.terminals
        steiner_active = active - instance.terminals
        st.markdown(f"Terminales: `{sorted(map(str, terminals_active))}`")
        if steiner_active:
            st.markdown(f"Steiner insertados: `{sorted(map(str, steiner_active))}`")

        if step["tree"] is not None:
            from steiner.graph_utils import tree_cost as tc
            st.metric("Costo actual", f"{tc(step['tree']):.4f}")

    st.caption(
        "**Leyenda:** Cuadrados dorados = terminales. "
        "Estrellas rojas = vertice recien insertado. "
        "Circulos naranjas = candidatos (tamanho ∝ ahorro). "
        "Linea azul = arbol actual. "
        "Hover sobre cualquier nodo o arista para mas detalles."
    )


# ---------------------------------------------------------------------------
# Tab 3: Paso a paso — RSPH
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def _compute_rsph_steps(instance_key: str, _instance: Instance) -> list[tuple]:
    return list(rsph_steps(_instance))


def tab_rsph_steps(instance: Instance) -> None:
    st.subheader("RSPH — Repetitive Shortest Path Heuristic (paso a paso)")
    st.markdown(
        "**Criterio greedy:** en cada iteracion conecta el terminal mas cercano "
        "al arbol parcial via su camino mas corto. "
        "Diferente a GSVI (que inserta hubs) y KMB (que usa MST de clausura metrica)."
    )

    layout = compute_plotly_layout(instance.graph, instance)

    with st.spinner("Calculando pasos de RSPH..."):
        inst_key = str(sorted(instance.terminals)) + str(instance.n)
        steps = _compute_rsph_steps(inst_key, instance)

    if not steps:
        st.error("No se pudo ejecutar RSPH.")
        return

    step_labels = []
    for step_num, target, tree in steps:
        if target is None:
            step_labels.append(f"{step_num}: Poda final")
        elif step_num == 0:
            step_labels.append(f"0: Terminal inicial '{target}'")
        else:
            from steiner.graph_utils import tree_cost as tc
            step_labels.append(
                f"{step_num}: Conectar '{target}' (costo={tc(tree):.4f})"
            )

    chosen_label = st.select_slider(
        "Paso del algoritmo", options=step_labels, value=step_labels[0]
    )
    step_idx = step_labels.index(chosen_label)
    _, target, tree = steps[step_idx]

    from steiner.graph_utils import tree_cost as tc

    description = (
        f"Paso {step_idx}: "
        + (f"terminal '{target}' conectado al arbol parcial. " if target else "Poda de hojas no terminales. ")
        + f"Costo acumulado = {tc(tree):.4f}."
    )

    fig = make_network_figure(
        instance.graph, layout, instance,
        tree=tree,
        step_description=description,
        title=f"RSPH — {chosen_label}",
        show_all_weights=(instance.m <= 40),
    )
    st.plotly_chart(fig, use_container_width=True)

    connected = set(tree.nodes) & instance.terminals
    remaining = instance.terminals - connected
    col1, col2, col3 = st.columns(3)
    col1.metric("Terminales conectados", len(connected))
    col2.metric("Terminales restantes", len(remaining))
    col3.metric("Costo actual", f"{tc(tree):.4f}")

    st.caption(
        "Linea azul = arbol parcial. Cuadrados dorados = terminales. "
        "Hover sobre nodos y aristas para ver detalles y pesos."
    )


# ---------------------------------------------------------------------------
# Tab 4: Bench — Tiempos y calidad
# ---------------------------------------------------------------------------


def tab_bench() -> None:
    st.subheader("Tiempos de ejecucion y calidad (resultados del experimento)")

    csv_path = _ROOT / "bench" / "results" / "raw.csv"
    if not csv_path.exists():
        st.warning(
            "No se encontro `bench/results/raw.csv`. "
            "Ejecuta primero: `python -m bench.run_experiments --quick`"
        )
        return

    df = pd.read_csv(csv_path)

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(_ROOT / "docs" / "figures" / "time_scatter.png"),
                 caption="Tiempo log-log: DP vs greedies", use_container_width=True)
    with col2:
        st.image(str(_ROOT / "docs" / "figures" / "spider_ratio.png"),
                 caption="Cociente empirico spider vs cota teorica", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image(str(_ROOT / "docs" / "figures" / "quality_box.png"),
                 caption="Cociente por familia y algoritmo", use_container_width=True)
    with col4:
        st.image(str(_ROOT / "docs" / "figures" / "heatmap_kmb_euclidean.png"),
                 caption="Heatmap (n, k) → cociente KMB/Euclidean", use_container_width=True)

    st.markdown("### Tabla resumen del ratio greedy/optimo")
    summary = (
        df[df["ratio_vs_dp"].notna() & (df["algo"] != "dreyfus_wagner")]
        .groupby(["instance_family", "algo"])["ratio_vs_dp"]
        .agg(mediana="median", maximo="max", minimo="min", n="count")
        .reset_index()
        .round(4)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Steiner Tree — DP vs Greedy",
        page_icon="🌳",
        layout="wide",
    )
    st.title("Steiner Tree: Programacion Dinamica vs Heuristicas Greedy")
    st.markdown(
        "Comparacion interactiva entre **Dreyfus-Wagner** (exacto) y cuatro heuristicas: "
        "**KMB**, **Mehlhorn**, **RSPH** y **GSVI** (insercion codiciosa de puntos de Steiner). "
        "Usa la barra lateral para elegir la instancia."
    )

    instance = sidebar_instance()
    if instance is None:
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Comparacion final",
        "Paso a paso: GSVI",
        "Paso a paso: RSPH",
        "Tiempos y calidad",
    ])

    with tab1:
        tab_comparison(instance)
    with tab2:
        tab_gsvi_steps(instance)
    with tab3:
        tab_rsph_steps(instance)
    with tab4:
        tab_bench()


if __name__ == "__main__":
    main()
