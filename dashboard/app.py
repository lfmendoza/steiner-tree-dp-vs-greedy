"""Dashboard: Steiner Tree — visualizacion y comparacion de algoritmos."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from steiner import dreyfus_wagner, gsvi, mst_heuristic
from steiner.gsvi import gsvi_steps
from steiner.graph_utils import tree_cost
from steiner.instances import euclidean, geometric, random_er, spider
from steiner.instances.pathological import double_spider, grid_with_shortcut
from steiner.instances.steinlib import list_steinlib, parse_stp
from steiner.mehlhorn import mehlhorn
from steiner.rsph import rsph, rsph_steps
from viz.interactive import compute_plotly_layout, make_comparison_figure, make_pyvis_html

ALGOS = {
    "KMB": mst_heuristic,
    "Mehlhorn": mehlhorn,
    "RSPH": rsph,
    "GSVI": gsvi,
    "DP exacto": dreyfus_wagner,
}

COLORS = {
    "KMB": "#d62728",
    "Mehlhorn": "#2ca02c",
    "RSPH": "#9467bd",
    "GSVI": "#e377c2",
    "DP exacto": "#1f77b4",
}

STEP_FNS = {"RSPH": rsph_steps, "GSVI": gsvi_steps}


# ---------------------------------------------------------------------------
# Construccion de instancias
# ---------------------------------------------------------------------------


def _build(family: str, **kw):
    mapping = {
        "Spider": spider,
        "Double Spider": double_spider,
        "Grid": grid_with_shortcut,
        "Erdos-Renyi": random_er,
        "Euclidean": euclidean,
        "Geometric": geometric,
    }
    return mapping[family](**kw)


# ---------------------------------------------------------------------------
# Calculo de pasos y tiempo (cacheados por parametros primitivos)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Calculando...")
def _get_steps(family: str, algo: str, kw_json: str):
    kw = json.loads(kw_json)
    inst = _build(family, **kw)
    layout = compute_plotly_layout(inst.graph, inst)

    if algo not in STEP_FNS:
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            cost, tree = ALGOS[algo](inst)
            samples.append(time.perf_counter() - t0)
        elapsed_ms = statistics.median(samples) * 1000
        steps = [{"step_num": 0, "type": "done", "tree": tree, "cost": cost,
                  "inserted_vertex": None, "candidate_savings": {}, "new_edges": [],
                  "description": "Resultado directo (algoritmo sin pasos intermedios)."}]
        return steps, layout, inst, elapsed_ms

    t0 = time.perf_counter()
    raw = list(STEP_FNS[algo](inst))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    steps = _normalize(raw, algo)
    return steps, layout, inst, elapsed_ms


def _normalize(raw, algo):
    out, prev = [], None
    if algo == "GSVI":
        for s in raw:
            t = s.get("tree")
            out.append({**s, "new_edges": _diff(prev, t),
                        "cost": tree_cost(t) if t else 0.0})
            prev = t
    elif algo == "RSPH":
        for num, target, tree in raw:
            if num == 0:
                desc = f"Terminal inicial: {target}"
                stype = "initial"
            elif target is None:
                desc = "Poda de hojas no terminales."
                stype = "done"
            else:
                desc = f"Terminal conectado: {target}"
                stype = "insert"
            out.append({"step_num": num, "tree": tree, "cost": tree_cost(tree),
                        "inserted_vertex": target, "candidate_savings": {},
                        "new_edges": _diff(prev, tree), "description": desc,
                        "type": stype})
            prev = tree
    return out


def _diff(prev, curr):
    if curr is None:
        return []
    cs = {frozenset(e) for e in curr.edges()}
    ps = {frozenset(e) for e in prev.edges()} if prev else set()
    return [tuple(fe) for fe in cs - ps]


# ---------------------------------------------------------------------------
# Tabla de log de pasos
# ---------------------------------------------------------------------------


def _build_log(steps: list, algo: str, inst) -> pd.DataFrame:
    """Genera la tabla de log exhaustiva para todos los pasos del algoritmo."""
    terminals = inst.terminals
    rows = []

    for s in steps:
        num = s.get("step_num", 0)
        stype = s.get("type", "")
        new_edges = s.get("new_edges", [])
        cost = s.get("cost", 0.0)
        ins_v = s.get("inserted_vertex")
        cands = s.get("candidate_savings") or {}

        # Accion
        if stype == "initial":
            accion = "MST inicial sobre terminales"
        elif stype == "insert":
            accion = "Insertar punto de Steiner" if algo == "GSVI" else "Conectar terminal"
        elif stype == "done" and ins_v:
            accion = "Conectar terminal (ultimo)"
        elif stype == "done" and not ins_v:
            accion = "Poda / arbol final"
        else:
            accion = "Resultado"

        # Nodo involucrado
        nodo = str(ins_v) if ins_v else "—"
        tipo_nodo = ("Terminal" if ins_v in terminals else "Punto de Steiner") if ins_v else "—"

        # Aristas nuevas con tipo de conexion
        edge_parts = []
        for u, v in new_edges:
            ut = "T" if u in terminals else "S"
            vt = "T" if v in terminals else "S"
            w = inst.graph[u][v]["weight"] if inst.graph.has_edge(u, v) else 0.0
            edge_parts.append(f"{u}({ut})-{v}({vt}) w={w:.3g}")
        aristas = " | ".join(edge_parts) if edge_parts else "—"

        # Criterio de decision
        if algo == "GSVI" and ins_v and stype == "insert":
            sav = cands.get(ins_v, s.get("best_savings", 0))
            criterio = f"max ahorro en MST clausura = {sav:.5f}"
        elif algo == "RSPH" and ins_v:
            criterio = "terminal mas cercano al arbol parcial (Dijkstra multifuente)"
        elif stype == "initial":
            criterio = "MST sobre clausura metrica de terminales"
        elif stype == "done" and not ins_v:
            criterio = "eliminar hojas no terminales iterativamente"
        else:
            criterio = "—"

        rows.append({
            "Paso": num,
            "Accion": accion,
            "Nodo": nodo,
            "Tipo de nodo": tipo_nodo,
            "Aristas nuevas  T=terminal S=Steiner": aristas,
            "Costo acumulado": round(cost, 5),
            "Criterio de decision": criterio,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar():
    st.sidebar.header("Configuracion")
    family = st.sidebar.selectbox(
        "Familia",
        ["Spider", "Erdos-Renyi", "Euclidean", "Geometric",
         "Double Spider", "Grid", "SteinLib"],
    )

    kw: dict = {}
    if family == "Spider":
        kw["k"] = st.sidebar.slider("k (terminales)", 2, 15, 5)
        kw["epsilon"] = float(
            st.sidebar.select_slider("epsilon", [0.5, 0.2, 0.1, 0.05, 0.01], 0.05)
        )
    elif family == "Erdos-Renyi":
        n = st.sidebar.slider("n", 5, 30, 12)
        kw["n"] = n
        kw["p"] = st.sidebar.slider("p", 0.15, 1.0, 0.45)
        kw["k"] = st.sidebar.slider("k", 2, n, min(5, n))
        kw["seed"] = int(st.sidebar.number_input("seed", 0, 9999, 0))
    elif family == "Euclidean":
        n = st.sidebar.slider("n", 5, 25, 10)
        kw["n"] = n
        kw["k"] = st.sidebar.slider("k", 2, n, min(5, n))
        kw["seed"] = int(st.sidebar.number_input("seed", 0, 9999, 0))
    elif family == "Geometric":
        n = st.sidebar.slider("n", 8, 30, 15)
        kw["n"] = n
        kw["r"] = st.sidebar.slider("r", 0.15, 0.9, 0.4)
        kw["k"] = st.sidebar.slider("k", 2, n, min(5, n))
        kw["seed"] = int(st.sidebar.number_input("seed", 0, 9999, 0))
    elif family == "Double Spider":
        kw["k1"] = st.sidebar.slider("k1", 2, 8, 4)
        kw["k2"] = st.sidebar.slider("k2", 2, 8, 4)
        kw["epsilon"] = float(
            st.sidebar.select_slider("epsilon", [0.5, 0.2, 0.1, 0.05, 0.01], 0.05)
        )
    elif family == "Grid":
        kw["n"] = st.sidebar.slider("n (lado)", 3, 7, 4)
        kw["shortcut_weight"] = st.sidebar.slider("peso atajo diagonal", 0.0, 2.0, 0.5)

    algo = st.sidebar.radio("Algoritmo", list(ALGOS.keys()),
                            label_visibility="collapsed")
    return family, kw, algo


# ---------------------------------------------------------------------------
# Vista principal: grafo final + tabla de log
# ---------------------------------------------------------------------------


def _show_analysis(family, algo, kw):
    kw_json = json.dumps(kw, sort_keys=True)
    steps, layout, inst, elapsed_ms = _get_steps(family, algo, kw_json)

    final_step = steps[-1]
    final_tree = final_step.get("tree")
    final_cost = final_step.get("cost", 0.0)

    # Metricas de cabecera
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (vertices)", inst.n)
    c2.metric("k (terminales)", inst.k)
    c3.metric("Costo del arbol", f"{final_cost:.4f}")
    c4.metric("Tiempo de ejecucion", f"{elapsed_ms:.3f} ms")

    # Grafo: arbol final
    html = make_pyvis_html(
        inst.graph, layout, inst,
        tree=final_tree,
        height=490,
        dark=True,
    )
    components.html(html, height=505, scrolling=False)
    st.caption(
        "Arbol de Steiner final. "
        "Cuadrados dorados = terminales. "
        "Circulos azules = puntos de Steiner incluidos. "
        "Hover sobre nodos y aristas para ver pesos y tipo."
    )

    # Tabla de log
    st.markdown("**Log de ejecucion del algoritmo**")
    df = _build_log(steps, algo, inst)
    st.dataframe(df, use_container_width=True, hide_index=True)

    n_steps = len(steps)
    n_steiner = len(set(final_tree.nodes) - inst.terminals) if final_tree else 0
    st.caption(
        f"Pasos totales: {n_steps}  |  "
        f"Puntos de Steiner usados: {n_steiner}  |  "
        f"Aristas en el arbol: {final_tree.number_of_edges() if final_tree else 0}  |  "
        f"Aristas en el grafo original: {inst.m}"
    )


# ---------------------------------------------------------------------------
# Vista de comparacion
# ---------------------------------------------------------------------------


def _show_comparison(family, kw):
    inst = _build(family, **kw)
    layout = compute_plotly_layout(inst.graph, inst)

    skip_dp = inst.k > 13
    to_run = [a for a in ALGOS if not (a == "DP exacto" and skip_dp)]
    if skip_dp:
        st.info(f"DP omitido (k={inst.k} > 13).")

    selected = st.multiselect("Algoritmos a comparar", to_run, default=to_run)
    if not selected:
        return

    if st.button("Calcular"):
        results: dict = {}
        opt = None
        rows = []
        progress = st.progress(0)

        for i, name in enumerate(selected):
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                cost, tree = ALGOS[name](inst)
                samples.append(time.perf_counter() - t0)
            elapsed = statistics.median(samples) * 1000
            results[name] = (cost, tree, COLORS[name])
            if name == "DP exacto":
                opt = cost
            rows.append({
                "Algoritmo": name,
                "Costo": round(cost, 5),
                "Tiempo mediano (ms)": round(elapsed, 3),
                "Ratio vs DP": round(cost / opt, 5) if opt else "—",
                "Puntos de Steiner": len(set(tree.nodes) - inst.terminals),
                "Aristas del arbol": tree.number_of_edges(),
            })
            progress.progress((i + 1) / len(selected))
        progress.empty()

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        fig = make_comparison_figure(inst, results, layout, opt)
        st.plotly_chart(
            fig, use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Steiner Tree", layout="wide")
    st.title("Steiner Tree")

    family, kw, algo = _sidebar()

    if family == "SteinLib":
        files = list_steinlib(_ROOT / "docs" / "steinlib_data")
        if not files:
            st.error("Ejecuta: python -m bench.fetch_steinlib")
            return
        chosen = st.sidebar.selectbox("Archivo .stp", [p.name for p in files])
        inst = parse_stp(_ROOT / "docs" / "steinlib_data" / chosen)
        skip_dp = inst.k > 13
        rows = []
        for name in ALGOS:
            if name == "DP exacto" and skip_dp:
                continue
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                cost, tree = ALGOS[name](inst)
                samples.append(time.perf_counter() - t0)
            rows.append({"Algoritmo": name, "Costo": round(cost, 5),
                         "Tiempo (ms)": round(statistics.median(samples) * 1000, 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        return

    tab1, tab2 = st.tabs(["Paso a paso", "Comparacion"])
    with tab1:
        _show_analysis(family, algo, kw)
    with tab2:
        _show_comparison(family, kw)


if __name__ == "__main__":
    main()
