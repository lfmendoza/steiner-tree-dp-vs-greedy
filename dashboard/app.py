"""Dashboard: Steiner Tree paso a paso y comparacion de algoritmos."""
from __future__ import annotations

import json
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
# Calculo de pasos (cacheado por parametros primitivos)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Calculando pasos...")
def _get_steps(family: str, algo: str, kw_json: str):
    kw = json.loads(kw_json)
    inst = _build(family, **kw)
    layout = compute_plotly_layout(inst.graph, inst)

    if algo not in STEP_FNS:
        cost, tree = ALGOS[algo](inst)
        return (
            [{"step_num": 0, "tree": tree, "cost": cost,
              "inserted_vertex": None, "candidate_savings": {},
              "new_edges": [], "description": f"Resultado. Costo = {cost:.4f}"}],
            layout, inst,
        )

    raw = list(STEP_FNS[algo](inst))
    steps = _normalize(raw, algo)
    return steps, layout, inst


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
            elif target is None:
                desc = "Poda de hojas no terminales."
            else:
                desc = f"Terminal conectado: {target}"
            out.append({"step_num": num, "tree": tree, "cost": tree_cost(tree),
                        "inserted_vertex": target, "candidate_savings": {},
                        "new_edges": _diff(prev, tree), "description": desc})
            prev = tree
    return out


def _diff(prev, curr):
    if curr is None:
        return []
    cs = {frozenset(e) for e in curr.edges()}
    ps = {frozenset(e) for e in prev.edges()} if prev else set()
    return [tuple(fe) for fe in cs - ps]


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
        kw["shortcut_weight"] = st.sidebar.slider("peso atajo", 0.0, 2.0, 0.5)

    algo = st.sidebar.radio(
        "Algoritmo",
        list(ALGOS.keys()),
        label_visibility="collapsed",
    )

    return family, kw, algo


# ---------------------------------------------------------------------------
# Vista por pasos
# ---------------------------------------------------------------------------


def _show_steps(family, algo, kw):
    kw_json = json.dumps(kw, sort_keys=True)
    steps, layout, inst = _get_steps(family, algo, kw_json)
    n = len(steps)

    st.caption(f"n={inst.n}  m={inst.m}  k={inst.k}")

    params_hash = hash(f"{family}{algo}{kw_json}")
    if n > 1:
        step_idx = st.slider("Paso", 0, n - 1, 0, key=f"s{params_hash}")
    else:
        step_idx = 0

    s = steps[step_idx]
    tree = s.get("tree")

    html = make_pyvis_html(
        inst.graph, layout, inst,
        tree=tree,
        new_edges=s.get("new_edges", []),
        candidate_savings=s.get("candidate_savings") or None,
        inserted_vertex=s.get("inserted_vertex"),
        height=510,
        dark=True,
    )
    components.html(html, height=525, scrolling=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Costo", f"{s.get('cost', 0):.4f}")
    c2.metric("Paso", f"{step_idx + 1} / {n}")
    if tree:
        conn = len(set(tree.nodes) & inst.terminals)
        c3.metric("Terminales", f"{conn}/{inst.k}")

    if s.get("description"):
        st.write(s["description"])

    cands = {v: sv for v, sv in (s.get("candidate_savings") or {}).items() if sv > 1e-10}
    if cands:
        top = sorted(cands.items(), key=lambda x: -x[1])[:8]
        df = pd.DataFrame(
            [{"Vertice": str(v), "Ahorro": round(sv, 5),
              "Elegido": "si" if v == s.get("inserted_vertex") else ""}
             for v, sv in top]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Vista de comparacion
# ---------------------------------------------------------------------------


def _show_comparison(family, kw):
    kw_json = json.dumps(kw, sort_keys=True)
    inst = _build(family, **kw)
    layout = compute_plotly_layout(inst.graph, inst)

    skip_dp = inst.k > 13
    to_run = [a for a in ALGOS if not (a == "DP exacto" and skip_dp)]
    if skip_dp:
        st.info(f"DP omitida (k={inst.k} > 13).")

    selected = st.multiselect("Algoritmos", to_run, default=to_run)
    if not selected:
        return

    if st.button("Calcular"):
        results: dict = {}
        opt = None
        for name in selected:
            cost, tree = ALGOS[name](inst)
            results[name] = (cost, tree, COLORS[name])
            if name == "DP exacto":
                opt = cost

        rows = []
        for name, (cost, _, _) in results.items():
            rows.append({
                "Algoritmo": name,
                "Costo": round(cost, 4),
                "Ratio vs DP": round(cost / opt, 4) if opt else "-",
            })
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
            st.error("Archivos no encontrados. Ejecuta: python -m bench.fetch_steinlib")
            return
        chosen = st.sidebar.selectbox("Archivo .stp", [p.name for p in files])
        inst = parse_stp(_ROOT / "docs" / "steinlib_data" / chosen)
        st.caption(f"n={inst.n}  m={inst.m}  k={inst.k}")
        skip_dp = inst.k > 13
        for name in ALGOS:
            if name == "DP exacto" and skip_dp:
                continue
            cost, _ = ALGOS[name](inst)
            st.write(f"{name}: {cost:.4f}")
        return

    tab1, tab2 = st.tabs(["Paso a paso", "Comparacion"])
    with tab1:
        _show_steps(family, algo, kw)
    with tab2:
        _show_comparison(family, kw)


if __name__ == "__main__":
    main()
