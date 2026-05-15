"""
Steiner Tree Visualizer — Dashboard interactivo.

Un solo grafo a la vez, animacion paso a paso con controles de reproduccion,
panel de datos lateral y parametros modificables antes de ejecutar.

Motor de grafo: PyVis (vis.js) → draggable, zoom fluido, hover HTML.

Uso:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from steiner import Instance, dreyfus_wagner, gsvi, mst_heuristic
from steiner.gsvi import gsvi_steps
from steiner.instances import (
    double_spider, euclidean, geometric, grid_with_shortcut, random_er, spider,
)
from steiner.instances.steinlib import list_steinlib, parse_stp
from steiner.mehlhorn import mehlhorn
from steiner.rsph import rsph, rsph_steps
from viz.interactive import (
    compute_plotly_layout,
    make_comparison_figure,
    make_network_figure,
    make_pyvis_html,
)

# ---------------------------------------------------------------------------
# CSS global — apariencia moderna
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* Botones de control de animacion */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    border-radius: 8px;
    font-size: 18px;
    padding: 6px 14px;
    transition: background 0.15s;
}
/* Boton RUN destacado */
.run-btn > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.45) !important;
    letter-spacing: 0.04em;
}
/* Tarjeta de paso */
.step-card {
    background: #1E1E2E;
    border-radius: 10px;
    padding: 14px 18px;
    color: #ECECEC;
    font-size: 14px;
    line-height: 1.6;
    border-left: 4px solid #667eea;
    margin-top: 8px;
}
/* Panel de metricas */
.metric-box {
    background: #181825;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    color: #CDD6F4;
}
.metric-box .val { font-size: 24px; font-weight: 700; color: #89B4FA; }
.metric-box .lbl { font-size: 11px; color: #A6ADC8; }
/* Ocultar footer de Streamlit */
footer { visibility: hidden; }
</style>
"""

ALGO_META = {
    "GSVI — Insercion de Steiner": {
        "fn": gsvi,
        "steps_fn": gsvi_steps,
        "color": "#F48FB1",
        "icon": "⭐",
        "desc": "Inserta explicitamente vertices de Steiner como hubs. "
                "Criterio: maximo ahorro en el MST de la clausura extendida.",
    },
    "RSPH — Shortest Path Incremental": {
        "fn": rsph,
        "steps_fn": rsph_steps,
        "color": "#CE93D8",
        "icon": "🔗",
        "desc": "Crece el arbol conectando cada vez el terminal mas cercano "
                "al arbol parcial via su camino mas corto.",
    },
    "KMB — Clausura Metrica": {
        "fn": mst_heuristic,
        "steps_fn": None,
        "color": "#EF9A9A",
        "icon": "🕸️",
        "desc": "Construye el MST sobre la clausura metrica de terminales, "
                "expande caminos y poda.",
    },
    "Mehlhorn — Voronoi": {
        "fn": mehlhorn,
        "steps_fn": None,
        "color": "#80CBC4",
        "icon": "🗺️",
        "desc": "Dijkstra multifuente asigna Voronoi; MST sobre grafo auxiliar "
                "entre regiones distintas. O(m + n log n).",
    },
    "Dreyfus-Wagner — DP Exacta": {
        "fn": dreyfus_wagner,
        "steps_fn": None,
        "color": "#90CAF9",
        "icon": "🎯",
        "desc": "Algoritmo exacto. Complejidad O(3^k · n). Recomendado k ≤ 13.",
    },
}

# ---------------------------------------------------------------------------
# Construccion de instancias
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_instance(family: str, **kw) -> Instance:
    builders = {
        "Spider": spider, "Double Spider": double_spider,
        "Grid+Shortcut": grid_with_shortcut, "Erdos-Renyi": random_er,
        "Euclidean": euclidean, "Geometric": geometric,
    }
    return builders[family](**kw)


@st.cache_data(show_spinner=False)
def _cached_gsvi_steps(key: str, _inst: Instance) -> list[dict]:
    return list(gsvi_steps(_inst))


@st.cache_data(show_spinner=False)
def _cached_rsph_steps(key: str, _inst: Instance) -> list[tuple]:
    return list(rsph_steps(_inst))


# ---------------------------------------------------------------------------
# Sidebar — instancia + algoritmo + RUN
# ---------------------------------------------------------------------------


def _sidebar() -> tuple[Instance | None, str, float]:
    st.sidebar.markdown("## ⚙️ Configuracion")

    st.sidebar.markdown("### 📐 Instancia")
    family = st.sidebar.selectbox(
        "Familia", ["Spider", "Double Spider", "Grid+Shortcut",
                    "Erdos-Renyi", "Euclidean", "Geometric", "SteinLib (B)"],
        label_visibility="collapsed",
    )

    inst: Instance | None = None

    if family == "Spider":
        k = st.sidebar.slider("Terminales k", 2, 20, 5)
        eps = st.sidebar.select_slider("ε", [0.50, 0.20, 0.10, 0.05, 0.01], 0.05)
        inst = _cached_instance("Spider", k=k, epsilon=eps)

    elif family == "Double Spider":
        k1 = st.sidebar.slider("k₁", 2, 10, 4)
        k2 = st.sidebar.slider("k₂", 2, 10, 4)
        eps = st.sidebar.select_slider("ε", [0.50, 0.20, 0.10, 0.05, 0.01], 0.05)
        inst = double_spider(k1=k1, k2=k2, epsilon=eps)

    elif family == "Grid+Shortcut":
        n = st.sidebar.slider("Lado n", 3, 7, 4)
        sw = st.sidebar.slider("Peso atajo diagonal", 0.0, 2.0, 0.5)
        inst = grid_with_shortcut(n=n, shortcut_weight=sw)

    elif family == "Erdos-Renyi":
        n = st.sidebar.slider("n (nodos)", 5, 35, 14)
        p = st.sidebar.slider("p (densidad)", 0.15, 1.0, 0.45)
        k = st.sidebar.slider("k (terminales)", 2, n, min(5, n))
        seed = int(st.sidebar.number_input("Semilla", 0, 9999, 0))
        inst = _cached_instance("Erdos-Renyi", n=n, p=p, k=k, seed=seed)

    elif family == "Euclidean":
        n = st.sidebar.slider("n (puntos)", 5, 25, 12)
        k = st.sidebar.slider("k (terminales)", 2, n, min(5, n))
        seed = int(st.sidebar.number_input("Semilla", 0, 9999, 0))
        inst = _cached_instance("Euclidean", n=n, k=k, seed=seed)

    elif family == "Geometric":
        n = st.sidebar.slider("n (nodos)", 8, 30, 18)
        r = st.sidebar.slider("r (radio)", 0.15, 0.90, 0.40)
        k = st.sidebar.slider("k (terminales)", 2, n, min(5, n))
        seed = int(st.sidebar.number_input("Semilla", 0, 9999, 0))
        inst = geometric(n=n, r=r, k=k, seed=seed)

    elif family == "SteinLib (B)":
        files = list_steinlib(_ROOT / "docs" / "steinlib_data")
        if not files:
            st.sidebar.warning(
                "Ejecuta primero:\n`python -m bench.fetch_steinlib`"
            )
            return None, "", 1.0
        chosen = st.sidebar.selectbox("Instancia", [p.name for p in files])
        try:
            inst = parse_stp(_ROOT / "docs" / "steinlib_data" / chosen)
        except Exception as exc:
            st.sidebar.error(str(exc))
            return None, "", 1.0

    st.sidebar.divider()
    st.sidebar.markdown("### 🧮 Algoritmo")
    algo_name = st.sidebar.radio(
        "Algoritmo", list(ALGO_META.keys()), label_visibility="collapsed"
    )
    meta = ALGO_META[algo_name]
    st.sidebar.caption(f"{meta['icon']} {meta['desc']}")

    st.sidebar.divider()
    speed = st.sidebar.slider(
        "⏱ Velocidad de animacion (seg/paso)", 0.30, 2.5, 0.9, 0.05
    )

    st.sidebar.divider()
    st.sidebar.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run = st.sidebar.button("▶ EJECUTAR ALGORITMO", type="primary")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    if run:
        st.session_state["run_triggered"] = True
        st.session_state["run_algo"] = algo_name
        st.session_state["run_inst_key"] = id(inst)
        st.session_state["steps"] = None
        st.session_state["step_idx"] = 0
        st.session_state["playing"] = False
        st.session_state["result"] = None

    return inst, algo_name, speed


# ---------------------------------------------------------------------------
# Calculo de pasos + diff de aristas
# ---------------------------------------------------------------------------


def _compute_steps(inst: Instance, algo_name: str) -> list[dict]:
    """Convierte la salida de cada generador a un formato unificado."""
    meta = ALGO_META[algo_name]
    steps_fn = meta["steps_fn"]

    if steps_fn is None:
        # Algoritmos sin pasos: mostrar solo el resultado final
        fn = meta["fn"]
        cost, tree = fn(inst)
        return [{"step_num": 0, "type": "done", "tree": tree, "cost": cost,
                 "inserted_vertex": None, "candidate_savings": {},
                 "description": f"Resultado final. Costo = {cost:.4f}"}]

    raw = list(steps_fn(inst))

    if algo_name.startswith("GSVI"):
        # gsvi_steps ya devuelve dicts con el formato correcto
        unified = []
        prev_tree = None
        for s in raw:
            t = s.get("tree")
            new_edges = _diff_edges(prev_tree, t)
            unified.append({**s, "new_edges": new_edges,
                            "cost": _tree_cost(t) if t else 0.0})
            prev_tree = t
        return unified

    if algo_name.startswith("RSPH"):
        unified = []
        prev_tree = None
        for step_num, target, tree in raw:
            from steiner.graph_utils import tree_cost as tc
            new_edges = _diff_edges(prev_tree, tree)
            desc = (
                f"Paso {step_num}: terminal inicial '{target}'"
                if step_num == 0 and target else
                f"Paso {step_num}: conectado '{target}'" if target else
                "Poda de hojas no terminales → árbol final."
            )
            unified.append({
                "step_num": step_num, "type": "initial" if step_num == 0 else
                ("done" if target is None else "insert"),
                "tree": tree, "cost": _tree_cost(tree),
                "inserted_vertex": target,
                "candidate_savings": {},
                "new_edges": new_edges,
                "description": desc,
            })
            prev_tree = tree
        return unified

    return []


def _diff_edges(prev: nx.Graph | None, curr: nx.Graph | None) -> list[tuple]:
    if curr is None:
        return []
    curr_set = set(map(frozenset, curr.edges()))
    prev_set = set(map(frozenset, prev.edges())) if prev else set()
    new = curr_set - prev_set
    result = []
    for fe in new:
        u, v = tuple(fe)
        result.append((u, v))
    return result


def _tree_cost(tree: nx.Graph | None) -> float:
    if tree is None:
        return 0.0
    return float(sum(d.get("weight", 0.0) for _, _, d in tree.edges(data=True)))


# ---------------------------------------------------------------------------
# Renderizado de un paso
# ---------------------------------------------------------------------------


def _render_step(
    step: dict,
    inst: Instance,
    layout: dict,
    graph_ph,
    info_ph,
    ctrl_ph,
    algo_name: str,
    n_steps: int,
) -> None:
    idx = step["step_num"]
    tree = step.get("tree")
    new_edges = step.get("new_edges", [])
    cand = step.get("candidate_savings", {})
    ins_v = step.get("inserted_vertex")
    desc = step.get("description", "")
    cost = step.get("cost", 0.0)

    # Grafo PyVis
    with graph_ph.container():
        html = make_pyvis_html(
            inst.graph, layout, inst,
            tree=tree, new_edges=new_edges,
            candidate_savings=cand if cand else None,
            inserted_vertex=ins_v,
            height=530,
            dark=True,
        )
        components.html(html, height=545, scrolling=False)

    # Info / descripcion
    with info_ph.container():
        connected = len(set(tree.nodes) & inst.terminals) if tree else 0
        remaining = inst.k - connected

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Costo actual", f"{cost:.4f}")
        c2.metric("Terminales", f"{connected} / {inst.k}")
        c3.metric("Paso", f"{idx + 1} / {n_steps}")
        tipo = step.get("type", "")
        c4.metric("Estado", {"initial": "Inicio", "insert": "Insertando",
                              "done": "Finalizado"}.get(tipo, tipo))

        if desc:
            st.markdown(
                f'<div class="step-card">{desc}</div>',
                unsafe_allow_html=True,
            )

        # Tabla de candidatos (GSVI)
        if cand:
            top = sorted(
                [(v, s) for v, s in cand.items() if s > 1e-10],
                key=lambda x: -x[1],
            )[:10]
            if top:
                st.markdown("**Candidatos Steiner (top 10 por ahorro):**")
                df = pd.DataFrame(
                    [{"Vértice": str(v), "Ahorro": f"{s:.5f}",
                      "Elegido": "✅" if v == ins_v else ""} for v, s in top]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

    # Controles — se actualizan en el mismo placeholder
    with ctrl_ph.container():
        _render_controls(idx, n_steps)


def _render_controls(current_idx: int, n_steps: int) -> None:
    """Fila de botones de control de animacion."""
    c0, c1, c2, c3, c4, _, label_c = st.columns([1, 1, 2, 1, 1, 3, 3])
    with c0:
        if st.button("⏮", help="Inicio", key=f"ctl_reset_{current_idx}"):
            st.session_state["step_idx"] = 0
            st.session_state["playing"] = False
            st.rerun()
    with c1:
        if st.button("⏪", help="Paso anterior", key=f"ctl_prev_{current_idx}"):
            st.session_state["step_idx"] = max(0, current_idx - 1)
            st.session_state["playing"] = False
            st.rerun()
    with c2:
        playing = st.session_state.get("playing", False)
        lbl = "⏸ Pausa" if playing else "▶ Play"
        if st.button(lbl, key=f"ctl_play_{current_idx}"):
            st.session_state["playing"] = not playing
            st.rerun()
    with c3:
        if st.button("⏩", help="Paso siguiente", key=f"ctl_next_{current_idx}"):
            st.session_state["step_idx"] = min(n_steps - 1, current_idx + 1)
            st.session_state["playing"] = False
            st.rerun()
    with c4:
        if st.button("⏭", help="Ultimo paso", key=f"ctl_end_{current_idx}"):
            st.session_state["step_idx"] = n_steps - 1
            st.session_state["playing"] = False
            st.rerun()
    with label_c:
        frac = (current_idx + 1) / n_steps
        bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
        st.markdown(
            f"`{bar}` **{current_idx + 1}/{n_steps}**",
            help="Progreso del algoritmo",
        )


# ---------------------------------------------------------------------------
# Vista de resultados del bench
# ---------------------------------------------------------------------------


def _bench_panel() -> None:
    st.subheader("📊 Resultados del experimento")
    csv_path = _ROOT / "bench" / "results" / "raw.csv"
    if not csv_path.exists():
        st.info("Genera los datos con:\n```\npython -m bench.run_experiments --quick\n```")
        return

    df = pd.read_csv(csv_path)
    col1, col2 = st.columns(2)
    figs_dir = _ROOT / "docs" / "figures"
    for col, fname, caption in [
        (col1, "time_scatter.png", "Tiempo log-log"),
        (col2, "spider_ratio.png", "Cociente spider vs cota"),
    ]:
        p = figs_dir / fname
        if p.exists():
            col.image(str(p), caption=caption, use_container_width=True)

    col3, col4 = st.columns(2)
    for col, fname, caption in [
        (col3, "quality_box.png", "Calidad por familia"),
        (col4, "heatmap_kmb_euclidean.png", "Heatmap (n,k) KMB"),
    ]:
        p = figs_dir / fname
        if p.exists():
            col.image(str(p), caption=caption, use_container_width=True)

    st.markdown("### Ratio greedy / optimo por familia y algoritmo")
    summary = (
        df[df["ratio_vs_dp"].notna() & (df["algo"] != "dreyfus_wagner")]
        .groupby(["instance_family", "algo"])["ratio_vs_dp"]
        .agg(mediana="median", maximo="max", minimo="min", n="count")
        .reset_index().round(4)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Steiner Tree Visualizer",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Tabs principales (no condicionadas al RUN)
    tab_viz, tab_compare, tab_bench = st.tabs([
        "🔬 Visualizador paso a paso",
        "⚖️ Comparacion de algoritmos",
        "📊 Datos del experimento",
    ])

    inst, algo_name, speed = _sidebar()
    if inst is None:
        return

    # ------------------------------------------------------------------ #
    # Tab 1: Visualizador paso a paso
    # ------------------------------------------------------------------ #
    with tab_viz:
        meta = ALGO_META[algo_name]
        st.markdown(
            f"## {meta['icon']} {algo_name} &nbsp;&nbsp;"
            f"<small style='color:#888'>n={inst.n} · m={inst.m} · k={inst.k}</small>",
            unsafe_allow_html=True,
        )
        st.caption(meta["desc"])

        # Calcular pasos cuando se presiona RUN
        if st.session_state.get("run_triggered"):
            st.session_state["run_triggered"] = False
            with st.spinner(f"Ejecutando {algo_name}..."):
                inst_key = f"{algo_name}_{inst.n}_{inst.k}_{sorted(inst.terminals)}"
                steps = _compute_steps(inst, algo_name)
                st.session_state["steps"] = steps
                st.session_state["step_idx"] = 0
                st.session_state["playing"] = False
                # Layout fijo para toda la sesion
                st.session_state["layout"] = compute_plotly_layout(inst.graph, inst)

        steps = st.session_state.get("steps")

        if not steps:
            # Estado inicial: mostrar instrucciones
            st.info(
                "👈 Configura la instancia y el algoritmo en la barra lateral, "
                "luego presiona **▶ EJECUTAR ALGORITMO**."
            )
            # Preview del grafo sin arbol
            layout0 = compute_plotly_layout(inst.graph, inst)
            html_preview = make_pyvis_html(
                inst.graph, layout0, inst, height=500, dark=True
            )
            components.html(html_preview, height=515, scrolling=False)
            st.caption(
                "🟨 Cuadrados = terminales &nbsp;|&nbsp; "
                "⚪ Círculos = posibles puntos de Steiner &nbsp;|&nbsp; "
                "Hover sobre nodos/aristas para ver detalles &nbsp;|&nbsp; "
                "Zoom con rueda · Pan arrastrando el fondo"
            )
            return

        layout = st.session_state.get("layout") or compute_plotly_layout(inst.graph, inst)
        n_steps = len(steps)
        step_idx = st.session_state.get("step_idx", 0)
        step_idx = min(max(step_idx, 0), n_steps - 1)

        # Placeholders fijos para evitar saltos en el layout
        ctrl_ph = st.empty()
        graph_ph = st.empty()
        info_ph = st.empty()

        # Auto-play
        if st.session_state.get("playing"):
            for i in range(step_idx, n_steps):
                st.session_state["step_idx"] = i
                _render_step(steps[i], inst, layout, graph_ph, info_ph, ctrl_ph,
                             algo_name, n_steps)
                time.sleep(speed)
            st.session_state["playing"] = False
            st.rerun()
        else:
            _render_step(steps[step_idx], inst, layout, graph_ph, info_ph, ctrl_ph,
                         algo_name, n_steps)

    # ------------------------------------------------------------------ #
    # Tab 2: Comparacion de algoritmos
    # ------------------------------------------------------------------ #
    with tab_compare:
        st.subheader("⚖️ Comparacion final — todos los algoritmos")

        skip_dp = inst.k > 13
        if skip_dp:
            st.warning(f"k={inst.k} > 13: DP omitida (muy lenta para este tamanho).")

        to_run = [a for a in ALGO_META if not (a.startswith("Dreyfus") and skip_dp)]
        chosen = st.multiselect("Algoritmos a comparar", to_run, default=to_run)

        if st.button("▶ Calcular comparacion", type="primary"):
            layout_cmp = compute_plotly_layout(inst.graph, inst)
            results: dict = {}
            opt_cost: float | None = None
            progress = st.progress(0)
            for i, name in enumerate(chosen):
                with st.spinner(f"Calculando {name}..."):
                    t0 = time.perf_counter()
                    cost, tree = ALGO_META[name]["fn"](inst)
                    elapsed = time.perf_counter() - t0
                    results[name] = (cost, tree, ALGO_META[name]["color"], elapsed)
                    if "Dreyfus" in name:
                        opt_cost = cost
                progress.progress((i + 1) / len(chosen))
            progress.empty()

            # Tabla
            rows = []
            for name, (cost, _, _, elapsed) in results.items():
                ratio = f"{cost/opt_cost:.4f}" if opt_cost else "—"
                rows.append({"Algoritmo": name, "Costo": f"{cost:.4f}",
                             "Tiempo (ms)": f"{elapsed*1000:.1f}", "Ratio vs DP": ratio})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Figura Plotly con pesos visibles
            plot_res = {n: (c, t, col) for n, (c, t, col, _) in results.items()}
            fig = make_comparison_figure(inst, plot_res, layout_cmp, opt_cost)
            st.plotly_chart(
                fig, use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
            )
            st.caption(
                "🟨 Cuadrados = terminales · Pasa el cursor sobre aristas para ver pesos · "
                "Rueda del raton para zoom · Arrastra para mover"
            )

    # ------------------------------------------------------------------ #
    # Tab 3: Datos del experimento
    # ------------------------------------------------------------------ #
    with tab_bench:
        _bench_panel()


if __name__ == "__main__":
    main()
