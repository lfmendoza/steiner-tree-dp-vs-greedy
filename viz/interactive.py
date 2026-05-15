"""
Visualizacion interactiva de instancias y algoritmos de Steiner Tree.

Dos motores disponibles:
  - :func:`make_pyvis_html`      — PyVis (vis.js): draggable, physics, hover HTML,
                                   zoom/pan fluido. Motor principal del dashboard.
  - :func:`make_network_figure`  — Plotly: para figuras estaticas del paper/bench.
  - :func:`make_comparison_figure` — Subplots Plotly para comparaciones.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from steiner import Instance

# ---------------------------------------------------------------------------
# Paleta de colores
# ---------------------------------------------------------------------------
C = {
    "terminal":         "#FFB300",
    "terminal_border":  "#795500",
    "tree_node":        "#42A5F5",
    "tree_border":      "#0D47A1",
    "inserted":         "#EF5350",
    "inserted_border":  "#B71C1C",
    "cand_hot":         "#FF6D00",
    "cand_cold":        "#FFCC80",
    "inactive":         "#78909C",
    "inactive_border":  "#455A64",
    "tree_edge":        "#42A5F5",
    "new_edge":         "#66BB6A",
    "bg_edge":          "rgba(160,160,160,0.20)",
    "bg_dark":          "#0F1117",
    "bg_graph":         "#1E1E2E",
    "node_label":       "#FFFFFF",
    "edge_label":       "#AAAAAA",
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def compute_plotly_layout(G: nx.Graph, instance: Instance, seed: int = 42) -> dict:
    """Posiciones de nodos para la visualizacion."""
    if all("pos" in G.nodes[v] for v in G.nodes):
        return {v: list(G.nodes[v]["pos"]) for v in G.nodes}
    if G.number_of_nodes() <= 35:
        try:
            return {
                v: list(p)
                for v, p in nx.kamada_kawai_layout(G, weight="weight").items()
            }
        except Exception:
            pass
    return {
        v: list(p)
        for v, p in nx.spring_layout(G, seed=seed, weight="weight").items()
    }


# ---------------------------------------------------------------------------
# Motor PyVis (vis.js) — dashboard principal
# ---------------------------------------------------------------------------


def make_pyvis_html(
    G: nx.Graph,
    layout: dict,
    instance: Instance,
    *,
    tree: nx.Graph | None = None,
    new_edges: list[tuple] | None = None,
    candidate_savings: dict | None = None,
    inserted_vertex: Any = None,
    height: int = 560,
    dark: bool = True,
) -> str:
    """Genera HTML interactivo con vis.js (via pyvis).

    Caracteristicas:
    - Zoom / pan fluido con rueda y arrastre de fondo.
    - Nodos arrastrables individualmente.
    - Hover HTML por nodo (tipo, ahorro, vecinos) y arista (peso).
    - Botones de navegacion integrados (zoom in/out/fit).
    - Posiciones fijas (precomputadas) para coherencia entre pasos.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color:red'>pyvis no instalado. Ejecuta: pip install pyvis</p>"

    bg = C["bg_graph"] if dark else "#F5F7FA"
    font_color = "#FFFFFF" if dark else "#1A1A2A"

    net = Network(height=f"{height}px", width="100%", bgcolor=bg, font_color=font_color)

    opts = {
        "physics": {"enabled": False},
        "nodes": {
            "borderWidth": 2,
            "font": {"size": 13, "vadjust": -3},
        },
        "edges": {
            "smooth": False,
            "font": {"size": 9, "color": C["edge_label"], "align": "middle"},
            "scaling": {"min": 1, "max": 6},
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "zoomView": True,
            "dragView": True,
            "multiselect": False,
            "tooltipDelay": 80,
        },
        "configure": {"enabled": False},
    }
    net.set_options(json.dumps(opts))

    tree_nodes: set = set(tree.nodes) if tree else set()
    new_edge_set = {frozenset(e) for e in (new_edges or [])}
    tree_edges_set = {frozenset((u, v)) for u, v in (tree.edges() if tree else [])}
    max_sav = max(candidate_savings.values()) if candidate_savings else 1.0
    if max_sav < 1e-9:
        max_sav = 1.0

    # Escalar layout a pixeles (vis.js usa px)
    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0
    W, H = 600, 460

    def px(v):
        x, y = layout[v]
        return (
            int((x - min_x) / span_x * W - W / 2),
            int(-(y - min_y) / span_y * H + H / 2),
        )

    # --- Nodos ---
    for v in G.nodes():
        vx, vy = px(v)
        is_t = v in instance.terminals
        in_tree = v in tree_nodes
        is_ins = v == inserted_vertex
        sav = (candidate_savings or {}).get(v, 0.0)
        is_cand = (not is_t) and (not in_tree) and (not is_ins) and (sav > 1e-9)

        if is_ins:
            color = {"background": C["inserted"], "border": C["inserted_border"],
                     "highlight": {"background": "#FF8A80", "border": C["inserted_border"]}}
            shape = "star"
            size = 32
        elif is_t:
            color = {"background": C["terminal"], "border": C["terminal_border"],
                     "highlight": {"background": "#FFE082", "border": C["terminal_border"]}}
            shape = "square"
            size = 26
        elif in_tree:
            color = {"background": C["tree_node"], "border": C["tree_border"],
                     "highlight": {"background": "#90CAF9", "border": C["tree_border"]}}
            shape = "dot"
            size = 20
        elif is_cand:
            norm = sav / max_sav
            r = int(255 * norm + 200 * (1 - norm))
            g_val = int(100 * (1 - norm) + 140 * norm)
            b_val = 30
            hex_c = f"#{r:02X}{g_val:02X}{b_val:02X}"
            color = {"background": hex_c, "border": "#BF360C",
                     "highlight": {"background": "#FFAB40", "border": "#BF360C"}}
            shape = "dot"
            size = int(12 + 14 * norm)
        else:
            color = {"background": C["inactive"], "border": C["inactive_border"],
                     "highlight": {"background": "#B0BEC5", "border": C["inactive_border"]}}
            shape = "dot"
            size = 10

        nbrs = [str(n) for n in G.neighbors(v)]
        if len(nbrs) > 6:
            nbrs = nbrs[:6] + [f"+{len(nbrs)-6}"]

        tipo = "Terminal" if is_t else ("Steiner (arbol)" if in_tree else ("Insertado este paso" if is_ins else "No activo"))
        parts = [str(v), tipo]
        if is_ins:
            parts.append("Insertado en este paso")
        if is_cand:
            parts.append(f"Ahorro potencial: {sav:.4f}")
        if in_tree and tree:
            parts.append(f"Grado en arbol: {tree.degree(v)}")
        parts.append(f"Vecinos: {', '.join(nbrs)}")
        tip = "\n".join(parts)

        net.add_node(str(v), label=str(v), x=vx, y=vy,
                     color=color, shape=shape, size=size,
                     title=tip, fixed=True)

    # --- Aristas ---
    for u, v, data in G.edges(data=True):
        key = frozenset((u, v))
        is_new = key in new_edge_set
        in_tree_edge = key in tree_edges_set
        w = data.get("weight", 0.0)
        parts_e = [f"({u}) - ({v})", f"Peso: {w:.4f}"]
        if is_new:
            parts_e.append("Arista añadida en este paso")
        elif in_tree_edge:
            parts_e.append("En el arbol actual")
        tip_e = "\n".join(parts_e)

        if is_new:
            e_color = C["new_edge"]
            width = 5
        elif in_tree_edge:
            e_color = C["tree_edge"]
            width = 4
        else:
            e_color = "rgba(120,120,150,0.25)"
            width = 1

        net.add_edge(
            str(u), str(v),
            title=tip_e,
            label=f"{w:.2g}",
            color=e_color,
            width=width,
        )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    net.save_graph(tmp.name)
    tmp.close()
    html = open(tmp.name, encoding="utf-8").read()
    os.unlink(tmp.name)

    # Parche: scroll interno del iframe sin capturar el del navegador
    html = html.replace(
        "<body>",
        "<body style='margin:0;overflow:hidden;background:" + bg + ";'>",
    )
    return html


# ---------------------------------------------------------------------------
# Motor Plotly — paper y comparacion
# ---------------------------------------------------------------------------


def make_network_figure(
    G: nx.Graph,
    layout: dict,
    instance: Instance,
    *,
    tree: nx.Graph | None = None,
    new_edges: list[tuple] | None = None,
    candidate_savings: dict | None = None,
    inserted_vertex: Any = None,
    step_description: str = "",
    title: str = "",
    show_all_weights: bool = True,
    height: int = 520,
) -> go.Figure:
    """Figura Plotly para comparacion estatica (paper / bench)."""
    terminals = instance.terminals
    tree_nodes = set(tree.nodes) if tree else set()
    tree_edges_set = {frozenset((u, v)) for u, v in (tree.edges() if tree else [])}
    new_edge_set = {frozenset(e) for e in (new_edges or [])}
    max_sav = max(candidate_savings.values()) if candidate_savings else 1.0
    if max_sav < 1e-9:
        max_sav = 1.0
    traces = []

    # Fondo
    bx, by = [], []
    for u, v in G.edges():
        x0, y0 = layout[u]; x1, y1 = layout[v]
        bx += [x0, x1, None]; by += [y0, y1, None]
    traces.append(go.Scatter(x=bx, y=by, mode="lines",
                             line=dict(color="rgba(200,200,200,0.35)", width=1),
                             hoverinfo="none", showlegend=False))

    # Midpoints hover
    mx, my, mh = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = layout[u]; x1, y1 = layout[v]
        mx.append((x0+x1)/2); my.append((y0+y1)/2)
        mh.append(f"({u})—({v}): {d.get('weight',0):.4f}")
    traces.append(go.Scatter(x=mx, y=my, mode="markers",
                             marker=dict(size=8, opacity=0),
                             customdata=mh,
                             hovertemplate="%{customdata}<extra></extra>",
                             showlegend=False))

    # Aristas del arbol
    if tree and tree.number_of_edges():
        tx, ty = [], []
        for u, v in tree.edges():
            x0, y0 = layout.get(u, (0,0)); x1, y1 = layout.get(v, (0,0))
            tx += [x0, x1, None]; ty += [y0, y1, None]
        traces.append(go.Scatter(x=tx, y=ty, mode="lines",
                                 line=dict(color=C["tree_edge"], width=4),
                                 hoverinfo="none", showlegend=True, name="Árbol"))

    # Aristas nuevas
    if new_edges:
        nx_x, nx_y = [], []
        for u, v in new_edges:
            x0, y0 = layout.get(u,(0,0)); x1, y1 = layout.get(v,(0,0))
            nx_x += [x0, x1, None]; nx_y += [y0, y1, None]
        traces.append(go.Scatter(x=nx_x, y=nx_y, mode="lines",
                                 line=dict(color=C["new_edge"], width=5, dash="solid"),
                                 hoverinfo="none", showlegend=True, name="Nuevas aristas"))

    # Pesos
    if show_all_weights and G.number_of_edges() <= 50:
        lx, ly, lt = [], [], []
        for u, v, d in G.edges(data=True):
            x0, y0 = layout[u]; x1, y1 = layout[v]
            lx.append((x0+x1)/2); ly.append((y0+y1)/2)
            lt.append(f"{d.get('weight',0):.2g}")
        traces.append(go.Scatter(x=lx, y=ly, mode="text", text=lt,
                                 textfont=dict(size=9, color="#222"),
                                 hoverinfo="none", showlegend=False))

    # Nodos
    for v in G.nodes():
        x, y = layout[v]
        is_t = v in terminals; in_t = v in tree_nodes
        is_ins = v == inserted_vertex
        sav = (candidate_savings or {}).get(v, 0.0)
        is_cand = (not is_t) and (not in_t) and (not is_ins) and (sav > 1e-9)

        if is_ins: color, sym, sz = C["inserted"], "star", 22
        elif is_t: color, sym, sz = C["terminal"], "square", 18
        elif in_t: color, sym, sz = C["tree_node"], "circle", 14
        elif is_cand:
            norm = sav / max_sav
            r = int(255*norm+200*(1-norm)); g_v = int(109*norm+200*(1-norm))
            color = f"rgb({r},{g_v},0)"
            sym, sz = "circle", int(10+10*norm)
        else: color, sym, sz = C["inactive"], "circle", 9

        nbrs = [str(n) for n in G.neighbors(v)]
        hover = (f"<b>{v}</b><br>"
                 f"{'Terminal' if is_t else 'Steiner'}<br>"
                 + (f"Ahorro: {sav:.4f}<br>" if is_cand else "")
                 + (f"Grado árbol: {tree.degree(v)}<br>" if in_t and tree else "")
                 + f"Vecinos: {', '.join(nbrs[:6])}")
        traces.append(go.Scatter(x=[x], y=[y], mode="markers+text",
                                 marker=dict(size=sz, color=color, symbol=sym,
                                             line=dict(color="black", width=1.5)),
                                 text=[str(v)], textposition="top center",
                                 textfont=dict(size=10),
                                 hovertemplate=f"{hover}<extra></extra>",
                                 showlegend=False))

    fig = go.Figure(data=traces)
    ann = []
    if step_description:
        ann.append(dict(text=step_description, xref="paper", yref="paper",
                        x=0, y=-0.08, showarrow=False, font=dict(size=10),
                        align="left", bgcolor="rgba(240,240,240,0.9)"))
    fig.update_layout(
        title=title or None,
        height=height, showlegend=True, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white", plot_bgcolor="#F5F7FA",
        margin=dict(l=10, r=10, t=40 if title else 10, b=80 if step_description else 20),
        annotations=ann,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Comparacion estatica Plotly (subplots 2x2)
# ---------------------------------------------------------------------------


def make_comparison_figure(
    instance: Instance,
    results: dict[str, tuple[float, nx.Graph, str]],
    layout: dict,
    opt_cost: float | None = None,
) -> go.Figure:
    items = list(results.items())
    n = len(items)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    titles = [
        f"{name}<br>costo={cost:.4f}"
        + (f" | ratio={cost/opt_cost:.4f}" if (opt_cost and opt_cost > 0) else "")
        for name, (cost, _, _) in items
    ]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.05, vertical_spacing=0.12)

    for idx, (name, (cost, tree, color)) in enumerate(items):
        row, col = idx // cols + 1, idx % cols + 1
        bx, by = [], []
        for u, v in instance.graph.edges():
            x0, y0 = layout[u]; x1, y1 = layout[v]
            bx += [x0, x1, None]; by += [y0, y1, None]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                                 line=dict(color="rgba(200,200,200,0.4)", width=1),
                                 hoverinfo="none", showlegend=False), row=row, col=col)
        if tree and tree.number_of_edges():
            tx, ty = [], []
            for u, v in tree.edges():
                x0, y0 = layout[u]; x1, y1 = layout[v]
                tx += [x0, x1, None]; ty += [y0, y1, None]
            fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
                                     line=dict(color=color, width=4),
                                     hoverinfo="none", showlegend=False), row=row, col=col)
        for u, v, d in instance.graph.edges(data=True):
            x0, y0 = layout[u]; x1, y1 = layout[v]
            fig.add_trace(go.Scatter(x=[(x0+x1)/2], y=[(y0+y1)/2],
                                     mode="text", text=[f"{d.get('weight',0):.2g}"],
                                     textfont=dict(size=9, color="#CCCCCC"),
                                     hoverinfo="none", showlegend=False), row=row, col=col)
        for v in instance.graph.nodes():
            x, y = layout[v]
            is_t = v in instance.terminals
            in_t = tree and v in tree.nodes()
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text",
                                     marker=dict(size=16 if is_t else 10,
                                                 color=C["terminal"] if is_t else (color if in_t else "#556070"),
                                                 symbol="square" if is_t else "circle",
                                                 line=dict(color="#CCCCCC", width=1)),
                                     text=[str(v)], textposition="top center",
                                     textfont=dict(size=10, color="#FFFFFF"),
                                     hovertemplate=f"<b>{v}</b><br>{'Terminal' if is_t else 'Steiner'}<extra></extra>",
                                     showlegend=False), row=row, col=col)

    for i in range(1, rows * cols + 1):
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                         row=(i-1)//cols+1, col=(i-1)%cols+1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                         row=(i-1)//cols+1, col=(i-1)%cols+1)
    fig.update_layout(
        height=420 * rows,
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#2A2A3E",
        margin=dict(l=10, r=10, t=70, b=20),
        font=dict(color="#ECECEC"),
    )
    # Subplot titles son anotaciones — forzar color blanco y tamaño legible
    fig.update_annotations(font=dict(color="#FFFFFF", size=13))
    return fig
