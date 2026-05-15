"""
Visualizacion interactiva de instancias y algoritmos de Steiner Tree.

Usa Plotly Graph Objects para producir figuras completamente interactivas:
zoom, pan, hover con pesos de aristas, resaltado de nodos y aristas,
y diferenciacion visual entre terminales, puntos de Steiner y candidatos.

API publica
-----------
:func:`compute_plotly_layout`
    Calcula las posiciones de los nodos.
:func:`make_network_figure`
    Figura Plotly con el estado completo de un paso de algoritmo.
:func:`make_comparison_figure`
    Subplots 2x2 comparando los cuatro algoritmos.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from steiner import Instance


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def compute_plotly_layout(G: nx.Graph, instance: Instance, seed: int = 42) -> dict:
    """Calcula posiciones de nodos para la visualizacion.

    Prioridades:
    1. Atributo ``pos`` en los nodos (Euclidean/Geometric).
    2. Kamada-Kawai si n <= 35.
    3. Spring layout con semilla fija.
    """
    if all("pos" in G.nodes[v] for v in G.nodes):
        return {v: list(G.nodes[v]["pos"]) for v in G.nodes}
    if G.number_of_nodes() <= 35:
        try:
            return {v: list(pos) for v, pos in nx.kamada_kawai_layout(G, weight="weight").items()}
        except Exception:
            pass
    return {v: list(pos) for v, pos in nx.spring_layout(G, seed=seed, weight="weight").items()}


# ---------------------------------------------------------------------------
# Figura principal
# ---------------------------------------------------------------------------


def make_network_figure(
    G: nx.Graph,
    layout: dict,
    instance: Instance,
    *,
    tree: nx.Graph | None = None,
    highlight_edges: list[tuple] | None = None,
    candidate_savings: dict | None = None,
    inserted_vertex: Any = None,
    step_description: str = "",
    title: str = "",
    show_all_weights: bool = True,
    height: int = 520,
) -> go.Figure:
    """Produce una figura Plotly completamente interactiva.

    Caracteristicas:
    - Zoom / pan automaticos.
    - Hover sobre aristas: muestra peso y nodos conectados.
    - Hover sobre nodos: muestra nombre, tipo (terminal/Steiner),
      grado en el arbol, ahorro potencial (GSVI) y vecinos.
    - Terminales = cuadrados dorados.
    - Nodos en arbol = circulos azules.
    - Candidatos GSVI = circulos naranjas (tamanho proporcional al ahorro).
    - Mejor candidato = estrella roja.
    - Aristas del arbol = linea azul gruesa.
    - Aristas a resaltar = linea naranja.
    - Fondo del grafo = lineas grises delgadas.

    Parameters
    ----------
    G : nx.Graph
        Grafo completo de la instancia.
    layout : dict
        Posiciones {nodo: [x, y]}.
    instance : Instance
        Instancia (para saber cuales son terminales).
    tree : nx.Graph, optional
        Arbol de Steiner actual; sus aristas se muestran en azul grueso.
    highlight_edges : list[tuple], optional
        Aristas extra a resaltar en naranja.
    candidate_savings : dict, optional
        {vertice: ahorro} para visualizar candidatos GSVI.
    inserted_vertex : any, optional
        Vertice recien insertado (estrella roja).
    step_description : str
        Texto informativo mostrado como anotacion en la figura.
    title : str
        Titulo de la figura.
    show_all_weights : bool
        Si True, muestra etiquetas de peso en todas las aristas del grafo.
    height : int
        Alto de la figura en pixeles.
    """
    terminals = instance.terminals
    tree_nodes = set(tree.nodes) if tree is not None else set()
    tree_edges_set = (
        {frozenset((u, v)) for u, v in tree.edges()} if tree is not None else set()
    )
    highlight_set = {frozenset(e) for e in (highlight_edges or [])}

    traces: list[go.BaseTraceType] = []

    # ------------------------------------------------------------------ #
    # 1. Aristas del grafo completo (fondo, gris muy claro)
    # ------------------------------------------------------------------ #
    bg_x: list[float | None] = []
    bg_y: list[float | None] = []
    for u, v in G.edges():
        x0, y0 = layout[u]
        x1, y1 = layout[v]
        bg_x += [x0, x1, None]
        bg_y += [y0, y1, None]

    traces.append(
        go.Scatter(
            x=bg_x, y=bg_y,
            mode="lines",
            line=dict(color="rgba(200,200,200,0.5)", width=1),
            hoverinfo="none",
            showlegend=False,
            name="",
        )
    )

    # ------------------------------------------------------------------ #
    # 2. Puntos invisibles en el centro de cada arista para hover de peso
    # ------------------------------------------------------------------ #
    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_hover: list[str] = []

    for u, v, data in G.edges(data=True):
        x0, y0 = layout[u]
        x1, y1 = layout[v]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        w = data.get("weight", 0.0)
        in_tree_marker = " [arbol]" if frozenset((u, v)) in tree_edges_set else ""
        mid_hover.append(
            f"<b>({u}) — ({v})</b><br>"
            f"Peso: <b>{w:.4f}</b>{in_tree_marker}"
        )

    traces.append(
        go.Scatter(
            x=mid_x, y=mid_y,
            mode="markers",
            marker=dict(size=8, opacity=0, color="white"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=mid_hover,
            showlegend=False,
            name="",
        )
    )

    # ------------------------------------------------------------------ #
    # 3. Aristas del arbol (azul grueso)
    # ------------------------------------------------------------------ #
    if tree is not None and tree.number_of_edges() > 0:
        tree_x: list[float | None] = []
        tree_y: list[float | None] = []
        for u, v, data in tree.edges(data=True):
            if u not in layout or v not in layout:
                continue
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            tree_x += [x0, x1, None]
            tree_y += [y0, y1, None]
        traces.append(
            go.Scatter(
                x=tree_x, y=tree_y,
                mode="lines",
                line=dict(color="rgba(31,119,180,0.85)", width=4),
                hoverinfo="none",
                showlegend=True,
                name="Arbol actual",
            )
        )

    # ------------------------------------------------------------------ #
    # 4. Aristas resaltadas (naranja)
    # ------------------------------------------------------------------ #
    if highlight_edges:
        hl_x: list[float | None] = []
        hl_y: list[float | None] = []
        for u, v in highlight_edges:
            if u not in layout or v not in layout:
                continue
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            hl_x += [x0, x1, None]
            hl_y += [y0, y1, None]
        if hl_x:
            traces.append(
                go.Scatter(
                    x=hl_x, y=hl_y,
                    mode="lines",
                    line=dict(color="rgba(255,127,14,0.9)", width=3, dash="dash"),
                    hoverinfo="none",
                    showlegend=True,
                    name="Aristas resaltadas",
                )
            )

    # ------------------------------------------------------------------ #
    # 5. Nodos
    # ------------------------------------------------------------------ #
    max_savings = max(candidate_savings.values()) if candidate_savings else 1.0
    if max_savings == 0:
        max_savings = 1.0

    for v in G.nodes():
        x, y = layout[v]
        is_terminal = v in terminals
        in_tree = v in tree_nodes
        is_inserted = v == inserted_vertex
        savings_val = (candidate_savings or {}).get(v, 0.0)
        is_candidate = (candidate_savings is not None) and (v not in instance.terminals) and (v not in tree_nodes) and (not is_inserted)

        # --- apariencia ---
        if is_inserted:
            color = "#d62728"
            symbol = "star"
            size = 22
            border_color = "#8b0000"
            legend_group = "inserted"
            name = "Insertado"
        elif is_terminal:
            color = "#f4c430"
            symbol = "square"
            size = 18
            border_color = "#8b6914"
            legend_group = "terminal"
            name = "Terminal"
        elif in_tree:
            color = "#1f77b4"
            symbol = "circle"
            size = 14
            border_color = "#0a3d70"
            legend_group = "steiner_tree"
            name = "Steiner (en arbol)"
        elif is_candidate and savings_val > 1e-9:
            norm = savings_val / max_savings
            r = int(255 * norm + 200 * (1 - norm))
            g_val = int(100 * (1 - norm) + 140 * norm)
            b_val = int(50 * (1 - norm))
            color = f"rgb({r},{g_val},{b_val})"
            symbol = "circle"
            size = int(10 + 10 * norm)
            border_color = "#7f3f00"
            legend_group = "candidate"
            name = "Candidato Steiner"
        else:
            color = "rgba(180,180,180,0.6)"
            symbol = "circle"
            size = 9
            border_color = "#888888"
            legend_group = "inactive"
            name = "No activo"

        # --- hover text ---
        neighbors = list(G.neighbors(v))
        nbr_str = ", ".join(str(n) for n in neighbors[:8])
        if len(neighbors) > 8:
            nbr_str += f" (+{len(neighbors)-8})"

        hover = (
            f"<b>{v}</b><br>"
            f"Tipo: {'Terminal' if is_terminal else ('Steiner (en arbol)' if in_tree else 'No terminal')}<br>"
        )
        if is_inserted:
            hover += f"<b>Insertado en este paso</b><br>Ahorro: {savings_val:.4f}<br>"
        elif is_candidate and savings_val > 1e-9:
            hover += f"Ahorro potencial: <b>{savings_val:.4f}</b><br>"
        if in_tree and tree is not None:
            hover += f"Grado en arbol: {tree.degree(v)}<br>"
        hover += f"Vecinos en G: {nbr_str}"

        traces.append(
            go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(color=border_color, width=1.5),
                ),
                text=[str(v)],
                textposition="top center",
                textfont=dict(size=11),
                hovertemplate=f"{hover}<extra></extra>",
                showlegend=False,
                name=name,
                legendgroup=legend_group,
            )
        )

    # ------------------------------------------------------------------ #
    # 6. Etiquetas de peso sobre todas las aristas (si se piden)
    # ------------------------------------------------------------------ #
    if show_all_weights and G.number_of_edges() <= 50:
        label_x: list[float] = []
        label_y: list[float] = []
        label_text: list[str] = []
        for u, v, data in G.edges(data=True):
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            w = data.get("weight", 0.0)
            label_text.append(f"{w:.2g}")
        traces.append(
            go.Scatter(
                x=label_x, y=label_y,
                mode="text",
                text=label_text,
                textfont=dict(size=9, color="#666666"),
                hoverinfo="none",
                showlegend=False,
                name="",
            )
        )

    # ------------------------------------------------------------------ #
    # 7. Layout de la figura
    # ------------------------------------------------------------------ #
    annotations = []
    if step_description:
        annotations.append(
            dict(
                text=step_description,
                xref="paper", yref="paper",
                x=0, y=-0.08,
                showarrow=False,
                font=dict(size=11),
                align="left",
                bgcolor="rgba(240,240,240,0.85)",
                bordercolor="#aaaaaa",
                borderwidth=1,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        showlegend=True,
        hovermode="closest",
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=80 if step_description else 20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10),
        ),
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Figura de comparacion 2x2
# ---------------------------------------------------------------------------


def make_comparison_figure(
    instance: Instance,
    results: dict[str, tuple[float, nx.Graph, str]],
    layout: dict,
    opt_cost: float | None = None,
) -> go.Figure:
    """Figura 2x2 comparando hasta 4 algoritmos.

    Parameters
    ----------
    instance : Instance
    results : dict
        {nombre: (cost, tree, color_hex)} — hasta 4 entradas.
    layout : dict
        Layout compartido para todas las subfiguras.
    opt_cost : float, optional
        Costo optimo (DP) para calcular cocientes.
    """
    items = list(results.items())
    n = len(items)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    titles = []
    for name, (cost, _, _) in items:
        ratio_str = (
            f" | ratio={cost/opt_cost:.4f}" if (opt_cost and opt_cost > 0) else ""
        )
        titles.append(f"{name}<br>costo={cost:.4f}{ratio_str}")

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
    )

    for idx, (name, (cost, tree, color)) in enumerate(items):
        row = idx // cols + 1
        col = idx % cols + 1

        # Aristas fondo
        bg_x: list[float | None] = []
        bg_y: list[float | None] = []
        for u, v in instance.graph.edges():
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            bg_x += [x0, x1, None]
            bg_y += [y0, y1, None]
        fig.add_trace(
            go.Scatter(x=bg_x, y=bg_y, mode="lines",
                       line=dict(color="rgba(200,200,200,0.5)", width=1),
                       hoverinfo="none", showlegend=False),
            row=row, col=col,
        )

        # Aristas del arbol
        if tree is not None and tree.number_of_edges() > 0:
            tx: list[float | None] = []
            ty: list[float | None] = []
            for u, v, data in tree.edges(data=True):
                if u in layout and v in layout:
                    x0, y0 = layout[u]
                    x1, y1 = layout[v]
                    tx += [x0, x1, None]
                    ty += [y0, y1, None]
            fig.add_trace(
                go.Scatter(x=tx, y=ty, mode="lines",
                           line=dict(color=color, width=4),
                           hoverinfo="none", showlegend=False),
                row=row, col=col,
            )

        # Pesos en aristas
        for u, v, data in instance.graph.edges(data=True):
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            fig.add_trace(
                go.Scatter(
                    x=[(x0+x1)/2], y=[(y0+y1)/2],
                    mode="text",
                    text=[f"{data.get('weight', 0.0):.2g}"],
                    textfont=dict(size=8, color="#888"),
                    hoverinfo="none", showlegend=False,
                ),
                row=row, col=col,
            )

        # Nodos
        for v in instance.graph.nodes():
            x, y = layout[v]
            is_t = v in instance.terminals
            in_tree = tree is not None and v in tree.nodes()
            fig.add_trace(
                go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=15 if is_t else 10,
                        color="#f4c430" if is_t else (color if in_tree else "#cccccc"),
                        symbol="square" if is_t else "circle",
                        line=dict(color="black", width=1),
                    ),
                    text=[str(v)],
                    textposition="top center",
                    textfont=dict(size=9),
                    hovertemplate=f"<b>{v}</b><br>{'Terminal' if is_t else 'Steiner'}<extra></extra>",
                    showlegend=False,
                ),
                row=row, col=col,
            )

    for i in range(1, rows * cols + 1):
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=(i-1)//cols+1, col=(i-1)%cols+1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=(i-1)//cols+1, col=(i-1)%cols+1)

    fig.update_layout(
        height=420 * rows,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig
