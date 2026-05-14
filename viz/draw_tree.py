"""
Dibujo estático de instancias y árboles de Steiner.

Funciones puras (sin estado global) sobre matplotlib + networkx:

- :func:`compute_layout`        — layout reutilizable.
- :func:`draw_instance`         — pinta vértices, terminales y aristas.
- :func:`overlay_tree`          — superpone un árbol resaltado.
- :func:`compare_side_by_side`  — figura con dos paneles (DP vs greedy).
- :func:`compare_grid`          — figura con cuatro paneles (DP, KMB, Mehlhorn, RSPH).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

# Backend headless por defecto: ahorra problemas al correr sin pantalla.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import networkx as nx

from steiner import Instance


def compute_layout(G: nx.Graph, seed: int = 0) -> dict:
    """Calcula una disposición consistente para los nodos de ``G``.

    Prioridades:
      1. Si los nodos tienen atributo ``pos`` (caso Euclidean/Geometric),
         se reusa directamente.
      2. Si ``|V| <= 30``, se usa Kamada–Kawai (mejor estructura).
      3. En otro caso, spring layout con la semilla dada.
    """
    if all("pos" in d for _, d in G.nodes(data=True)):
        return {v: G.nodes[v]["pos"] for v in G.nodes}
    if G.number_of_nodes() <= 30:
        return nx.kamada_kawai_layout(G, weight="weight")
    return nx.spring_layout(G, seed=seed, weight="weight")


def draw_instance(
    ax,
    instance: Instance,
    layout: Mapping,
    *,
    show_weights: bool = True,
    node_size: int = 350,
    edge_color: str = "#bbbbbb",
    edge_width: float = 1.0,
) -> None:
    """Dibuja el grafo y resalta los terminales (cuadrados) y Steiner (círculos)."""
    G = instance.graph
    terms = list(instance.terminals)
    steiners = [v for v in G.nodes if v not in instance.terminals]

    nx.draw_networkx_edges(
        G, layout, ax=ax, edge_color=edge_color, width=edge_width, alpha=0.7
    )
    nx.draw_networkx_nodes(
        G, layout, nodelist=steiners, ax=ax,
        node_color="#cccccc", node_shape="o", node_size=node_size,
        edgecolors="black", linewidths=0.7,
    )
    nx.draw_networkx_nodes(
        G, layout, nodelist=terms, ax=ax,
        node_color="#ffcc66", node_shape="s", node_size=node_size,
        edgecolors="black", linewidths=1.0,
    )
    nx.draw_networkx_labels(G, layout, ax=ax, font_size=8)
    if show_weights and G.number_of_edges() <= 40:
        labels = {(u, v): f"{d['weight']:.2g}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, layout, edge_labels=labels, ax=ax, font_size=7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
        )


def overlay_tree(
    ax,
    tree: nx.Graph,
    layout: Mapping,
    *,
    color: str,
    label: str | None = None,
    width: float = 3.0,
) -> None:
    """Superpone las aristas de ``tree`` con color y grosor distintivos."""
    if tree.number_of_edges() == 0:
        return
    edges = list(tree.edges())
    nx.draw_networkx_edges(
        tree, layout, edgelist=edges, ax=ax,
        edge_color=color, width=width, alpha=0.85, label=label,
    )


def compare_side_by_side(
    instance: Instance,
    trees: Mapping[str, tuple[nx.Graph, str]],
    out_path: str | Path,
    *,
    title: str | None = None,
    show_weights: bool = True,
) -> Path:
    """Figura con un panel por algoritmo: ``trees`` es ``{nombre: (tree, color)}``.

    Returns the saved PNG path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = compute_layout(instance.graph)
    n_panels = len(trees)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (name, (tree, color)) in zip(axes, trees.items()):
        draw_instance(ax, instance, layout, show_weights=show_weights)
        overlay_tree(ax, tree, layout, color=color, label=name)
        cost = sum(d.get("weight", 0.0) for _, _, d in tree.edges(data=True))
        ax.set_title(f"{name}\ncost = {cost:.4f}")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compare_grid(
    instance: Instance,
    trees: Mapping[str, tuple[nx.Graph, str]],
    out_path: str | Path,
    *,
    title: str | None = None,
    cols: int = 2,
) -> Path:
    """Variante con grilla de paneles (2 columnas por defecto)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = compute_layout(instance.graph)
    items = list(trees.items())
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, (name, (tree, color)) in zip(axes, items):
        draw_instance(ax, instance, layout, show_weights=False)
        overlay_tree(ax, tree, layout, color=color, label=name)
        cost = sum(d.get("weight", 0.0) for _, _, d in tree.edges(data=True))
        ax.set_title(f"{name}\ncost = {cost:.4f}")
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
