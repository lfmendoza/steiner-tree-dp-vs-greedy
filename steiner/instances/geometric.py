"""
Generador de instancias por Random Geometric Graph.

``n`` puntos uniformes en ``[0, 1]^2``; existe arista entre puntos a
distancia euclidiana <= ``r``. Familia intermedia entre dispersa y
densa: aparece naturalmente en redes ad-hoc y de sensores.

Referencia
----------
Penrose, M. (2003). *Random Geometric Graphs*. Oxford Studies in
Probability.
"""
from __future__ import annotations

import math
import random

import networkx as nx

from ..graph_utils import Instance


def geometric(
    n: int,
    r: float,
    k: int,
    seed: int = 0,
    max_attempts: int = 60,
) -> Instance:
    """Random Geometric Graph con ``n`` nodos, radio ``r``, ``k`` terminales.

    Si para algún ``r`` el grafo no resulta conexo, se incrementa ``r``
    multiplicativamente y se reintenta hasta ``max_attempts`` veces.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1.")
    if r <= 0:
        raise ValueError("r debe ser positivo.")
    if not 1 <= k <= n:
        raise ValueError("k debe satisfacer 1 <= k <= n.")

    rng = random.Random(seed)
    pts = [(rng.random(), rng.random()) for _ in range(n)]

    G: nx.Graph | None = None
    r_eff = r
    for attempt in range(max_attempts):
        candidate = nx.Graph()
        for i, p in enumerate(pts):
            candidate.add_node(i, pos=p)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if d <= r_eff:
                    candidate.add_edge(i, j, weight=d)
        if nx.is_connected(candidate):
            G = candidate
            break
        r_eff *= 1.2

    if G is None:
        # Como último recurso, completar el grafo (deja de ser geométrico).
        G = nx.complete_graph(n)
        for i, p in enumerate(pts):
            G.nodes[i]["pos"] = p
        for i, j in G.edges():
            G[i][j]["weight"] = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    terminals = frozenset(rng.sample(list(G.nodes), k))
    return Instance(graph=G, terminals=terminals)
