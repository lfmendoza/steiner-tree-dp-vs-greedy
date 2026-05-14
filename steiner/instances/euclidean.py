"""
Generador de instancias euclidianas.

Toma ``n`` puntos uniformes en ``[0, 1]^2``, construye el grafo
completo y asigna a cada arista la distancia euclidiana entre sus
extremos. Es la familia más cercana a aplicaciones reales (ruteo,
VLSI, telecomunicaciones).
"""
from __future__ import annotations

import math
import random

import networkx as nx

from ..graph_utils import Instance


def euclidean(n: int, k: int, seed: int = 0) -> Instance:
    """``n`` puntos uniformes en ``[0, 1]^2``; grafo completo con
    pesos = distancia euclidiana; ``k`` terminales aleatorios.

    Las posiciones se guardan en cada nodo bajo el atributo ``pos`` para
    poder reutilizarlas como layout en :func:`viz.draw_tree.compute_layout`.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1.")
    if not 1 <= k <= n:
        raise ValueError("k debe satisfacer 1 <= k <= n.")

    rng = random.Random(seed)
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    G = nx.complete_graph(n)
    for v, p in enumerate(pts):
        G.nodes[v]["pos"] = p
    for u, v in G.edges():
        dx = pts[u][0] - pts[v][0]
        dy = pts[u][1] - pts[v][1]
        G[u][v]["weight"] = math.hypot(dx, dy)

    terminals = frozenset(rng.sample(list(G.nodes), k))
    return Instance(graph=G, terminals=terminals)
