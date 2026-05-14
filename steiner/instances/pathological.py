"""
Instancias patológicas (tight) para Steiner Tree.

Familias diseñadas para forzar comportamientos extremos en heurísticas
con cota 2-aproximación: la idea es exhibir instancias donde el cociente
empírico greedy/óptimo se acerque arbitrariamente a la cota teórica.

Construcción :func:`spider`
---------------------------
Sea ``k`` el número de terminales y ``epsilon ∈ (0, 1)``.

  Vértices:   t_1, ..., t_k  y un único punto de Steiner ``s``.
  Aristas:    (t_i, s)        con peso 1                para i = 1..k
              (t_i, t_{i+1})  con peso 2 − epsilon       para i = 1..k-1

  Óptimo:     la estrella centrada en ``s`` con costo total = k.
  KMB:        elige las aristas perimetrales y produce la cadena
              t_1 - t_2 - ... - t_k de costo (k-1)(2 − epsilon).
  Cociente:   (k-1)(2 − epsilon) / k → 2(1 − 1/k) cuando epsilon → 0.

Esto muestra que la cota 2(1 − 1/L) de KMB es esencialmente apretada.
"""
from __future__ import annotations

import networkx as nx

from ..graph_utils import Instance


def spider(k: int, epsilon: float = 0.05) -> Instance:
    """Construcción tight para KMB.

    Parameters
    ----------
    k : int
        Número de terminales, ``k >= 2``.
    epsilon : float
        Pequeña perturbación; ``0 < epsilon < 1``. Cuanto menor sea,
        más se acerca el cociente empírico a la cota teórica.
    """
    if k < 2:
        raise ValueError("k debe ser >= 2.")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon debe estar en (0, 1).")

    G = nx.Graph()
    terms = [f"t{i+1}" for i in range(k)]
    G.add_node("s")
    for t in terms:
        G.add_edge(t, "s", weight=1.0)
    perim_w = 2.0 - epsilon
    for i in range(k - 1):
        G.add_edge(terms[i], terms[i + 1], weight=perim_w)

    return Instance(graph=G, terminals=frozenset(terms))


def double_spider(k1: int, k2: int, epsilon: float = 0.05, bridge: float = 1.0) -> Instance:
    """Dos arañas unidas por un puente entre sus centros de Steiner.

    Útil para estresar heurísticas que toman decisiones locales: la
    estructura óptima atraviesa el puente, pero KMB puede preferir
    aristas perimetrales en cada lado.
    """
    if k1 < 2 or k2 < 2:
        raise ValueError("k1 y k2 deben ser >= 2.")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon debe estar en (0, 1).")
    if bridge <= 0:
        raise ValueError("bridge debe ser positivo.")

    G = nx.Graph()
    left = [f"l{i+1}" for i in range(k1)]
    right = [f"r{i+1}" for i in range(k2)]
    G.add_node("sL")
    G.add_node("sR")
    perim_w = 2.0 - epsilon

    for t in left:
        G.add_edge(t, "sL", weight=1.0)
    for i in range(k1 - 1):
        G.add_edge(left[i], left[i + 1], weight=perim_w)

    for t in right:
        G.add_edge(t, "sR", weight=1.0)
    for i in range(k2 - 1):
        G.add_edge(right[i], right[i + 1], weight=perim_w)

    G.add_edge("sL", "sR", weight=bridge)

    terminals = frozenset(left + right)
    return Instance(graph=G, terminals=terminals)


def grid_with_shortcut(n: int, shortcut_weight: float = 0.5) -> Instance:
    """Cuadrícula ``n x n`` con un atajo "barato" en diagonal.

    Vértices etiquetados ``(i, j)``. Aristas horizontales y verticales
    con peso 1. Se añade la diagonal ``(0, 0) - (n-1, n-1)`` con peso
    ``shortcut_weight``. Terminales: las cuatro esquinas.

    Sirve para ilustrar cómo una sola arista "trampa" engaña a heurísticas
    que toman decisiones locales sobre caminos más cortos.
    """
    if n < 2:
        raise ValueError("n debe ser >= 2.")
    if shortcut_weight < 0:
        raise ValueError("shortcut_weight no puede ser negativo.")

    G = nx.grid_2d_graph(n, n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    G.add_edge((0, 0), (n - 1, n - 1), weight=shortcut_weight)

    corners = frozenset({(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)})
    return Instance(graph=G, terminals=corners)
