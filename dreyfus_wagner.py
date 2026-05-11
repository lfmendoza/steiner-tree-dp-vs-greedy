"""
Algoritmo exacto de Dreyfus-Wagner para Steiner Tree en grafos.

Referencia
----------
S. E. Dreyfus and R. A. Wagner. "The Steiner problem in graphs."
*Networks* 1 (1971/72), no. 3, 195-207.

Recurrencia
-----------
Para D ⊆ T con |D| ≥ 2 y v ∈ V definimos:

    f(D, v) = costo mínimo de un árbol de Steiner que contiene D ∪ {v}
    g(D, v) = costo mínimo restringido a árboles donde v actúa como
              punto de ramificación (degree ≥ 2)

Base:
    f({t}, v) = d(t, v),   donde d es distancia más corta en G

Paso 1 (merge):
    g(D, v) = min_{∅ ⊊ D' ⊊ D}  f(D', v) + f(D \\ D', v)

Paso 2 (extensión por camino):
    f(D, v) = min_{u ∈ V}  g(D, u) + d(u, v)

La solución óptima al problema original es f(T, t₀) para cualquier
terminal t₀ ∈ T fijo (porque T ∪ {t₀} = T).

Complejidad
-----------
Tiempo:  O(3^k · n + 2^k · n² + n³)
Espacio: O(2^k · n)
donde n = |V| y k = |T|.
"""
from __future__ import annotations

import math
from itertools import combinations

import networkx as nx

from .graph_utils import Instance, all_pairs_shortest_paths


def dreyfus_wagner(instance: Instance) -> tuple[float, nx.Graph]:
    """Resuelve Steiner Tree exactamente vía DP de Dreyfus-Wagner.

    Parameters
    ----------
    instance : Instance
        Instancia con grafo y terminales.

    Returns
    -------
    (cost, tree)
        ``cost`` es el peso total del árbol óptimo y ``tree`` es el subgrafo
        de ``instance.graph`` que materializa la solución.
    """
    G = instance.graph
    T = tuple(sorted(instance.terminals))
    k = len(T)

    # Caso trivial: un único terminal no requiere aristas.
    if k == 1:
        tree = nx.Graph()
        tree.add_node(T[0])
        return 0.0, tree

    dist, sp_path = all_pairs_shortest_paths(G)
    vertices = list(G.nodes)

    # Tablas de DP:
    #   f[D][v] = costo óptimo del árbol que cubre D ∪ {v}.
    # Para reconstrucción guardamos dos punteros de origen:
    #   parent_f[D][v] = u tal que f[D][v] = g[D][u] + dist[u][v]
    #   parent_g[D][u] = (D1, D2) tal que g[D][u] = f[D1][u] + f[D2][u]
    f: dict[frozenset, dict] = {}
    parent_f: dict[frozenset, dict] = {}
    parent_g: dict[frozenset, dict] = {}

    # Base: |D| = 1.
    for t in T:
        D = frozenset({t})
        f[D] = {v: dist[t][v] for v in vertices}
        # No requiere parent_f/parent_g: la reconstrucción detecta |D|=1.

    # Subconjuntos de tamaño 2, 3, ..., k.
    for size in range(2, k + 1):
        for D_tuple in combinations(T, size):
            D = frozenset(D_tuple)

            # --- Paso 1: merge ---
            # Anclamos el primer elemento de D para enumerar cada partición
            # no ordenada exactamente una vez: D' siempre contiene `anchor`.
            D_sorted = sorted(D)
            anchor = D_sorted[0]
            rest = tuple(D_sorted[1:])

            g_D = {v: math.inf for v in vertices}
            g_par: dict = {}

            # subconjuntos de `rest` de tamaño 0..|rest|-1 (esto fuerza
            # que D' = {anchor} ∪ sub sea subconjunto propio de D).
            for r in range(len(rest)):
                for sub in combinations(rest, r):
                    D1 = frozenset({anchor} | set(sub))
                    D2 = D - D1
                    f1 = f[D1]
                    f2 = f[D2]
                    for v in vertices:
                        val = f1[v] + f2[v]
                        if val < g_D[v]:
                            g_D[v] = val
                            g_par[v] = (D1, D2)

            parent_g[D] = g_par

            # --- Paso 2: extensión vía caminos más cortos ---
            # f[D][v] = min_u g[D][u] + d(u, v); incluye u = v con d=0.
            f_D = {}
            f_par = {}
            for v in vertices:
                best_val = math.inf
                best_u = v
                for u in vertices:
                    cand = g_D[u] + dist[u][v]
                    if cand < best_val:
                        best_val = cand
                        best_u = u
                f_D[v] = best_val
                f_par[v] = best_u

            f[D] = f_D
            parent_f[D] = f_par

    # Solución: f(T, t₀) para cualquier terminal t₀ ∈ T fijo.
    # Tomamos t₀ = T[0]; como t₀ ∈ T, T ∪ {t₀} = T.
    full = frozenset(T)
    t0 = T[0]
    cost = f[full][t0]

    tree = nx.Graph()
    _reconstruct(full, t0, parent_f, parent_g, sp_path, G, tree)
    return cost, tree


def _reconstruct(
    D: frozenset,
    v,
    parent_f: dict,
    parent_g: dict,
    sp_path: dict,
    G: nx.Graph,
    tree: nx.Graph,
) -> None:
    """Reconstruye el árbol óptimo a partir de los punteros guardados.

    Sigue el árbol de derivación de f(D, v):
      - Si |D| = 1, agrega las aristas del camino más corto del terminal
        único a v.
      - Si no, va de v a u (camino más corto), luego baja recursivamente
        por la partición (D1, D2) que produjo g[D][u].
    """
    if len(D) == 1:
        t = next(iter(D))
        p = sp_path[t][v]
        for a, b in zip(p[:-1], p[1:]):
            tree.add_edge(a, b, weight=G[a][b]["weight"])
        return

    u = parent_f[D][v]
    p = sp_path[u][v]
    for a, b in zip(p[:-1], p[1:]):
        tree.add_edge(a, b, weight=G[a][b]["weight"])

    D1, D2 = parent_g[D][u]
    _reconstruct(D1, u, parent_f, parent_g, sp_path, G, tree)
    _reconstruct(D2, u, parent_f, parent_g, sp_path, G, tree)
