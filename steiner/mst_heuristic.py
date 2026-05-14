"""
Heurística de Kou–Markowsky–Berman (1981) para Steiner Tree.

Algoritmo
---------
1. Construir la clausura métrica G1 sobre los terminales: grafo completo
   con peso(s, t) = distancia más corta en G.
2. Calcular un árbol generador mínimo T1 de G1.
3. Reemplazar cada arista (s, t) de T1 por el camino más corto en G,
   formando un subgrafo Gs.
4. Calcular un árbol generador mínimo T2 de Gs.
5. Podar hojas no terminales de T2.

Garantía
--------
Costo(T_KMB) ≤ 2·(1 − 1/L) · Costo(T*), donde L es el número de hojas
del árbol de Steiner óptimo T*.

Referencia
----------
L. Kou, G. Markowsky, L. Berman. "A fast algorithm for Steiner trees."
*Acta Informatica* 15 (1981), pp. 141-145.

Complejidad
-----------
Tiempo:  O(k · (m + n log n)) por las k Dijkstras de la clausura.
Espacio: O(n²) por las APSP.
"""
from __future__ import annotations

import networkx as nx

from .graph_utils import (
    Instance,
    all_pairs_shortest_paths,
    induced_metric_closure,
    prune_non_terminal_leaves,
    tree_cost,
)


def mst_heuristic(instance: Instance) -> tuple[float, nx.Graph]:
    """Resuelve Steiner Tree con la heurística KMB.

    Parameters
    ----------
    instance : Instance
        Instancia con grafo y terminales.

    Returns
    -------
    cost : float
        Peso total del árbol greedy.
    tree : networkx.Graph
        Subgrafo de ``instance.graph`` que materializa la solución.
    """
    G = instance.graph
    terminals = instance.terminals

    if len(terminals) == 1:
        t = next(iter(terminals))
        tree = nx.Graph()
        tree.add_node(t)
        return 0.0, tree

    dist, sp_path = all_pairs_shortest_paths(G)
    closure, _, _ = induced_metric_closure(G, terminals, dist=dist, sp_path=sp_path)

    # Paso 2: MST sobre la clausura.
    mst_closure = nx.minimum_spanning_tree(closure, weight="weight")

    # Paso 3: reexpandir cada arista de la clausura por su camino más corto.
    expanded = nx.Graph()
    for s, t in mst_closure.edges():
        path = sp_path[s][t]
        for a, b in zip(path[:-1], path[1:]):
            w = G[a][b]["weight"]
            # Si ya existe la arista, igual seguimos: nx.Graph la sobreescribe
            # con el mismo peso, que es lo deseado para unir caminos solapados.
            expanded.add_edge(a, b, weight=w)

    # Paso 4: MST del subgrafo expandido.
    mst_expanded = nx.minimum_spanning_tree(expanded, weight="weight")

    # Paso 5: podar hojas no terminales.
    pruned = prune_non_terminal_leaves(mst_expanded, terminals)
    return tree_cost(pruned), pruned
