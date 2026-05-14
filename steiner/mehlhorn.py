"""
Heurística de Mehlhorn (1988) para Steiner Tree.

Mejora de KMB que evita las k Dijkstras de la clausura métrica
explícita; en su lugar usa una única Dijkstra multifuente desde
los terminales y construye un grafo auxiliar tipo Voronoi.

Algoritmo
---------
1. Dijkstra multifuente: para cada vértice v ∈ V, registrar
       π(v) = terminal más cercano,
       d(v) = distancia a π(v),
       pred(v) = predecesor en el árbol de caminos más cortos.
2. Construir grafo auxiliar G' sobre los terminales: para cada arista
   (u, v) ∈ E con π(u) ≠ π(v), considerar la arista
       (π(u), π(v))   con peso  d(u) + w(u, v) + d(v).
   Conservar, para cada par de terminales, la arista de mínimo peso
   junto con el "testigo" (u, v) que la produce.
3. Calcular el MST T' de G'.
4. Reexpandir cada arista de T' por el camino π(u) → u y v → π(v),
   uniendo todo en un subgrafo Gs.
5. MST de Gs y poda de hojas no terminales (igual que KMB).

Garantía
--------
Costo(T_M) ≤ 2 · (1 − 1/L) · Costo(T*); misma cota que KMB.

Referencia
----------
K. Mehlhorn. "A faster approximation algorithm for the Steiner
problem in graphs." *Information Processing Letters* 27 (1988),
no. 3, pp. 125-128.

Complejidad
-----------
Tiempo:  O(m + n log n)        (una Dijkstra multifuente + MST sobre G').
Espacio: O(n + m).
"""
from __future__ import annotations

import heapq

import networkx as nx

from .graph_utils import Instance, prune_non_terminal_leaves, tree_cost


def _multi_source_dijkstra(
    G: nx.Graph, terminals: frozenset
) -> tuple[dict, dict, dict]:
    """Dijkstra multifuente desde todos los terminales simultáneamente.

    Devuelve tres diccionarios:
      ``base[v]``  = terminal más cercano a ``v``.
      ``dist[v]``  = distancia a ese terminal.
      ``pred[v]``  = predecesor de ``v`` en el árbol de caminos más cortos
                     (o ``None`` para los terminales mismos).
    """
    dist: dict = {v: float("inf") for v in G.nodes}
    base: dict = {v: None for v in G.nodes}
    pred: dict = {v: None for v in G.nodes}
    pq: list[tuple[float, int, object]] = []
    counter = 0
    for t in terminals:
        dist[t] = 0.0
        base[t] = t
        heapq.heappush(pq, (0.0, counter, t))
        counter += 1
    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, data in G[u].items():
            w = data["weight"]
            nd = d + w
            if nd < dist[v] - 1e-15:
                dist[v] = nd
                base[v] = base[u]
                pred[v] = u
                heapq.heappush(pq, (nd, counter, v))
                counter += 1
    return base, dist, pred


def _path_to_terminal(v, pred: dict) -> list:
    """Devuelve la lista de vértices de ``v`` hasta su terminal base (incluido)."""
    path = [v]
    while pred[v] is not None:
        v = pred[v]
        path.append(v)
    return path


def mehlhorn(instance: Instance) -> tuple[float, nx.Graph]:
    """Resuelve Steiner Tree con la heurística de Mehlhorn.

    Parameters
    ----------
    instance : Instance
        Instancia con grafo y terminales.

    Returns
    -------
    cost : float
    tree : networkx.Graph
    """
    G = instance.graph
    terminals = instance.terminals

    if len(terminals) == 1:
        t = next(iter(terminals))
        tree = nx.Graph()
        tree.add_node(t)
        return 0.0, tree

    base, dist, pred = _multi_source_dijkstra(G, terminals)

    # Paso 2: construir el grafo auxiliar G' sobre terminales.
    # Para cada par de terminales conservar la mejor arista candidata
    # junto con sus dos extremos (u, v) en G.
    aux_best: dict[tuple, tuple[float, object, object]] = {}
    for u, v, data in G.edges(data=True):
        bu, bv = base[u], base[v]
        if bu is None or bv is None or bu == bv:
            continue
        key = (bu, bv) if bu < bv else (bv, bu)
        weight = dist[u] + data["weight"] + dist[v]
        cur = aux_best.get(key)
        if cur is None or weight < cur[0]:
            aux_best[key] = (weight, u, v)

    G_prime = nx.Graph()
    G_prime.add_nodes_from(terminals)
    for (a, b), (w, _, _) in aux_best.items():
        G_prime.add_edge(a, b, weight=w)

    # Si G' no es conexo, el grafo original no permite conectar todos los
    # terminales por aristas con bases distintas (cosa rara en un G conexo,
    # pero por completitud caemos a un MST sobre la clausura completa).
    if not nx.is_connected(G_prime):
        # Como fallback, importar KMB clásico.
        from .mst_heuristic import mst_heuristic as _kmb

        return _kmb(instance)

    # Paso 3: MST de G'.
    mst_prime = nx.minimum_spanning_tree(G_prime, weight="weight")

    # Paso 4: reexpandir cada arista (a, b) ∈ mst_prime usando su testigo.
    expanded = nx.Graph()
    for a, b in mst_prime.edges():
        key = (a, b) if a < b else (b, a)
        _, u, v = aux_best[key]
        # Camino π(u) = a hacia u, luego la arista (u, v), luego v hacia π(v) = b.
        path_u = _path_to_terminal(u, pred)  # u, pred(u), ..., a
        path_v = _path_to_terminal(v, pred)  # v, pred(v), ..., b
        for x, y in zip(path_u[:-1], path_u[1:]):
            expanded.add_edge(x, y, weight=G[x][y]["weight"])
        if u != v:
            expanded.add_edge(u, v, weight=G[u][v]["weight"])
        for x, y in zip(path_v[:-1], path_v[1:]):
            expanded.add_edge(x, y, weight=G[x][y]["weight"])

    # Paso 5: MST + poda de hojas no terminales.
    mst_expanded = nx.minimum_spanning_tree(expanded, weight="weight")
    pruned = prune_non_terminal_leaves(mst_expanded, terminals)
    return tree_cost(pruned), pruned
