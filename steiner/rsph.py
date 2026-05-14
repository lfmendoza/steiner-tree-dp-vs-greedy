"""
Repetitive Shortest Path Heuristic (RSPH) para Steiner Tree.

A diferencia de KMB / Mehlhorn, RSPH no construye un MST sobre una
clausura: hace crecer el árbol iterativamente, conectando un terminal
a la vez por su camino más corto al árbol parcial.

Algoritmo
---------
1. Tomar un terminal inicial t₀ (el menor por ``sorted``, para
   determinismo).
2. Mientras queden terminales sin conectar:
   a. Dijkstra multifuente desde el árbol parcial.
   b. Elegir el terminal sin conectar con distancia mínima al árbol.
   c. Añadir las aristas del camino más corto que lo une al árbol.
3. Podar hojas no terminales.

Garantía
--------
Costo(T_RSPH) ≤ 2 · (1 − 1/L) · Costo(T*); misma cota teórica que KMB.

Referencias
-----------
S. Voß. "Steiner's problem in graphs: heuristic methods." *Discrete
Applied Mathematics* 40 (1992), pp. 45-72.

Takahashi, H., Matsuyama, A. (1980). "An approximate solution for the
Steiner problem in graphs." *Math. Japonica* 24, pp. 573-577.

Complejidad
-----------
Tiempo:  O(k · (m + n log n))   (k iteraciones, cada una una Dijkstra).
Espacio: O(n + m).
"""
from __future__ import annotations

import heapq

import networkx as nx

from .graph_utils import Instance, prune_non_terminal_leaves, tree_cost


def _dijkstra_from_tree(
    G: nx.Graph, tree_nodes: set, targets: set
) -> tuple[object | None, list]:
    """Encuentra el camino más corto desde cualquier nodo de ``tree_nodes``
    hasta el primer ``target`` alcanzado.

    Returns
    -------
    (target, path) :
        ``target`` es el terminal sin conectar más cercano, ``path`` es
        la lista de vértices del camino (target → ... → algún nodo del
        árbol). Si no se alcanza ningún target, devuelve ``(None, [])``.
    """
    dist: dict = {v: float("inf") for v in G.nodes}
    pred: dict = {v: None for v in G.nodes}
    pq: list[tuple[float, int, object]] = []
    counter = 0
    for v in tree_nodes:
        dist[v] = 0.0
        heapq.heappush(pq, (0.0, counter, v))
        counter += 1
    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u in targets:
            path = [u]
            cur = u
            while pred[cur] is not None:
                cur = pred[cur]
                path.append(cur)
            return u, path
        for v, data in G[u].items():
            nd = d + data["weight"]
            if nd < dist[v] - 1e-15:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, counter, v))
                counter += 1
    return None, []


def rsph(instance: Instance) -> tuple[float, nx.Graph]:
    """Resuelve Steiner Tree con la Repetitive Shortest Path Heuristic.

    Parameters
    ----------
    instance : Instance

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

    # Determinismo: ordenamos los terminales según una clave estable.
    sorted_terms = sorted(terminals, key=lambda x: (str(type(x).__name__), repr(x)))
    t0 = sorted_terms[0]

    tree = nx.Graph()
    tree.add_node(t0)
    tree_nodes: set = {t0}
    remaining: set = set(terminals) - {t0}

    while remaining:
        target, path = _dijkstra_from_tree(G, tree_nodes, remaining)
        if target is None:
            # No alcanzable: la instancia debería ser conexa, así que esto
            # no debería ocurrir. Fallar ruidosamente.
            raise RuntimeError(
                f"RSPH: no se pudo conectar terminales {remaining} al árbol parcial."
            )
        for a, b in zip(path[:-1], path[1:]):
            tree.add_edge(a, b, weight=G[a][b]["weight"])
            tree_nodes.add(a)
            tree_nodes.add(b)
        remaining.discard(target)

    # El árbol parcial puede tener hojas no terminales si el último camino
    # añadido dejó "colas". Una poda final lo limpia.
    pruned = prune_non_terminal_leaves(tree, terminals)
    return tree_cost(pruned), pruned


def rsph_steps(instance: Instance):
    """Versión generadora para animaciones: yields ``(step, target, partial_tree)``.

    Genera el estado del árbol después de cada terminal añadido, lo cual
    es útil para :func:`viz.animate.animate_rsph`.
    """
    G = instance.graph
    terminals = instance.terminals

    if len(terminals) == 1:
        t = next(iter(terminals))
        tree = nx.Graph()
        tree.add_node(t)
        yield 0, t, tree.copy()
        return

    sorted_terms = sorted(terminals, key=lambda x: (str(type(x).__name__), repr(x)))
    t0 = sorted_terms[0]
    tree = nx.Graph()
    tree.add_node(t0)
    tree_nodes: set = {t0}
    remaining: set = set(terminals) - {t0}
    yield 0, t0, tree.copy()

    step = 1
    while remaining:
        target, path = _dijkstra_from_tree(G, tree_nodes, remaining)
        if target is None:
            raise RuntimeError("RSPH: terminales no alcanzables.")
        for a, b in zip(path[:-1], path[1:]):
            tree.add_edge(a, b, weight=G[a][b]["weight"])
            tree_nodes.add(a)
            tree_nodes.add(b)
        remaining.discard(target)
        yield step, target, tree.copy()
        step += 1

    pruned = prune_non_terminal_leaves(tree, terminals)
    yield step, None, pruned
