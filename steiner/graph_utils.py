"""
Estructuras de datos y utilidades comunes para el problema de Steiner Tree.

Define la clase `Instance` (grafo + terminales) y rutinas auxiliares
compartidas por todos los algoritmos del paquete: caminos más cortos,
costo de árboles, validación, clausura métrica y conteo de hojas.

Referencias
-----------
Hwang, F. K., Richards, D. S., Winter, P. (1992). *The Steiner Tree
Problem*. Annals of Discrete Mathematics 53, North-Holland.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import networkx as nx


@dataclass(frozen=True)
class Instance:
    """Instancia del problema de Steiner Tree en grafos.

    Parameters
    ----------
    graph : networkx.Graph
        Grafo no dirigido conexo con pesos no negativos en el atributo
        ``weight`` de cada arista.
    terminals : frozenset
        Conjunto de vértices terminales que el árbol debe contener.
        ``terminals`` debe ser subconjunto de ``graph.nodes``.

    Invariantes (verificadas en :meth:`__post_init__`)
    --------------------------------------------------
    1. ``terminals`` no es vacío.
    2. ``terminals ⊆ graph.nodes``.
    3. Todas las aristas tienen ``weight`` definido y no negativo.
    4. El grafo es conexo (sobre todos sus vértices).
    """

    graph: nx.Graph
    terminals: frozenset

    def __post_init__(self) -> None:
        if not isinstance(self.terminals, frozenset):
            object.__setattr__(self, "terminals", frozenset(self.terminals))

        if len(self.terminals) == 0:
            raise ValueError("El conjunto de terminales no puede ser vacío.")

        missing = self.terminals - set(self.graph.nodes)
        if missing:
            raise ValueError(
                f"Terminales no presentes en el grafo: {sorted(map(str, missing))}"
            )

        for u, v, data in self.graph.edges(data=True):
            if "weight" not in data:
                raise ValueError(f"Arista ({u}, {v}) sin atributo 'weight'.")
            if data["weight"] < 0:
                raise ValueError(
                    f"Arista ({u}, {v}) tiene peso negativo {data['weight']}."
                )

        if self.graph.number_of_nodes() > 0 and not nx.is_connected(self.graph):
            raise ValueError("El grafo de la instancia no es conexo.")

    @property
    def n(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def m(self) -> int:
        return self.graph.number_of_edges()

    @property
    def k(self) -> int:
        return len(self.terminals)


def all_pairs_shortest_paths(
    G: nx.Graph,
) -> tuple[dict[Hashable, dict[Hashable, float]], dict[Hashable, dict[Hashable, list]]]:
    """Computa distancias y caminos más cortos entre todos los pares.

    Wrapper sobre :func:`networkx.all_pairs_dijkstra` que devuelve
    diccionarios anidados independientes (no vistas perezosas).

    Parameters
    ----------
    G : networkx.Graph
        Grafo con pesos no negativos en el atributo ``weight``.

    Returns
    -------
    dist : dict
        ``dist[u][v]`` es la distancia más corta de ``u`` a ``v``.
    sp_path : dict
        ``sp_path[u][v]`` es la lista de vértices que forman el camino
        más corto de ``u`` a ``v`` (incluye ambos extremos).
    """
    dist: dict[Hashable, dict[Hashable, float]] = {}
    sp_path: dict[Hashable, dict[Hashable, list]] = {}
    for source, (d_src, p_src) in nx.all_pairs_dijkstra(G, weight="weight"):
        dist[source] = dict(d_src)
        sp_path[source] = {target: list(path) for target, path in p_src.items()}
    return dist, sp_path


def tree_cost(tree: nx.Graph) -> float:
    """Suma de los pesos de las aristas del árbol/grafo dado."""
    return float(sum(data.get("weight", 0.0) for _, _, data in tree.edges(data=True)))


def is_valid_steiner_tree(tree: nx.Graph, instance: Instance) -> bool:
    """Verifica que ``tree`` sea un árbol válido que cubre todos los terminales.

    Comprueba:
      - ``tree`` es un árbol (acíclico y conexo, o un único nodo).
      - Todos los terminales aparecen en ``tree.nodes``.
      - Todas las aristas de ``tree`` existen en ``instance.graph`` con
        el mismo peso.
    """
    if tree.number_of_nodes() == 0:
        return False
    if not instance.terminals.issubset(set(tree.nodes)):
        return False
    if tree.number_of_nodes() == 1:
        return tree.number_of_edges() == 0
    if not nx.is_tree(tree):
        return False
    G = instance.graph
    for u, v, data in tree.edges(data=True):
        if not G.has_edge(u, v):
            return False
        if abs(G[u][v]["weight"] - data.get("weight", 0.0)) > 1e-9:
            return False
    return True


def induced_metric_closure(
    G: nx.Graph,
    terminals: frozenset,
    dist: dict | None = None,
    sp_path: dict | None = None,
) -> tuple[nx.Graph, dict, dict]:
    """Clausura métrica de ``G`` restringida a los terminales.

    El grafo resultante es completo sobre ``terminals``; el peso de la
    arista ``(s, t)`` es la distancia más corta entre ``s`` y ``t`` en
    ``G``. Es la construcción base de KMB (Kou–Markowsky–Berman, 1981).

    Parameters
    ----------
    G : networkx.Graph
        Grafo subyacente.
    terminals : frozenset
        Subconjunto de terminales sobre el que se construye la clausura.
    dist, sp_path : dict, optional
        Si ya se computaron las APSP, pasarlas para evitar recomputo.

    Returns
    -------
    closure : networkx.Graph
        Grafo completo con vértices = ``terminals``.
    dist, sp_path : dict
        Las mismas estructuras devueltas por :func:`all_pairs_shortest_paths`.
    """
    if dist is None or sp_path is None:
        dist, sp_path = all_pairs_shortest_paths(G)
    closure = nx.Graph()
    closure.add_nodes_from(terminals)
    terms = sorted(terminals)
    for i, s in enumerate(terms):
        for t in terms[i + 1 :]:
            closure.add_edge(s, t, weight=dist[s][t])
    return closure, dist, sp_path


def prune_non_terminal_leaves(tree: nx.Graph, terminals: frozenset) -> nx.Graph:
    """Elimina iterativamente las hojas no terminales del árbol.

    Devuelve una copia modificada; no muta ``tree``. Si el resultado
    quedara vacío (caso degenerado), devuelve el árbol original.
    """
    T = tree.copy()
    changed = True
    while changed:
        changed = False
        leaves = [v for v in list(T.nodes) if T.degree(v) == 1 and v not in terminals]
        if not leaves:
            break
        T.remove_nodes_from(leaves)
        changed = True
    if T.number_of_nodes() == 0:
        return tree.copy()
    return T


def count_leaves(tree: nx.Graph) -> int:
    """Cuenta el número de hojas (grado 1) del árbol."""
    if tree.number_of_nodes() <= 1:
        return tree.number_of_nodes()
    return sum(1 for v in tree.nodes if tree.degree(v) == 1)
