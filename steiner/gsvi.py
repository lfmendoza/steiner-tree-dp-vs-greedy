"""
Greedy Steiner Vertex Insertion Heuristic (GSVI).

Criterio greedy fundamentalmente distinto al de KMB/Mehlhorn/RSPH:

* KMB, Mehlhorn, RSPH: piensan en *caminos entre terminales* (clausura
  metrica o camino mas corto al arbol parcial).
* GSVI: piensa en *que vertices de Steiner incluir explicitamente* como
  puntos de ramificacion para reducir el costo del arbol.

Algoritmo
---------
1. Inicializar con el MST de la clausura metrica sobre los terminales
   solamente (mismo punto de partida que KMB, para que la comparacion
   sea justa). Expandir a caminos reales en G.
2. Iterativamente: para cada vertice no-terminal v no incluido todavia,
   calcular el *ahorro* que produce al anhadirlo al conjunto activo:
       ahorro(v) = MST_cost(activo) - MST_cost(activo U {v})
   El ahorro es positivo cuando v puede actuar como HUB que permite
   conectar varios terminales de forma mas barata que las rutas directas.
3. Insertar el vertice con mayor ahorro. Repetir hasta que ningun vertice
   aporte ahorro positivo.
4. Expandir el MST final sobre (terminales U vertices de Steiner elegidos)
   a caminos reales en G, tomar MST del resultado y podar hojas.

Diferencia clave respecto a los otros greedies
-----------------------------------------------
Los algoritmos basados en clausura metrica deciden implicitamente los
puntos de Steiner al expandir caminos. GSVI los decide *explicitamente*
y de forma incremental, lo que produce arboles estructuralmente distintos
en grafos donde un solo hub reduce el costo de varias rutas simultaneamente.

Referencias
-----------
Takahashi, H., Matsuyama, A. (1980). Sec. 5 — discute insercion de
Steiner points como refinamiento iterativo.
Voß, S. (1992). "Steiner's problem in graphs: heuristic methods."
Discrete Applied Mathematics 40, pp. 45-72.

Complejidad
-----------
Tiempo:  O(n * n * n^2) = O(n^4) en el peor caso, pero en la practica
         pocas iteraciones. Para instancias de demo (n<=30) es rapido.
Espacio: O(n^2) por APSP.
"""
from __future__ import annotations

from typing import Generator

import networkx as nx

from .graph_utils import (
    Instance,
    all_pairs_shortest_paths,
    prune_non_terminal_leaves,
    tree_cost,
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _mst_cost_on_metric_closure(nodes: frozenset, dist: dict) -> float:
    """Costo del MST sobre la clausura metrica de `nodes`."""
    nodes_list = sorted(nodes, key=repr)
    if len(nodes_list) <= 1:
        return 0.0
    g = nx.Graph()
    for i, s in enumerate(nodes_list):
        for t in nodes_list[i + 1 :]:
            g.add_edge(s, t, weight=dist[s][t])
    mst = nx.minimum_spanning_tree(g, weight="weight")
    return float(sum(d["weight"] for _, _, d in mst.edges(data=True)))


def _expand_metric_mst(active: frozenset, dist: dict, sp_path: dict, G: nx.Graph) -> nx.Graph:
    """Construye el arbol expandido a partir del MST de la clausura metrica de `active`."""
    nodes_list = sorted(active, key=repr)
    closure = nx.Graph()
    for i, s in enumerate(nodes_list):
        for t in nodes_list[i + 1 :]:
            closure.add_edge(s, t, weight=dist[s][t])
    mst_closure = nx.minimum_spanning_tree(closure, weight="weight")

    expanded = nx.Graph()
    for s, t in mst_closure.edges():
        path = sp_path[s][t]
        for a, b in zip(path[:-1], path[1:]):
            expanded.add_edge(a, b, weight=G[a][b]["weight"])

    return nx.minimum_spanning_tree(expanded, weight="weight")


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------


def gsvi(instance: Instance) -> tuple[float, nx.Graph]:
    """Heuristica GSVI: insercion codiciosa de puntos de Steiner.

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
    dist, sp_path = all_pairs_shortest_paths(G)

    all_vertices = list(G.nodes)
    active = frozenset(terminals)

    # Iterar hasta que no haya mas ahorros positivos
    while True:
        current_cost = _mst_cost_on_metric_closure(active, dist)
        best_v = None
        best_savings = 1e-9  # umbral minimo para evitar insertar por ruido numerico

        for v in all_vertices:
            if v in active:
                continue
            new_cost = _mst_cost_on_metric_closure(active | {v}, dist)
            savings = current_cost - new_cost
            if savings > best_savings:
                best_savings = savings
                best_v = v

        if best_v is None:
            break
        active = active | {best_v}

    # Construir el arbol expandido final
    expanded_mst = _expand_metric_mst(active, dist, sp_path, G)
    pruned = prune_non_terminal_leaves(expanded_mst, terminals)
    return tree_cost(pruned), pruned


# ---------------------------------------------------------------------------
# Generador de pasos para la visualizacion interactiva
# ---------------------------------------------------------------------------


def gsvi_steps(instance: Instance) -> Generator[dict, None, None]:
    """Genera el estado del algoritmo GSVI paso a paso.

    Cada paso es un diccionario con:
      ``step_num``        : indice del paso (0-based).
      ``type``            : ``'initial'``, ``'insert'``, ``'done'``.
      ``active_set``      : frozenset de vertices activos (terminales +
                            Steiner insertados hasta este paso).
      ``tree``            : arbol expandido en este paso.
      ``candidate_savings``: dict {v: savings} para todos los candidatos.
      ``inserted_vertex`` : vertice insertado en este paso (o None).
      ``best_savings``    : ahorro del mejor candidato.
      ``description``     : descripcion legible del paso.
    """
    G = instance.graph
    terminals = instance.terminals
    dist, sp_path = all_pairs_shortest_paths(G)
    all_vertices = list(G.nodes)
    active = frozenset(terminals)

    # Paso 0: arbol inicial sobre solo terminales
    initial_tree = _expand_metric_mst(active, dist, sp_path, G)
    initial_cost = _mst_cost_on_metric_closure(active, dist)

    # Calcular ahorros iniciales para todos los candidatos
    initial_savings: dict = {}
    for v in all_vertices:
        if v not in active:
            new_cost = _mst_cost_on_metric_closure(active | {v}, dist)
            s = initial_cost - new_cost
            initial_savings[v] = max(0.0, s)

    yield {
        "step_num": 0,
        "type": "initial",
        "active_set": active,
        "tree": initial_tree,
        "candidate_savings": initial_savings,
        "inserted_vertex": None,
        "best_savings": max(initial_savings.values()) if initial_savings else 0.0,
        "description": (
            f"Estado inicial: MST sobre los {len(terminals)} terminales. "
            f"Costo = {tree_cost(initial_tree):.4f}. "
            "Los nodos naranjas son candidatos a punto de Steiner; "
            "el tamaño indica el ahorro potencial."
        ),
    }

    step = 1
    while True:
        current_cost = _mst_cost_on_metric_closure(active, dist)
        best_v = None
        best_savings_val = 1e-9

        candidate_savings: dict = {}
        for v in all_vertices:
            if v in active:
                continue
            new_cost = _mst_cost_on_metric_closure(active | {v}, dist)
            s = current_cost - new_cost
            candidate_savings[v] = max(0.0, s)
            if s > best_savings_val:
                best_savings_val = s
                best_v = v

        if best_v is None:
            break

        active = active | {best_v}
        new_tree = _expand_metric_mst(active, dist, sp_path, G)

        yield {
            "step_num": step,
            "type": "insert",
            "active_set": active,
            "tree": new_tree,
            "candidate_savings": candidate_savings,
            "inserted_vertex": best_v,
            "best_savings": best_savings_val,
            "description": (
                f"Paso {step}: se inserta el punto de Steiner '{best_v}' "
                f"(ahorro = {best_savings_val:.4f}). "
                f"Nuevo costo del arbol = {tree_cost(new_tree):.4f}."
            ),
        }
        step += 1

    # Paso final: poda
    final_tree = _expand_metric_mst(active, dist, sp_path, G)
    pruned = prune_non_terminal_leaves(final_tree, terminals)

    yield {
        "step_num": step,
        "type": "done",
        "active_set": active,
        "tree": pruned,
        "candidate_savings": {},
        "inserted_vertex": None,
        "best_savings": 0.0,
        "description": (
            f"Resultado final tras poda de hojas no terminales. "
            f"Puntos de Steiner insertados: "
            f"{sorted(active - terminals, key=repr)}. "
            f"Costo final = {tree_cost(pruned):.4f}."
        ),
    }
