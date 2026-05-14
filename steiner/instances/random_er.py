"""
Generador de instancias Erdős–Rényi ponderadas.

Construye G(n, p) y le asigna pesos uniformes en ``[1, w_max]`` a cada
arista. Reintenta hasta obtener un grafo conexo (o eleva ``p`` para
asegurar conectividad cuando ``p`` es muy bajo). Selecciona ``k``
terminales al azar.

Referencias
-----------
Erdős, P., Rényi, A. (1960). "On the evolution of random graphs."
*Publ. Math. Inst. Hungarian Acad. Sci.* 5, pp. 17-61.
"""
from __future__ import annotations

import random

import networkx as nx

from ..graph_utils import Instance


def random_er(
    n: int,
    p: float,
    k: int,
    w_max: float = 10.0,
    seed: int = 0,
    max_attempts: int = 100,
) -> Instance:
    """Erdős–Rényi G(n, p) con pesos U[1, w_max] y ``k`` terminales al azar.

    Parameters
    ----------
    n : int
        Número de vértices, ``n >= 1``.
    p : float
        Probabilidad de cada arista, ``p ∈ (0, 1]``.
    k : int
        Número de terminales, ``1 <= k <= n``.
    w_max : float, default 10.0
        Tope superior del intervalo de pesos uniformes.
    seed : int, default 0
        Semilla reproducible.
    max_attempts : int, default 100
        Reintentos antes de aumentar ``p`` para forzar conectividad.

    Returns
    -------
    Instance
    """
    if n < 1:
        raise ValueError("n debe ser >= 1.")
    if not 0 < p <= 1.0:
        raise ValueError("p debe estar en (0, 1].")
    if not 1 <= k <= n:
        raise ValueError("k debe satisfacer 1 <= k <= n.")
    if w_max <= 0:
        raise ValueError("w_max debe ser positivo.")

    rng = random.Random(seed)
    G = None
    p_eff = p
    for attempt in range(max_attempts):
        sub_seed = rng.randint(0, 10**9)
        candidate = nx.erdos_renyi_graph(n, p_eff, seed=sub_seed)
        if nx.is_connected(candidate):
            G = candidate
            break
        # Cada cierto número de fallos, suben las probabilidades.
        if attempt > 0 and attempt % 10 == 0:
            p_eff = min(1.0, p_eff * 1.25)

    if G is None:
        # Fallback: conectar todos los componentes con un árbol generador.
        candidate = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9))
        components = list(nx.connected_components(candidate))
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            candidate.add_edge(u, v)
        G = candidate

    for u, v in G.edges():
        G[u][v]["weight"] = round(rng.uniform(1.0, w_max), 6)

    terminals = frozenset(rng.sample(list(G.nodes), k))
    return Instance(graph=G, terminals=terminals)
