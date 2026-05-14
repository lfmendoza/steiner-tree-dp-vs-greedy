"""
Utilidades compartidas por la batería de tests.

Genera instancias chicas y conectadas de manera determinista, sin
depender de los generadores formales de ``steiner.instances`` (que
se prueban por separado en Fase 2).
"""
from __future__ import annotations

import random

import networkx as nx

from steiner import Instance


def small_random_instance(
    seed: int,
    n_range: tuple[int, int] = (5, 12),
    k_range: tuple[int, int] = (2, 5),
    w_max: float = 10.0,
    max_attempts: int = 50,
) -> Instance:
    """Devuelve una instancia conexa pequeña reproducible bajo ``seed``."""
    rng = random.Random(seed)
    for _ in range(max_attempts):
        n = rng.randint(*n_range)
        p = rng.uniform(0.35, 0.85)
        G = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9))
        if not nx.is_connected(G):
            continue
        for u, v in G.edges():
            G[u][v]["weight"] = round(rng.uniform(1.0, w_max), 3)
        k = min(rng.randint(*k_range), n)
        terminals = frozenset(rng.sample(list(G.nodes), k))
        return Instance(graph=G, terminals=terminals)
    raise RuntimeError(
        f"No se pudo generar una instancia conexa tras {max_attempts} intentos (seed={seed})."
    )
