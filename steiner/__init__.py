"""
steiner — Algoritmos para el problema del árbol de Steiner en grafos.

Reexporta el API público de los módulos principales:

- :class:`Instance`           — par grafo + terminales.
- :func:`dreyfus_wagner`      — algoritmo exacto por programación dinámica.
- :func:`mst_heuristic`       — heurística KMB (Kou–Markowsky–Berman, 1981).
- :func:`mehlhorn`            — heurística de Mehlhorn (1988).
- :func:`rsph`                — Repetitive Shortest Path Heuristic.
"""
from __future__ import annotations

from .dreyfus_wagner import dreyfus_wagner
from .graph_utils import (
    Instance,
    all_pairs_shortest_paths,
    count_leaves,
    induced_metric_closure,
    is_valid_steiner_tree,
    prune_non_terminal_leaves,
    tree_cost,
)
from .gsvi import gsvi
from .mehlhorn import mehlhorn
from .mst_heuristic import mst_heuristic
from .rsph import rsph

__all__ = [
    "Instance",
    "all_pairs_shortest_paths",
    "count_leaves",
    "dreyfus_wagner",
    "gsvi",
    "induced_metric_closure",
    "is_valid_steiner_tree",
    "mehlhorn",
    "mst_heuristic",
    "prune_non_terminal_leaves",
    "rsph",
    "tree_cost",
]
