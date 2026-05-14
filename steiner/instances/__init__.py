"""
Generadores de instancias para Steiner Tree.

Cada submódulo aporta una familia estructuralmente distinta:

- :mod:`steiner.instances.random_er`     — Erdős–Rényi ponderado.
- :mod:`steiner.instances.euclidean`     — puntos uniformes en el plano.
- :mod:`steiner.instances.geometric`     — random geometric graph.
- :mod:`steiner.instances.pathological`  — familia tight para la cota 2.
- :mod:`steiner.instances.steinlib`      — parser del formato ``.stp``.
"""
from __future__ import annotations

from .euclidean import euclidean
from .geometric import geometric
from .pathological import double_spider, grid_with_shortcut, spider
from .random_er import random_er
from .steinlib import list_steinlib, load_steinlib_B, parse_stp

__all__ = [
    "double_spider",
    "euclidean",
    "geometric",
    "grid_with_shortcut",
    "list_steinlib",
    "load_steinlib_B",
    "parse_stp",
    "random_er",
    "spider",
]
