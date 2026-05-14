"""
bench — pipeline empírico para Steiner Tree DP vs Greedy.

- :mod:`bench.timing`           — medición de tiempos con warmup y repeticiones.
- :mod:`bench.quality`          — estadística descriptiva de cocientes greedy/óptimo.
- :mod:`bench.regression`       — ajustes polinomiales con bootstrap CI.
- :mod:`bench.run_experiments`  — CLI principal que orquesta los sweeps.
- :mod:`bench.analyze`          — CSV → figuras + tablas LaTeX.
- :mod:`bench.fetch_steinlib`   — descarga el subset B de SteinLib.
"""
from __future__ import annotations
