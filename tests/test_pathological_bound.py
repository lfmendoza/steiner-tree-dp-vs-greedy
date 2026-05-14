"""
Test asintótico de la familia spider.

Verifica empíricamente que la cota teórica 2(1 − 1/k) de KMB es
esencialmente apretada para la construcción de :func:`spider`.

Razonamiento numérico
---------------------
Para ``spider(k, epsilon)``:
    opt   = k                           (estrella en el Steiner)
    kmb   = (k - 1) · (2 − epsilon)     (cadena perimetral)
    ratio = (k - 1)(2 − epsilon) / k

El supremo de ratio cuando ``epsilon → 0`` es ``2(1 − 1/k)``; para
``epsilon = 0.01`` el ratio se queda ``0.5·epsilon·(1 − 1/k)`` por
debajo de la cota. Para garantizar ratio ≥ 1.9 con ``epsilon = 0.01``
hace falta ``k ≥ 21`` (en k=20 sale 1.8905). Por eso el test exigente
se corre a ``k = 40``.

  * Con ``k=10, epsilon=0.01``: cociente cercano (5%) a 1.8.
  * Con ``k=40, epsilon=0.01``: cociente ≥ 1.9 (define el "done").

También chequea que Dreyfus–Wagner identifica el óptimo (la estrella)
en el caso chico.
"""
from __future__ import annotations

import unittest

from steiner import dreyfus_wagner, mst_heuristic
from steiner.instances import spider


class TestPathologicalBound(unittest.TestCase):
    def test_spider_k10(self):
        k = 10
        inst = spider(k=k, epsilon=0.01)
        opt_cost, _ = dreyfus_wagner(inst)
        grd_cost, _ = mst_heuristic(inst)
        ratio = grd_cost / opt_cost
        expected = 2.0 * (1.0 - 1.0 / k) * (1.0 - 0.01 / 2.0)
        # Tolerancia del 5%.
        self.assertAlmostEqual(opt_cost, float(k), places=6)
        self.assertLess(abs(ratio - expected) / expected, 0.05)

    def test_spider_k40_meets_threshold(self):
        k = 40
        inst = spider(k=k, epsilon=0.01)
        # Para la spider el óptimo es exactamente k (estrella centrada
        # en el único Steiner s). Evitamos correr DP a k grande
        # (que sería 3^k · n operaciones) y usamos esa propiedad
        # estructural demostrada en el análisis teórico.
        opt_cost_analytic = float(k)
        grd_cost, _ = mst_heuristic(inst)
        ratio = grd_cost / opt_cost_analytic
        self.assertGreaterEqual(
            ratio,
            1.9,
            f"Spider(k={k}, eps=0.01): ratio={ratio:.6f} debe ser >= 1.9",
        )


if __name__ == "__main__":
    unittest.main()
