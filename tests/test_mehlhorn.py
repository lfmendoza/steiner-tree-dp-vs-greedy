"""
Tests de corrección y cota teórica para Mehlhorn (1988).

Sobre 50 instancias chicas seeded comparamos contra Dreyfus–Wagner:
  - greedy_cost ≥ optimal_cost − 1e-9
  - greedy_cost / optimal_cost ≤ 2·(1 − 1/L) + 1e-9, donde L = hojas de T*
  - el resultado es un árbol válido que cubre todos los terminales
"""
from __future__ import annotations

import unittest

import networkx as nx

from steiner import dreyfus_wagner, is_valid_steiner_tree
from steiner.mehlhorn import mehlhorn
from tests._helpers import small_random_instance


N_SEEDS = 50


def steiner_leaves(tree: nx.Graph) -> int:
    """Hojas del árbol; al menos 2 (o el propio nodo si es trivial)."""
    if tree.number_of_nodes() <= 1:
        return max(1, tree.number_of_nodes())
    return max(2, sum(1 for v in tree.nodes if tree.degree(v) == 1))


class TestMehlhorn(unittest.TestCase):
    def test_correctness_and_bound(self):
        bound_violations = []
        for seed in range(N_SEEDS):
            inst = small_random_instance(seed)
            opt_cost, opt_tree = dreyfus_wagner(inst)
            grd_cost, grd_tree = mehlhorn(inst)

            with self.subTest(seed=seed, kind="cost_lower_bound"):
                self.assertGreaterEqual(grd_cost + 1e-9, opt_cost)

            with self.subTest(seed=seed, kind="valid_tree"):
                self.assertTrue(is_valid_steiner_tree(grd_tree, inst))

            if opt_cost > 0:
                L = steiner_leaves(opt_tree)
                bound = 2.0 * (1.0 - 1.0 / L)
                ratio = grd_cost / opt_cost
                if ratio > bound + 1e-6:
                    bound_violations.append((seed, ratio, bound, L))

        self.assertFalse(
            bound_violations,
            f"Violaciones de la cota 2(1 - 1/L): {bound_violations[:5]} ...",
        )


if __name__ == "__main__":
    unittest.main()
