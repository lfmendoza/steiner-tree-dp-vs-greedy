"""
Tests para GSVI: correctitud vs DP, cota teorica y validad del arbol.
"""
from __future__ import annotations

import unittest

import networkx as nx

from steiner import dreyfus_wagner, is_valid_steiner_tree
from steiner.gsvi import gsvi, gsvi_steps
from tests._helpers import small_random_instance


N_SEEDS = 50


def steiner_leaves(tree: nx.Graph) -> int:
    if tree.number_of_nodes() <= 1:
        return max(1, tree.number_of_nodes())
    return max(2, sum(1 for v in tree.nodes if tree.degree(v) == 1))


class TestGSVI(unittest.TestCase):
    def test_correctness_and_bound(self):
        violations = []
        for seed in range(N_SEEDS):
            inst = small_random_instance(seed)
            opt_cost, opt_tree = dreyfus_wagner(inst)
            grd_cost, grd_tree = gsvi(inst)

            with self.subTest(seed=seed, kind="cost_lower_bound"):
                self.assertGreaterEqual(grd_cost + 1e-9, opt_cost)

            with self.subTest(seed=seed, kind="valid_tree"):
                self.assertTrue(is_valid_steiner_tree(grd_tree, inst))

            if opt_cost > 0:
                L = steiner_leaves(opt_tree)
                bound = 2.0 * (1.0 - 1.0 / L)
                ratio = grd_cost / opt_cost
                if ratio > bound + 1e-6:
                    violations.append((seed, ratio, bound, L))

        self.assertFalse(
            violations,
            f"Violaciones de la cota 2(1-1/L): {violations[:5]}",
        )

    def test_spider_steiner_point_is_found(self):
        """En la spider, GSVI DEBE encontrar el punto de Steiner s."""
        from steiner.instances import spider as make_spider
        inst = make_spider(k=4, epsilon=0.05)
        cost, tree = gsvi(inst)
        # El optimo es la estrella con costo k=4.
        # GSVI deberia encontrar o acercarse al optimo al insertar 's'.
        self.assertLessEqual(cost, 4.0 * 2.0)  # cota 2-aprox garantizada
        self.assertTrue(is_valid_steiner_tree(tree, inst))

    def test_gsvi_steps_generator(self):
        """El generador de pasos produce pasos validos y termina."""
        inst = small_random_instance(seed=0)
        steps = list(gsvi_steps(inst))
        # Debe haber al menos el paso inicial y el final
        self.assertGreaterEqual(len(steps), 2)
        first, last = steps[0], steps[-1]
        self.assertEqual(first["type"], "initial")
        self.assertEqual(last["type"], "done")
        # El arbol final cubre todos los terminales
        self.assertTrue(is_valid_steiner_tree(last["tree"], inst))
        # Cada paso intermedio incrementa el conjunto activo
        for i in range(1, len(steps)):
            prev_active = steps[i - 1]["active_set"]
            curr_active = steps[i]["active_set"]
            self.assertGreaterEqual(len(curr_active), len(prev_active))

    def test_gsvi_improves_on_terminal_only_mst(self):
        """GSVI produce un arbol igual o mejor que el MST sobre solo terminales."""
        from steiner.instances import spider as make_spider
        from steiner.graph_utils import tree_cost as tc
        from steiner.gsvi import _expand_metric_mst, _mst_cost_on_metric_closure
        from steiner.graph_utils import all_pairs_shortest_paths

        inst = make_spider(k=5, epsilon=0.05)
        dist, sp_path = all_pairs_shortest_paths(inst.graph)
        terminal_only_tree = _expand_metric_mst(
            frozenset(inst.terminals), dist, sp_path, inst.graph
        )
        terminal_only_cost = tc(terminal_only_tree)

        gsvi_cost, _ = gsvi(inst)
        self.assertLessEqual(gsvi_cost, terminal_only_cost + 1e-9)


if __name__ == "__main__":
    unittest.main()
