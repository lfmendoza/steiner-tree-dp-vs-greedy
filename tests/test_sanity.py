"""
Pruebas de sanidad sobre instancias verificables a mano.

Cubre:
  * Instancia "araña" con 4 terminales: óptimo 4.0, KMB 5.7.
  * Triángulo simple: ambos algoritmos coinciden en 3.0.
  * Casos extremos: 1 terminal, 2 terminales (camino más corto),
    todos los vértices terminales (debe coincidir con MST).
  * Validación de invariantes de :class:`Instance`.
  * Validez estructural del árbol greedy.
"""
from __future__ import annotations

import math
import unittest

import networkx as nx

from steiner import Instance, dreyfus_wagner, is_valid_steiner_tree, mst_heuristic


def build_spider() -> Instance:
    """4 terminales + 1 Steiner; óptimo estrella (4), KMB cadena (5.7)."""
    G = nx.Graph()
    for t in ("t1", "t2", "t3", "t4"):
        G.add_edge(t, "s", weight=1.0)
    G.add_edge("t1", "t2", weight=1.9)
    G.add_edge("t2", "t3", weight=1.9)
    G.add_edge("t3", "t4", weight=1.9)
    return Instance(graph=G, terminals=frozenset({"t1", "t2", "t3", "t4"}))


def build_triangle() -> Instance:
    """3 terminales conectados por un punto de Steiner; ambos algoritmos = 3.0."""
    G = nx.Graph()
    for t in ("a", "b", "c"):
        G.add_edge(t, "s", weight=1.0)
    return Instance(graph=G, terminals=frozenset({"a", "b", "c"}))


def build_path(n: int = 5) -> Instance:
    """Camino con n vértices; sólo extremos como terminales."""
    G = nx.path_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return Instance(graph=G, terminals=frozenset({0, n - 1}))


def build_complete_all_terminals(n: int = 4) -> Instance:
    """Grafo completo con todos los vértices como terminales (=> MST)."""
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = float(u + v + 1)
    return Instance(graph=G, terminals=frozenset(G.nodes))


class TestSanity(unittest.TestCase):
    # --- 1. Spider, DP óptimo --------------------------------------------
    def test_spider_dp_cost(self):
        cost, tree = dreyfus_wagner(build_spider())
        self.assertAlmostEqual(cost, 4.0, places=9)
        self.assertTrue(is_valid_steiner_tree(tree, build_spider()))

    # --- 2. Spider, KMB greedy -------------------------------------------
    def test_spider_kmb_cost(self):
        cost, tree = mst_heuristic(build_spider())
        self.assertAlmostEqual(cost, 5.7, places=9)
        self.assertTrue(is_valid_steiner_tree(tree, build_spider()))

    # --- 3. Triángulo simple --------------------------------------------
    def test_triangle_both_match(self):
        inst = build_triangle()
        opt_cost, _ = dreyfus_wagner(inst)
        grd_cost, _ = mst_heuristic(inst)
        self.assertAlmostEqual(opt_cost, 3.0, places=9)
        self.assertAlmostEqual(grd_cost, 3.0, places=9)

    # --- 4. Edge case: 1 solo terminal -----------------------------------
    def test_single_terminal(self):
        G = nx.path_graph(3)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        inst = Instance(graph=G, terminals=frozenset({1}))
        c1, t1 = dreyfus_wagner(inst)
        c2, t2 = mst_heuristic(inst)
        self.assertEqual(c1, 0.0)
        self.assertEqual(c2, 0.0)
        self.assertEqual(t1.number_of_edges(), 0)
        self.assertEqual(t2.number_of_edges(), 0)

    # --- 5. 2 terminales = camino más corto ------------------------------
    def test_two_terminals_shortest_path(self):
        inst = build_path(n=5)
        opt_cost, _ = dreyfus_wagner(inst)
        grd_cost, _ = mst_heuristic(inst)
        self.assertAlmostEqual(opt_cost, 4.0, places=9)
        self.assertAlmostEqual(grd_cost, 4.0, places=9)

    # --- 6. Todos los vértices terminales = MST --------------------------
    def test_all_terminals_equals_mst(self):
        inst = build_complete_all_terminals(n=5)
        mst_cost = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(inst.graph))
        opt_cost, _ = dreyfus_wagner(inst)
        grd_cost, _ = mst_heuristic(inst)
        self.assertAlmostEqual(opt_cost, mst_cost, places=9)
        self.assertAlmostEqual(grd_cost, mst_cost, places=9)

    # --- 7. Instancia rechaza grafo desconectado -------------------------
    def test_disconnected_graph_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_node(2)
        with self.assertRaises(ValueError):
            Instance(graph=G, terminals=frozenset({0, 2}))

    # --- 8. Terminales fuera de V se rechazan ----------------------------
    def test_unknown_terminal_raises(self):
        G = nx.path_graph(3)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        with self.assertRaises(ValueError):
            Instance(graph=G, terminals=frozenset({99}))

    # --- 9. Peso negativo se rechaza -------------------------------------
    def test_negative_weight_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=-1.0)
        with self.assertRaises(ValueError):
            Instance(graph=G, terminals=frozenset({0, 1}))

    # --- 10. KMB devuelve un árbol válido --------------------------------
    def test_kmb_returns_tree(self):
        for builder in (build_spider, build_triangle, build_complete_all_terminals):
            inst = builder()
            _, tree = mst_heuristic(inst)
            self.assertTrue(nx.is_tree(tree) or tree.number_of_nodes() == 1)
            self.assertTrue(inst.terminals.issubset(set(tree.nodes)))

    # --- 11. KMB ≥ DP (DP es óptimo) -------------------------------------
    def test_kmb_at_least_dp(self):
        for builder in (build_spider, build_triangle, build_path):
            inst = builder()
            opt_cost, _ = dreyfus_wagner(inst)
            grd_cost, _ = mst_heuristic(inst)
            self.assertGreaterEqual(grd_cost + 1e-9, opt_cost)
            self.assertFalse(math.isnan(grd_cost))


if __name__ == "__main__":
    unittest.main()
