"""
Tests para los generadores de instancias.

Verifica que cada generador:
  * Produzca un grafo conexo.
  * Respete |T| solicitado y terminales ⊆ V.
  * Sea determinista bajo la misma semilla.
"""
from __future__ import annotations

import unittest

import networkx as nx

from steiner.instances import (
    double_spider,
    euclidean,
    geometric,
    grid_with_shortcut,
    random_er,
    spider,
)


class TestRandomER(unittest.TestCase):
    def test_basic_properties(self):
        inst = random_er(n=20, p=0.5, k=5, seed=0)
        self.assertEqual(inst.n, 20)
        self.assertEqual(inst.k, 5)
        self.assertTrue(nx.is_connected(inst.graph))
        self.assertTrue(inst.terminals.issubset(set(inst.graph.nodes)))

    def test_seed_deterministic(self):
        a = random_er(n=15, p=0.4, k=4, seed=42)
        b = random_er(n=15, p=0.4, k=4, seed=42)
        self.assertEqual(set(a.graph.edges), set(b.graph.edges))
        self.assertEqual(a.terminals, b.terminals)


class TestEuclidean(unittest.TestCase):
    def test_basic_properties(self):
        inst = euclidean(n=12, k=4, seed=1)
        self.assertEqual(inst.n, 12)
        self.assertEqual(inst.k, 4)
        # El grafo completo es conexo por construcción.
        self.assertTrue(nx.is_connected(inst.graph))
        # Las posiciones quedan registradas en los nodos.
        for v in inst.graph.nodes:
            self.assertIn("pos", inst.graph.nodes[v])

    def test_seed_deterministic(self):
        a = euclidean(n=10, k=3, seed=7)
        b = euclidean(n=10, k=3, seed=7)
        self.assertEqual(a.terminals, b.terminals)
        for v in a.graph.nodes:
            self.assertEqual(a.graph.nodes[v]["pos"], b.graph.nodes[v]["pos"])


class TestGeometric(unittest.TestCase):
    def test_basic_properties(self):
        inst = geometric(n=20, r=0.35, k=5, seed=2)
        self.assertEqual(inst.n, 20)
        self.assertEqual(inst.k, 5)
        self.assertTrue(nx.is_connected(inst.graph))


class TestPathological(unittest.TestCase):
    def test_spider_structure(self):
        inst = spider(k=5, epsilon=0.05)
        # k+1 vértices (terminales + Steiner s), 2k-1 aristas.
        self.assertEqual(inst.n, 6)
        self.assertEqual(inst.m, 2 * 5 - 1)
        self.assertEqual(inst.k, 5)
        for t in (f"t{i+1}" for i in range(5)):
            self.assertTrue(inst.graph.has_edge(t, "s"))
            self.assertAlmostEqual(inst.graph["s"][t]["weight"], 1.0)

    def test_double_spider(self):
        inst = double_spider(k1=3, k2=3, epsilon=0.1)
        self.assertEqual(inst.k, 6)
        self.assertTrue(nx.is_connected(inst.graph))
        self.assertTrue(inst.graph.has_edge("sL", "sR"))

    def test_grid_with_shortcut(self):
        inst = grid_with_shortcut(n=4, shortcut_weight=0.5)
        self.assertEqual(inst.k, 4)
        self.assertTrue(inst.graph.has_edge((0, 0), (3, 3)))
        self.assertAlmostEqual(inst.graph[(0, 0)][(3, 3)]["weight"], 0.5)


class TestSteinlibParser(unittest.TestCase):
    """Round-trip sobre un mini ``.stp`` escrito a mano."""

    SAMPLE = """33D32945 STP File, STP Format Version 1.0
SECTION Comment
Name "demo"
END
SECTION Graph
Nodes 5
Edges 5
E 1 2 3.0
E 2 3 1.0
E 3 4 2.0
E 4 5 4.0
E 1 5 7.0
END
SECTION Terminals
Terminals 3
T 1
T 3
T 5
END
EOF
"""

    def test_parse(self):
        from pathlib import Path
        import tempfile

        from steiner.instances.steinlib import parse_stp

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mini.stp"
            p.write_text(self.SAMPLE, encoding="utf-8")
            inst = parse_stp(p)
        self.assertEqual(inst.n, 5)
        self.assertEqual(inst.m, 5)
        self.assertEqual(inst.terminals, frozenset({1, 3, 5}))
        self.assertAlmostEqual(inst.graph[2][3]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
