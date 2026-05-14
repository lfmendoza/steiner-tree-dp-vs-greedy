"""
Tests rápidos para los módulos de visualización.

No validan correctitud pixel-a-pixel, solo que:
  * Las llamadas no lancen excepciones.
  * Los archivos de salida se creen en el destino indicado.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from steiner import dreyfus_wagner, mst_heuristic
from steiner.instances import spider
from steiner.mehlhorn import mehlhorn
from steiner.rsph import rsph


class TestVizSmoke(unittest.TestCase):
    def test_compare_side_by_side(self):
        from viz.draw_tree import compare_side_by_side

        inst = spider(k=4, epsilon=0.05)
        trees = {
            "DP": (dreyfus_wagner(inst)[1], "#1f77b4"),
            "KMB": (mst_heuristic(inst)[1], "#d62728"),
            "Mehlhorn": (mehlhorn(inst)[1], "#2ca02c"),
            "RSPH": (rsph(inst)[1], "#9467bd"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "side.png"
            saved = compare_side_by_side(inst, trees, out, title="smoke")
            self.assertTrue(saved.exists())
            self.assertGreater(saved.stat().st_size, 1000)

    def test_compare_grid(self):
        from viz.draw_tree import compare_grid

        inst = spider(k=4, epsilon=0.05)
        trees = {
            "DP": (dreyfus_wagner(inst)[1], "#1f77b4"),
            "KMB": (mst_heuristic(inst)[1], "#d62728"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "grid.png"
            saved = compare_grid(inst, trees, out, title="smoke", cols=2)
            self.assertTrue(saved.exists())


if __name__ == "__main__":
    unittest.main()
