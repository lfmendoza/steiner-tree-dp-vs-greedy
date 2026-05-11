"""
Demostración de las dos implementaciones sobre instancias verificables a mano.

Uso:
    python demo.py
"""
from __future__ import annotations

import networkx as nx

from steiner import Instance, dreyfus_wagner, mst_heuristic


def build_spider() -> Instance:
    """Instancia "araña": 4 terminales conectados por un punto de Steiner s.

    Aristas:
      (ti, s) = 1   para i = 1..4
      (t1,t2) = (t2,t3) = (t3,t4) = 1.9

    Árbol óptimo: estrella centrada en s, peso 4.0
    KMB greedy:   prefiere las aristas perimetrales 1.9, peso 5.7
    Cociente esperado: 5.7 / 4.0 = 1.425
    """
    G = nx.Graph()
    for t in ("t1", "t2", "t3", "t4"):
        G.add_edge(t, "s", weight=1.0)
    G.add_edge("t1", "t2", weight=1.9)
    G.add_edge("t2", "t3", weight=1.9)
    G.add_edge("t3", "t4", weight=1.9)
    return Instance(graph=G, terminals=frozenset({"t1", "t2", "t3", "t4"}))


def build_triangle() -> Instance:
    """Instancia "triángulo simple": 3 terminales, 1 Steiner, aristas peso 1.

    Óptimo: estrella en s, peso 3.0
    KMB greedy: también 3.0 (la heurística da el óptimo aquí)
    """
    G = nx.Graph()
    for t in ("a", "b", "c"):
        G.add_edge(t, "s", weight=1.0)
    return Instance(graph=G, terminals=frozenset({"a", "b", "c"}))


def report(name: str, instance: Instance, expected_opt: float, expected_greedy: float):
    opt_cost, opt_tree = dreyfus_wagner(instance)
    grd_cost, grd_tree = mst_heuristic(instance)
    print(f"\n=== {name} ===")
    print(f"  Óptimo (Dreyfus-Wagner): {opt_cost:.4f}  (esperado {expected_opt})")
    print(f"  Greedy (KMB):            {grd_cost:.4f}  (esperado {expected_greedy})")
    print(f"  Cociente greedy/óptimo:  {grd_cost / opt_cost:.4f}")
    print(f"  Aristas óptimas: {sorted(opt_tree.edges(data='weight'))}")
    print(f"  Aristas greedy:  {sorted(grd_tree.edges(data='weight'))}")
    # Sanidades.
    assert abs(opt_cost - expected_opt) < 1e-9, "óptimo no coincide"
    assert abs(grd_cost - expected_greedy) < 1e-9, "greedy no coincide"


def main():
    report("Araña 4 terminales (greedy subóptimo)", build_spider(), 4.0, 5.7)
    report("Triángulo simple (greedy óptimo)", build_triangle(), 3.0, 3.0)
    print("\nTodas las verificaciones pasan.")


if __name__ == "__main__":
    main()
