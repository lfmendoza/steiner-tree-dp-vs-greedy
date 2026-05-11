# Steiner Tree: DP vs Greedy

Implementación y comparación empírica de dos algoritmos para el problema
del **Árbol de Steiner en grafos**:

- **Exacto**: programación dinámica de Dreyfus–Wagner (1971).
  Complejidad temporal `O(3^k · n + 2^k · n² + n³)`, donde `n = |V|` y
  `k = |T|` (número de terminales). Superpolinomial en `k`.
- **Greedy**: heurística por caminos más cortos / MST de Kou–Markowsky–Berman
  (1981). Tiempo polinomial, garantía de `2 · (1 − 1/L)`-aproximación, donde
  `L` es el número de hojas del árbol óptimo.

Proyecto #2 de *Análisis y Diseño de Algoritmos* (UVG, Semestre 1, 2026),
Docente: Tomás Gálvez P.

## Integrantes

- Fernando
- Dilary

## Planteamiento

Dado un grafo no dirigido conexo `G = (V, E)` con pesos no negativos
`w: E → ℝ≥₀` y un conjunto de vértices *terminales* `T ⊆ V`, encontrar
un subárbol de `G` que contenga todos los terminales y cuyo peso total
sea mínimo. Los vértices no terminales que el árbol decide incluir se
llaman *puntos de Steiner*; permitirlos puede reducir estrictamente el
peso total, y eso distingue al problema del árbol generador mínimo.

La versión de decisión es NP-completa (Karp, 1972), de modo que cualquier
algoritmo exacto debe correr en tiempo superpolinomial salvo que P = NP.

## Estructura

```
steiner-tree-dp-vs-greedy/
├── README.md
├── requirements.txt
├── .gitignore
├── steiner/
│   ├── __init__.py
│   ├── graph_utils.py     # Instance, utilidades de grafo
│   ├── dreyfus_wagner.py  # DP exacta
│   └── mst_heuristic.py   # Greedy KMB
├── tests/
│   ├── __init__.py
│   └── test_sanity.py     # casos verificables a mano
├── demo.py                # corrida de demostración
└── bench/                 # análisis empírico (pendiente)
```

## Uso

```bash
pip install -r requirements.txt
python demo.py
python -m unittest discover tests
```

La corrida de `demo.py` debe reportar, en la instancia "spider" de
cuatro terminales, óptimo `4.0` contra greedy `5.7` (cociente `1.425`);
y en el triángulo simple, óptimo y greedy ambos `3.0`.

## Roadmap

- [x] Implementación exacta (Dreyfus–Wagner) con reconstrucción de árbol.
- [x] Implementación greedy (Kou–Markowsky–Berman) con poda de hojas no terminales.
- [x] Pruebas de sanidad en instancias verificables a mano.
- [ ] Generador de instancias aleatorias (Erdős–Rényi ponderado).
- [ ] Script de benchmarking y recolección de tiempos.
- [ ] Diagrama de dispersión y regresión polinomial.
- [ ] Documento PDF con análisis teórico.
