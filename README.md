# Steiner Tree: DP vs Greedy

Comparación empírica entre la programación dinámica exacta de
**Dreyfus–Wagner (1971)** y tres heurísticas con cota de
2-aproximación para el problema del *árbol de Steiner en grafos*:

- **KMB** — Kou–Markowsky–Berman (1981).
- **Mehlhorn** — Mehlhorn (1988), versión $O(m + n \log n)$.
- **RSPH** — Repetitive Shortest Path Heuristic (Takahashi–Matsuyama 1980, Voß 1992).

Proyecto #2 de *Análisis y Diseño de Algoritmos*, UVG, Semestre 1, 2026.
Integrantes: Fernando Mendoza, Dilary Cruz.

### División del trabajo (implementación y pruebas)

Cada bloque funcional se entrega **junto con sus pruebas**: no hay commits
solo de “código sin tests” ni solo de “tests sin código nuevo”. En el
historial de Git los bloques se reparten alternando autores; cada mensaje
de commit detalla ambas partes (implementación + validación).

| Bloque | Responsable principal (autor en Git) | Implementación | Pruebas |
|--------|----------------------------------------|----------------|---------|
| Núcleo `steiner` (DP, KMB, `Instance`) | Fernando | `graph_utils`, `dreyfus_wagner`, `mst_heuristic` | `test_sanity` (11 casos) |
| Mehlhorn y RSPH | Dilary | `mehlhorn`, `rsph` | `test_mehlhorn`, `test_rsph`, `_helpers` |
| Generadores de instancias | Fernando | `steiner/instances/*` | `test_instances`, `test_pathological_bound` |
| Visualización | Dilary | `viz/*` | `test_viz` |
| Benchmarks | Fernando | `bench/*` | `test_bench` |
| Dashboard, reproducibilidad y paper | Dilary | `dashboard`, `reproduce.sh`, `docs/*`, README | `test_dashboard` |

La revisión cruzada (pair review) entre integrantes se hace fuera de Git;
el reparto anterior por “solo tests vs solo código” no aplica.

## Pregunta de investigación

> *KMB tiene garantía teórica de 2-aproximación. Empíricamente, ¿cuándo se acerca a la cota y cuándo se queda lejos? ¿Qué estructuras de instancia provocan que falle de forma severa?*

El proyecto responde por tres ejes:

1. **Tiempo.** Explosión exponencial de DP en $k$ vs polinomialidad de greedy en $n$.
2. **Calidad.** Distribución del cociente *greedy/óptimo* a lo largo de cinco familias.
3. **Estructura.** Construcción patológica (`spider`) que alcanza asintóticamente la cota $2(1 - 1/k)$.

## Estructura del repositorio

```
steiner-tree-dp-vs-greedy/
├── steiner/
│   ├── graph_utils.py        Instance + APSP + utilidades
│   ├── dreyfus_wagner.py     DP exacta (con reconstrucción)
│   ├── mst_heuristic.py      KMB
│   ├── mehlhorn.py           Mehlhorn 1988
│   ├── rsph.py               Repetitive Shortest Path Heuristic
│   └── instances/            5 generadores (ER, euclidean, geometric, spider, SteinLib)
├── viz/                      draw_tree, animate (RSPH gif), heatmap
├── bench/                    timing, quality, regression, run_experiments, analyze, fetch_steinlib
├── dashboard/app.py          Streamlit interactivo
├── tests/                    33 tests unitarios
├── docs/
│   ├── paper.tex             documento estilo paper (IEEE-like)
│   ├── references.bib        13 referencias canónicas
│   ├── figures/              generadas por bench/analyze
│   ├── tables/               generadas por bench/analyze
│   └── steinlib_data/        18 instancias B (descargadas)
├── demo.py                   corrida de demostración
├── reproduce.sh              regenera todo de un comando
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
python -m unittest discover tests -v
```

## Reproducir todo

```bash
chmod +x reproduce.sh
./reproduce.sh           # smoke run (--quick)
QUICK="" ./reproduce.sh  # sweep completo (~horas)
```

Esto:

1. Instala las dependencias.
2. Descarga la serie B de SteinLib en `docs/steinlib_data/`.
3. Corre los experimentos sobre las 5 familias × 4 algoritmos.
4. Genera figuras y tablas LaTeX en `docs/figures/` y `docs/tables/`.
5. Compila `docs/paper.pdf` con `latexmk` si está disponible.

## Componentes individuales

```bash
python -m bench.fetch_steinlib                              # descarga SteinLib B
python -m bench.run_experiments --output bench/results/raw.csv --quick
python -m bench.analyze --input bench/results/raw.csv --figures docs/figures
```

Dashboard (interfaz web): desde la **raíz** del repositorio ejecuta `streamlit run dashboard/app.py`. El script añade la raíz a `sys.path` para que Python encuentre los paquetes `steiner` y `viz` (Streamlit suele poner primero la carpeta `dashboard/` en el path). Alternativa: instalar el proyecto con `pip install -e .` y usar el mismo comando.

## Pruebas

```bash
python -m unittest discover tests -v
```

Cobertura:

- 11 tests de sanidad sobre la spider de 4 terminales y el triángulo (Phase 0).
- 50 instancias seeded × 2 heurísticas comparadas contra DP, verificando la cota $2(1 - 1/L)$ (Phases 1–2).
- Test asintótico de la construcción tight: `spider(k=40, eps=0.01)` debe alcanzar ratio $\ge 1.9$.
- Tests de tiempos, regresión polinomial y bootstrap CI.
- Tests de los generadores de instancias (determinismo bajo semilla).
- Smoke tests de las visualizaciones.
- Tests del layout de imports del dashboard (`test_dashboard`).

## Resultados destacados

Después de un `./reproduce.sh`:

- **Tiempo.** La pendiente del ajuste $\log t_{\mathrm{DP}} = a + b\cdot k$ se aproxima a $\ln 3 \approx 1.10$, consistente con la complejidad teórica $O(3^k \cdot n)$.
- **Calidad.** En las cinco familias el cociente mediano de los tres greedy queda muy por debajo de 2 (típicamente $\le 1.10$), pero la familia *spider* lo lleva al límite.
- **Tight bound.** Con `spider(k=40, epsilon=0.01)` el cociente KMB es $\approx 1.94$, validando empíricamente que la cota $2(1 - 1/L)$ es esencialmente apretada.
- **SteinLib.** Los tres greedy resuelven `b01.stp`–`b18.stp` con cocientes muy cercanos a 1 frente a DP exacta donde es viable.

Las figuras correspondientes se generan en `docs/figures/`.

## Referencias

La bibliografía completa está en [`docs/references.bib`](docs/references.bib).
Las cinco fundacionales:

- S. E. Dreyfus, R. A. Wagner. *The Steiner problem in graphs.* Networks 1 (1971), 195–207.
- L. Kou, G. Markowsky, L. Berman. *A fast algorithm for Steiner trees.* Acta Informatica 15 (1981), 141–145.
- K. Mehlhorn. *A faster approximation algorithm for the Steiner problem in graphs.* IPL 27 (1988), 125–128.
- T. Koch, A. Martin, S. Voß. *SteinLib: An updated library on Steiner tree problems in graphs.* ZIB Report 00-37 (2001).
- R. M. Karp. *Reducibility among combinatorial problems.* In *Complexity of Computer Computations* (1972), 85–103.
