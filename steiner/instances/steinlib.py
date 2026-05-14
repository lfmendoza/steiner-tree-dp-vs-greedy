"""
Parser del formato SteinLib (``.stp``) para Steiner Tree en grafos.

SteinLib (Koch–Martin–Voß, 2001) es el repositorio canónico de
instancias benchmark; la serie B (OR Library) tiene tamaños chicos
(|V| <= 100, |T| <= 17) en los que Dreyfus–Wagner aún produce un
óptimo en tiempo razonable.

Formato resumido
----------------
    33D32945 STP File, STP Format Version 1.0
    SECTION Comment
        Name "..."
    END
    SECTION Graph
        Nodes 50
        Edges 63
        E 1 2 1
        ...
    END
    SECTION Terminals
        Terminals 9
        T 4
        T 7
        ...
    END
    EOF

El parser tolera comentarios (``#`` o líneas vacías), respeta el orden
de secciones del estándar y devuelve una :class:`Instance` con los
nodos etiquetados como enteros 1..N (consistente con el formato).

Referencia
----------
T. Koch, A. Martin, S. Voß. "SteinLib: An updated library on Steiner
tree problems in graphs." Technical Report ZIB 00-37 (2001).
"""
from __future__ import annotations

import re
from pathlib import Path

import networkx as nx

from ..graph_utils import Instance


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_stp(path: str | Path) -> Instance:
    """Lee un archivo ``.stp`` y devuelve la instancia correspondiente.

    Parameters
    ----------
    path : str or pathlib.Path
        Ruta al archivo ``.stp``.

    Returns
    -------
    Instance

    Raises
    ------
    ValueError
        Si faltan secciones obligatorias o el formato no cuadra.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    G = nx.Graph()
    terminals: set = set()
    section: str | None = None
    in_terminals_section = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("SECTION"):
            section = upper.split(maxsplit=1)[1] if len(upper.split()) > 1 else None
            in_terminals_section = section == "TERMINALS"
            continue
        if upper == "END":
            section = None
            in_terminals_section = False
            continue
        if upper == "EOF":
            break

        if section == "GRAPH":
            tok = line.split()
            head = tok[0].upper()
            if head == "NODES":
                count = int(tok[1])
                G.add_nodes_from(range(1, count + 1))
            elif head in ("EDGES", "ARCS"):
                # Conteo declarado; lo ignoramos y confiamos en las líneas E.
                continue
            elif head in ("E", "A"):
                u = int(tok[1])
                v = int(tok[2])
                w = float(tok[3])
                G.add_edge(u, v, weight=w)
        elif in_terminals_section:
            tok = line.split()
            head = tok[0].upper()
            if head == "TERMINALS":
                continue
            if head == "T":
                terminals.add(int(tok[1]))

    if G.number_of_nodes() == 0:
        raise ValueError(f"{p}: SECTION Graph sin nodos.")
    if not terminals:
        raise ValueError(f"{p}: SECTION Terminals vacía o ausente.")

    # SteinLib puede declarar más nodos que los efectivamente conectados;
    # el constructor de Instance va a exigir conexidad, lo cual es válido
    # para las instancias de la serie B.
    if not nx.is_connected(G):
        # Reintentar quitando nodos aislados (suelen ser anomalías de
        # archivos con `Nodes N` declarado pero E sólo en N' < N).
        isolated = [v for v in G.nodes if G.degree(v) == 0 and v not in terminals]
        G.remove_nodes_from(isolated)
        if not nx.is_connected(G):
            raise ValueError(f"{p}: el grafo del .stp no es conexo.")

    return Instance(graph=G, terminals=frozenset(terminals))


def load_steinlib_B(
    index: int, root: str | Path = "docs/steinlib_data"
) -> Instance:
    """Carga la instancia ``b{index:02d}.stp`` desde el directorio cache.

    Parameters
    ----------
    index : int
        Número de la instancia (1..18 en la serie B).
    root : path-like
        Directorio donde residen los archivos descargados.
    """
    root = Path(root)
    name = f"b{index:02d}.stp"
    return parse_stp(root / name)


def list_steinlib(root: str | Path = "docs/steinlib_data") -> list[Path]:
    """Enumera todos los ``.stp`` disponibles en el directorio cache."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.stp"))
