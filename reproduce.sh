#!/usr/bin/env bash
# reproduce.sh — un solo comando regenera figuras, tablas y el PDF.
#
# Idempotente: si ya hay datos descargados o un CSV reciente, los reusa.
# Para forzar todo de cero, borra `bench/results/` y `docs/figures/`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
RESULTS_DIR="bench/results"
RAW_CSV="${RESULTS_DIR}/raw.csv"
FIG_DIR="docs/figures"
TAB_DIR="docs/tables"

QUICK_FLAG="${QUICK:---quick}"

echo "[1/5] Instalando dependencias de Python ..."
$PY -m pip install --quiet -r requirements.txt

echo "[2/5] Descargando SteinLib serie B (best-effort) ..."
$PY -m bench.fetch_steinlib || echo "  (saltado — sin red o ya presente)"

echo "[3/5] Ejecutando experimentos (${QUICK_FLAG}) ..."
mkdir -p "$RESULTS_DIR" "$FIG_DIR" "$TAB_DIR"
$PY -m bench.run_experiments --output "$RAW_CSV" ${QUICK_FLAG}

echo "[4/5] Generando figuras y tablas LaTeX ..."
$PY -m bench.analyze --input "$RAW_CSV" --figures "$FIG_DIR" --tables "$TAB_DIR"

echo "[5/5] Compilando PDF ..."
if command -v latexmk >/dev/null 2>&1; then
  cd docs && latexmk -pdf -interaction=nonstopmode paper.tex || {
    echo "  (latexmk fallo — figuras/tablas siguen disponibles en docs/)"
    exit 0
  }
  cd "$ROOT"
  echo "Listo. PDF en docs/paper.pdf."
else
  echo "  latexmk no encontrado; PDF no compilado."
  echo "  Figuras: $FIG_DIR/*.png"
  echo "  Tablas:  $TAB_DIR/*.tex"
fi
