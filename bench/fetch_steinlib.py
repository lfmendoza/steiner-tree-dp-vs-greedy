"""
Descarga el subconjunto B de SteinLib (Koch–Martin–Voß, 2001).

SteinLib publica los benchmarks de Steiner Tree en
``https://steinlib.zib.de/``. La colección B (derivada de OR-Library)
tiene ``|V| <= 100`` y ``|T| <= 17``, lo cual encaja con la cota
práctica de nuestra DP exacta.

El script intenta una URL primaria; si falla, prueba un mirror
alternativo o instruye al usuario a colocar manualmente los archivos
en ``docs/steinlib_data/``.
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


PRIMARY_URL = "https://steinlib.zib.de/download/B.tgz"
ALT_URLS = [
    "http://steinlib.zib.de/download/B.tgz",
]
USER_AGENT = "steiner-bench/0.1 (academic; +https://github.com/lfmendoza/steiner-tree-dp-vs-greedy)"


def _try_download(url: str, timeout: float = 30.0) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [warn] {url} -> {exc}", file=sys.stderr)
        return None


def fetch_b_series(out_dir: str | Path = "docs/steinlib_data") -> int:
    """Intenta bajar ``B.tgz`` y extraer los ``.stp`` en ``out_dir``.

    Returns
    -------
    int
        Cantidad de archivos ``.stp`` que terminaron en disco.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.stp"))
    if existing:
        print(f"[fetch_steinlib] {len(existing)} .stp ya presentes en {out_dir}; saltando descarga.")
        return len(existing)

    for url in [PRIMARY_URL, *ALT_URLS]:
        print(f"[fetch_steinlib] intentando {url} ...")
        data = _try_download(url)
        if data is None:
            continue
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                stp_count = 0
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = Path(member.name).name
                    if not name.lower().endswith(".stp"):
                        continue
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    (out_dir / name).write_bytes(fh.read())
                    stp_count += 1
                print(f"[fetch_steinlib] extraídos {stp_count} archivos en {out_dir}.")
                return stp_count
        except tarfile.ReadError as exc:
            print(f"  [warn] no es un tarball válido: {exc}", file=sys.stderr)
            continue

    print(
        "[fetch_steinlib] No se pudo descargar SteinLib automáticamente.\n"
        "  Descárgalo manualmente desde https://steinlib.zib.de/testset.php "
        f"y copia los archivos b*.stp al directorio: {out_dir}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Descarga la serie B de SteinLib.")
    p.add_argument("--out-dir", type=Path, default=Path("docs/steinlib_data"))
    args = p.parse_args(argv or sys.argv[1:])
    n = fetch_b_series(args.out_dir)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
