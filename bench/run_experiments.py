"""
Orquestador del experimento empírico.

Recorre las cinco familias de instancias y los cuatro algoritmos
generando un CSV con columnas

    instance_family, n, m, k, algo, seed, time_s_median, time_s_iqr,
    cost, ratio_vs_dp, timed_out

``ratio_vs_dp`` queda vacía si Dreyfus–Wagner no completó dentro del
timeout (instancias grandes en n/k).

Uso
---
    python -m bench.run_experiments --output bench/results/raw.csv
    python -m bench.run_experiments --output bench/results/raw.csv --quick
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterator

from tqdm import tqdm

from steiner import Instance, dreyfus_wagner, mst_heuristic
from steiner.instances import (
    euclidean,
    geometric,
    random_er,
    spider,
)
from steiner.instances.steinlib import list_steinlib, parse_stp
from steiner.mehlhorn import mehlhorn
from steiner.rsph import rsph

from .timing import time_call


# ---------------------------------------------------------------------------
# Definición declarativa de los sweeps.
# ---------------------------------------------------------------------------


@dataclass
class SweepCase:
    """Una corrida individual del experimento."""

    family: str
    instance: Instance
    n: int
    m: int
    k: int
    seed: int


def sweeps_full() -> Iterator[SweepCase]:
    """Sweep completo (~horas en hardware modesto)."""
    yield from _sweep_er("er_sparse", p=0.3, ns=(20, 50, 100, 200), ks=(5, 10, 20), seeds=range(30))
    yield from _sweep_er("er_dense", p=0.7, ns=(20, 50, 100), ks=(5, 10, 20), seeds=range(30))
    yield from _sweep_euclidean(ns=(20, 50, 100, 200), ks=(5, 10, 20), seeds=range(30))
    yield from _sweep_geometric(ns=(20, 50, 100, 200), r=0.35, ks=(5, 10), seeds=range(30))
    yield from _sweep_spider(ks=range(3, 21), epsilons=(0.5, 0.1, 0.01))
    yield from _sweep_steinlib()


def sweeps_quick() -> Iterator[SweepCase]:
    """Sweep reducido para corridas rápidas (CI/smoke)."""
    yield from _sweep_er("er_sparse", p=0.3, ns=(12, 18, 25), ks=(3, 5, 7), seeds=range(3))
    yield from _sweep_er("er_dense", p=0.7, ns=(10, 15, 20), ks=(3, 5, 7), seeds=range(3))
    yield from _sweep_euclidean(ns=(10, 15, 25), ks=(3, 5, 7), seeds=range(3))
    yield from _sweep_geometric(ns=(15, 25), r=0.4, ks=(3, 5), seeds=range(3))
    yield from _sweep_spider(ks=(3, 5, 8, 10, 12, 15, 20, 30, 40), epsilons=(0.1, 0.01))
    yield from _sweep_steinlib(max_n=50, max_k=10)


def _sweep_er(
    family: str, *, p: float, ns: tuple, ks: tuple, seeds
) -> Iterator[SweepCase]:
    for n, k, s in product(ns, ks, seeds):
        if k > n:
            continue
        inst = random_er(n=n, p=p, k=k, seed=s)
        yield SweepCase(family=family, instance=inst, n=inst.n, m=inst.m, k=inst.k, seed=s)


def _sweep_euclidean(*, ns: tuple, ks: tuple, seeds) -> Iterator[SweepCase]:
    for n, k, s in product(ns, ks, seeds):
        if k > n:
            continue
        inst = euclidean(n=n, k=k, seed=s)
        yield SweepCase(family="euclidean", instance=inst, n=inst.n, m=inst.m, k=inst.k, seed=s)


def _sweep_geometric(*, ns: tuple, r: float, ks: tuple, seeds) -> Iterator[SweepCase]:
    for n, k, s in product(ns, ks, seeds):
        if k > n:
            continue
        inst = geometric(n=n, r=r, k=k, seed=s)
        yield SweepCase(family="geometric", instance=inst, n=inst.n, m=inst.m, k=inst.k, seed=s)


def _sweep_spider(*, ks, epsilons: tuple) -> Iterator[SweepCase]:
    for k, eps in product(list(ks), epsilons):
        if k < 2:
            continue
        inst = spider(k=k, epsilon=eps)
        seed = int(round(eps * 1000))
        yield SweepCase(family=f"spider_eps{eps}", instance=inst, n=inst.n, m=inst.m, k=inst.k, seed=seed)


def _sweep_steinlib(
    root: str = "docs/steinlib_data", max_n: int = 200, max_k: int = 17
) -> Iterator[SweepCase]:
    for path in list_steinlib(root):
        try:
            inst = parse_stp(path)
        except Exception:  # noqa: BLE001
            continue
        if inst.n > max_n or inst.k > max_k:
            continue
        yield SweepCase(
            family="steinlib_B", instance=inst,
            n=inst.n, m=inst.m, k=inst.k, seed=hash(path.name) & 0xFFFF,
        )


# ---------------------------------------------------------------------------
# Ejecución y serialización.
# ---------------------------------------------------------------------------


ALGOS: dict[str, Callable] = {
    "dreyfus_wagner": dreyfus_wagner,
    "mst_heuristic": mst_heuristic,
    "mehlhorn": mehlhorn,
    "rsph": rsph,
}


CSV_COLUMNS = [
    "instance_family", "n", "m", "k", "algo", "seed",
    "time_s_median", "time_s_iqr", "cost", "ratio_vs_dp", "timed_out",
]


def _run_one(
    algo_name: str, algo_fn: Callable, case: SweepCase,
    *, warmup: int, repeats: int, timeout_s: float,
) -> dict:
    """Mide un par (algoritmo, instancia) y devuelve la fila CSV (sin ratio)."""
    res = time_call(algo_fn, case.instance, warmup=warmup, repeats=repeats, timeout_s=timeout_s)
    if res.timed_out or res.last_value is None:
        cost = float("nan")
    else:
        cost, _ = res.last_value
    return {
        "instance_family": case.family,
        "n": case.n,
        "m": case.m,
        "k": case.k,
        "algo": algo_name,
        "seed": case.seed,
        "time_s_median": res.median_s,
        "time_s_iqr": res.iqr_s,
        "cost": cost,
        "ratio_vs_dp": None,  # rellenado abajo
        "timed_out": res.timed_out,
    }


def run_experiments(
    output: Path,
    *,
    quick: bool = False,
    families: set[str] | None = None,
    warmup: int = 1,
    repeats: int = 3,
    dp_timeout_s: float = 30.0,
    greedy_timeout_s: float = 30.0,
    skip_dp_above_k: int = 14,
) -> int:
    """Corre el experimento y escribe el CSV.

    Para cada instancia ejecuta DP (si ``k <= skip_dp_above_k``) y los
    tres greedies. ``ratio_vs_dp`` se rellena cuando DP tuvo éxito.

    Returns
    -------
    n_rows : int
    """
    sweep = sweeps_quick() if quick else sweeps_full()
    cases = list(sweep)
    if families is not None:
        cases = [c for c in cases if c.family in families or c.family.startswith("spider_eps")
                 and "spider" in families]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    pbar = tqdm(cases, desc="bench", unit="inst")
    for case in pbar:
        pbar.set_postfix(family=case.family, n=case.n, k=case.k)
        # 1) DP (si es factible).
        dp_cost = float("nan")
        if case.k <= skip_dp_above_k:
            dp_row = _run_one(
                "dreyfus_wagner", dreyfus_wagner, case,
                warmup=warmup, repeats=repeats, timeout_s=dp_timeout_s,
            )
            rows.append(dp_row)
            if not dp_row["timed_out"]:
                dp_cost = dp_row["cost"]
        # 2) Greedies.
        for name in ("mst_heuristic", "mehlhorn", "rsph"):
            row = _run_one(
                name, ALGOS[name], case,
                warmup=warmup, repeats=repeats, timeout_s=greedy_timeout_s,
            )
            # Para la spider tenemos óptimo analítico = k (la estrella).
            if case.family.startswith("spider"):
                analytic_opt = float(case.k)
            else:
                analytic_opt = dp_cost
            if analytic_opt > 0 and not row["timed_out"]:
                row["ratio_vs_dp"] = row["cost"] / analytic_opt
            rows.append(row)

    # Rellenar ratio_vs_dp en filas de DP (siempre 1.0 cuando hay valor).
    for row in rows:
        if row["algo"] == "dreyfus_wagner" and not row["timed_out"]:
            row["ratio_vs_dp"] = 1.0

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bench Steiner DP vs greedy")
    p.add_argument(
        "--output", type=Path, default=Path("bench/results/raw.csv"),
        help="Ruta del CSV de salida.",
    )
    p.add_argument("--quick", action="store_true", help="Sweep reducido para smoke runs.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dp-timeout", type=float, default=30.0)
    p.add_argument("--greedy-timeout", type=float, default=30.0)
    p.add_argument("--skip-dp-above-k", type=int, default=14)
    p.add_argument(
        "--families", type=str, default=None,
        help="Subconjunto a correr, separado por comas (ej.: 'spider,euclidean').",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    fams = set(args.families.split(",")) if args.families else None
    t0 = time.perf_counter()
    n_rows = run_experiments(
        output=args.output,
        quick=args.quick,
        families=fams,
        warmup=args.warmup,
        repeats=args.repeats,
        dp_timeout_s=args.dp_timeout,
        greedy_timeout_s=args.greedy_timeout,
        skip_dp_above_k=args.skip_dp_above_k,
    )
    print(
        f"[bench] {n_rows} filas escritas en {args.output} "
        f"en {time.perf_counter() - t0:.1f}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
