"""
Medición robusta de tiempos para los algoritmos de Steiner.

API
---
:func:`time_call`
    Mide ``fn(instance)`` con warmups y repeticiones, devolviendo
    estadísticas resistentes a outliers (mediana, IQR, mín, máx).

:func:`run_with_timeout`
    Ejecuta una función con corte de tiempo (timeout) por hilo;
    crítico para DP exacta a ``k`` grande.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TimingResult:
    """Resumen de una medición."""

    median_s: float
    iqr_s: float
    min_s: float
    max_s: float
    n_repeats: int
    timed_out: bool = False
    last_value: Any = None

    def to_dict(self) -> dict:
        return {
            "time_s_median": self.median_s,
            "time_s_iqr": self.iqr_s,
            "time_s_min": self.min_s,
            "time_s_max": self.max_s,
            "n_repeats": self.n_repeats,
            "timed_out": self.timed_out,
        }


def time_call(
    fn: Callable,
    instance,
    *,
    warmup: int = 3,
    repeats: int = 10,
    timeout_s: float | None = None,
) -> TimingResult:
    """Mide ``fn(instance)`` con warmup y múltiples repeticiones.

    Si ``timeout_s`` es positivo y una sola llamada lo excede, devuelve
    ``timed_out=True`` con un único valor (el outlier que disparó el
    corte) — útil para no atascar el bench en instancias inviables.

    Parameters
    ----------
    fn : callable
        Algoritmo bajo prueba, firma ``fn(instance) -> Any``.
    instance : Instance
        Instancia a procesar.
    warmup : int
        Llamadas previas que no se miden (calientan caches).
    repeats : int
        Número de mediciones reportadas.
    timeout_s : float, optional
        Corte por llamada en segundos.

    Returns
    -------
    TimingResult
    """
    last_value = None

    for _ in range(max(0, warmup)):
        ok, value = _run_one(fn, instance, timeout_s)
        if not ok:
            return TimingResult(
                median_s=float("inf"), iqr_s=0.0, min_s=float("inf"),
                max_s=float("inf"), n_repeats=0, timed_out=True, last_value=None,
            )
        last_value = value

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ok, value = _run_one(fn, instance, timeout_s)
        t1 = time.perf_counter()
        if not ok:
            return TimingResult(
                median_s=float("inf"), iqr_s=0.0, min_s=float("inf"),
                max_s=float("inf"), n_repeats=len(samples), timed_out=True,
                last_value=last_value,
            )
        samples.append(t1 - t0)
        last_value = value

    median = statistics.median(samples)
    if len(samples) >= 4:
        q1, q3 = statistics.quantiles(samples, n=4)[0], statistics.quantiles(samples, n=4)[2]
        iqr = q3 - q1
    else:
        iqr = 0.0
    return TimingResult(
        median_s=median, iqr_s=iqr, min_s=min(samples), max_s=max(samples),
        n_repeats=len(samples), timed_out=False, last_value=last_value,
    )


def _run_one(fn: Callable, instance, timeout_s: float | None) -> tuple[bool, Any]:
    """Una llamada a ``fn`` con timeout opcional. Devuelve ``(ok, value)``."""
    if timeout_s is None or timeout_s <= 0:
        return True, fn(instance)
    return run_with_timeout(fn, (instance,), timeout_s)


def run_with_timeout(
    fn: Callable, args: tuple = (), timeout_s: float = 60.0
) -> tuple[bool, Any]:
    """Ejecuta ``fn(*args)`` en un hilo con corte ``timeout_s``.

    Notas
    -----
    Si la función no respeta el corte el hilo permanece vivo en segundo
    plano hasta terminar (no podemos matarlo de forma segura desde
    Python puro). Esto es aceptable para nuestro bench: el thread daemon
    morirá al terminar el proceso.

    Returns
    -------
    (ok, value) :
        ``ok=False`` si la llamada superó el timeout.
    """
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn(*args)
            result["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            result["exc"] = exc
            result["ok"] = False

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return False, None
    if result.get("ok"):
        return True, result["value"]
    if "exc" in result:
        raise result["exc"]
    return False, None
