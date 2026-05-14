"""
Animación paso-a-paso de la heurística RSPH.

Usa el generador :func:`steiner.rsph.rsph_steps` para producir un fotograma
por terminal añadido y los guarda como ``.gif`` (Pillow) o ``.mp4`` (ffmpeg).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from steiner import Instance
from steiner.rsph import rsph_steps

from .draw_tree import compute_layout, draw_instance, overlay_tree


def animate_rsph(
    instance: Instance,
    out_path: str | Path = "rsph.gif",
    *,
    fps: int = 1,
    color: str = "#d7263d",
    title_prefix: str = "RSPH",
) -> Path:
    """Genera un ``.gif`` mostrando el árbol parcial tras cada terminal.

    Returns the saved path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = list(rsph_steps(instance))
    layout = compute_layout(instance.graph)

    fig, ax = plt.subplots(figsize=(6, 6))

    def render(frame_idx: int) -> None:
        ax.clear()
        step, target, tree = frames[frame_idx]
        draw_instance(ax, instance, layout, show_weights=False)
        overlay_tree(ax, tree, layout, color=color, label="árbol parcial")
        cost = sum(d.get("weight", 0.0) for _, _, d in tree.edges(data=True))
        if target is None:
            ax.set_title(f"{title_prefix} — paso {step} (poda final), cost = {cost:.4f}")
        elif step == 0:
            ax.set_title(f"{title_prefix} — terminal inicial {target}, cost = {cost:.4f}")
        else:
            ax.set_title(f"{title_prefix} — paso {step}: añadido {target}, cost = {cost:.4f}")
        ax.axis("off")

    anim = FuncAnimation(fig, render, frames=len(frames), interval=int(1000 / fps))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
