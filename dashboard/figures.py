"""Dendrograma interactivo del HRP.

``plotly.figure_factory.create_dendrogram`` se evaluó y se descartó: sus trazas no exponen la relación
nodo → hojas (necesaria para que el hover de un enlace resalte todo su subárbol) y solo colorean por
``color_threshold``. Se usa la misma geometría base de scipy: ``dendrogram(Z, no_plot=True,
link_color_func=str)`` devuelve en ``color_list`` el id de nodo de cada enlace **en el mismo orden** que
``icoord``/``dcoord`` (scipy anexa los tres en la misma recursión), y ``to_tree(Z, rd=True)`` da las hojas
de cada nodo.

Contrato de trazas (lo consume el callback ``sync`` de ``app.py``):
* una traza ``mode="lines"`` por enlace, ``customdata[i] = [fund_id, ...]`` (subárbol) para todos sus puntos;
* última traza ``name="leaves"`` con un marcador por hoja, ``customdata[i] = [fund_id]``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, to_tree

LEAF_STEP = 10.0                      # scipy sitúa las hojas en 5, 15, 25, …
POINTS_PER_SEGMENT = 8                # densificación de la U para que el hover responda en todo el trazo
LINK_WIDTH, LINK_WIDTH_HI = 2.0, 5.0
MARKER_SIZE, MARKER_SIZE_HI = 9, 15
MIXED_COLOR = "#9aa0a6"               # enlace que une clusters distintos
CLUSTER_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")


@dataclass(frozen=True)
class Link:
    node_id: int                        # id de nodo scipy (≥ n)
    height: float                       # distancia de fusión
    pos: tuple[float, float, float, float]      # icoord: 4 posiciones sobre el eje de hojas
    heights: tuple[float, float, float, float]  # dcoord: 4 alturas (0/h_izq, h, h, h_der/0)
    leaves: tuple[int, ...]             # posiciones de columna bajo el nodo


def dendrogram_geometry(z: np.ndarray) -> tuple[list[int], list[Link]]:
    """(orden de hojas, enlaces) a partir de la matriz de linkage."""
    z = np.asarray(z, dtype=float)
    d = dendrogram(z, no_plot=True, link_color_func=str)
    _, nodes = to_tree(z, rd=True)
    links = [
        Link(node_id=int(k), height=float(dc[1]), pos=tuple(float(v) for v in ic), heights=tuple(float(v) for v in dc),
             leaves=tuple(int(i) for i in nodes[int(k)].pre_order()))
        for ic, dc, k in zip(d["icoord"], d["dcoord"], d["color_list"])
    ]
    return [int(i) for i in d["leaves"]], links


def _densify(xs: Sequence[float], ys: Sequence[float], per_segment: int = POINTS_PER_SEGMENT) -> tuple[list[float], list[float]]:
    px: list[float] = []
    py: list[float] = []
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        px.extend((x0 + (x1 - x0) * t).tolist())
        py.extend((y0 + (y1 - y0) * t).tolist())
    px.append(float(xs[-1]))
    py.append(float(ys[-1]))
    return px, py


def build_dendrogram(z: np.ndarray, fund_ids: Sequence[str], labels: Mapping[str, str], cluster_labels: Sequence[int],
                     weights: Mapping[str, float], highlight: Iterable[str] = (), title: str | None = None) -> go.Figure:
    """Dendrograma horizontal (hojas en y, distancia en x) coloreado por cluster, con los fondos de ``highlight``
    (y los enlaces cuyo subárbol está íntegramente en ``highlight``) resaltados."""
    fund_ids = [str(f) for f in fund_ids]
    n = len(fund_ids)
    hi = set(highlight)
    leaves, links = dendrogram_geometry(z)
    pos = {leaf: LEAF_STEP * i + LEAF_STEP / 2 for i, leaf in enumerate(leaves)}
    palette = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(sorted(set(int(c) for c in cluster_labels)))}
    name = lambda f: labels.get(f, f)  # noqa: E731

    fig = go.Figure()
    for lk in links:
        ids = [fund_ids[i] for i in lk.leaves]
        clusters = {int(cluster_labels[i]) for i in lk.leaves}
        color = palette[next(iter(clusters))] if len(clusters) == 1 else MIXED_COLOR
        is_hi = bool(hi) and set(ids) <= hi
        leaf_pos, heights = _densify(lk.pos, lk.heights)
        shown = ", ".join(name(f) for f in ids[:6]) + (" …" if len(ids) > 6 else "")
        fig.add_trace(go.Scatter(
            x=heights, y=leaf_pos, mode="lines", name=f"link-{lk.node_id}", showlegend=False,
            line=dict(color=color, width=LINK_WIDTH_HI if is_hi else LINK_WIDTH),
            customdata=[ids] * len(heights),
            hovertemplate=f"<b>{len(ids)} fondos</b> · distancia {lk.height:.3f}<br>{shown}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[0.0] * n, y=[pos[i] for i in leaves], mode="markers", name="leaves", showlegend=False,
        marker=dict(size=[MARKER_SIZE_HI if fund_ids[i] in hi else MARKER_SIZE for i in leaves],
                    color=[palette[int(cluster_labels[i])] for i in leaves], line=dict(width=1, color="white")),
        customdata=[[fund_ids[i]] for i in leaves],
        text=[f"<b>{name(fund_ids[i])}</b><br>cluster {int(cluster_labels[i])} · peso {weights.get(fund_ids[i], 0.0):.2%}"
              for i in leaves],
        hovertemplate="%{text}<extra></extra>",
    ))

    max_h = max((lk.height for lk in links), default=1.0)
    fig.update_layout(
        title=title, height=max(420, int(28 * n + 120)), margin=dict(l=10, r=20, t=50 if title else 10, b=45),
        hovermode="closest", hoverdistance=25, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="distancia √((1 − ρ) / 2)", range=[-0.03 * max_h, 1.05 * max_h], zeroline=False,
                   gridcolor="#eeeeee"),
        yaxis=dict(tickvals=[pos[i] for i in leaves], ticktext=[fund_ids[i] for i in leaves],
                   range=[0, LEAF_STEP * n], showgrid=False, zeroline=False, tickfont=dict(size=11)),
    )
    return fig
