"""Plotly-Abbildungen (alle Achsen fest, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go

import hdb_constants as C


def _fixed(fig, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    return fig


def build_live_chart(runs, optimum):
    """Regret (bester Wert nach t Auswertungen minus Minimum, logarithmisch) je Verfahren des Live-Laufs."""
    fig = go.Figure()
    for m, run in runs.items():
        y = [max(v - optimum, 1e-3) for v in run.best_so_far]
        fig.add_trace(go.Scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines", name=C.METHOD_LABELS[m], line=dict(color=C.METHOD_COLORS[m], width=2.5 if m != "random" else 1.8),
                                 line_shape="hv"))
    fig.update_xaxes(title_text="Auswertungen")
    fig.update_yaxes(title_text="Regret (log)", type="log")
    return _fixed(fig, height=340)


def build_lengthscale_chart(ls, active, d):
    """Gelernte Längenmaße je Eingabe (logarithmisch); wirksame Eingaben grün, unwirksame rot. Große Längenmaße heißen: das Modell hält die Eingabe für unwichtig."""
    colors = [C.COLOR_OBS if i < active else C.COLOR_BAD for i in range(d)]
    fig = go.Figure(go.Bar(x=[str(i + 1) for i in range(d)], y=ls, marker_color=colors, hovertemplate="Eingabe %{x}<br>Längenmaß %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Eingabe")
    fig.update_yaxes(title_text="Gelerntes Längenmaß (log)", type="log")
    return _fixed(fig, height=300)


def build_study_chart(curves, methods=None):
    """Median und mittlere Hälfte der 10 festen Läufe über das Budget (Regret, logarithmisch)."""
    fig = go.Figure()
    for m in (methods or C.METHODS):
        if m not in curves:
            continue
        c = curves[m]
        n = len(c["median"])
        ts = list(range(1, n + 1))
        col = C.METHOD_COLORS[m]
        lo, hi = np.maximum(c["q25"], 1e-3), np.maximum(c["q75"], 1e-3)
        fig.add_trace(go.Scatter(x=ts + ts[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=col, opacity=0.1, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ts, y=np.maximum(c["median"], 1e-3), mode="lines", name=C.METHOD_LABELS[m], line=dict(color=col, width=2.5 if m != "random" else 1.8)))
    fig.update_xaxes(title_text="Auswertungen")
    fig.update_yaxes(title_text="Regret, Median und 25 bis 75 % (log)", type="log")
    return _fixed(fig, height=400)


def build_dimension_chart(summary_by_dim, checkpoint, methods):
    """Median des Regrets bei `checkpoint` Auswertungen je Dimension und Verfahren (Linien, logarithmisch)."""
    fig = go.Figure()
    dims = sorted(summary_by_dim)
    for m in methods:
        ys = [max(summary_by_dim[d][m][checkpoint]["median"], 1e-3) for d in dims]
        fig.add_trace(go.Scatter(x=[str(d) for d in dims], y=ys, mode="lines+markers", name=C.METHOD_LABELS[m], line=dict(color=C.METHOD_COLORS[m], width=2.5 if m != "random" else 1.8)))
    fig.update_xaxes(title_text="Zahl der Eingaben")
    fig.update_yaxes(title_text="Regret nach %d Auswertungen, Median (log)" % checkpoint, type="log")
    return _fixed(fig, height=360)
