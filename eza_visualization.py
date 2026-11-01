"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import eza_constants as C
from eza_distributions import get_dist


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _x_limit(kind: str) -> float:
    """Anzeigebereich der Werte: bei der Exponentialverteilung bis zum 99-%-Quantil."""
    d = get_dist(kind)
    return float(d.ppf(0.99)) if kind == "exp" else d.hi


def build_bids(view) -> go.Figure:
    """Gebote über dem Wert: Zweitpreis (= Wert), Erstpreis, All-pay."""
    top = _x_limit(view.kind)
    keep = view.vs <= top
    keep_a = view.vs_allpay <= top
    step = max(1, int(keep.sum()) // 400)
    step_a = max(1, int(keep_a.sum()) // 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Zweitpreis (Gebot = Wert)", line=dict(color=C.COLOR_SECOND, width=3, dash="dot")))
    fig.add_trace(go.Scatter(x=view.vs[keep][::step], y=view.bs[keep][::step], mode="lines", name="Erstpreis", line=dict(color=C.COLOR_FIRST, width=3),
                             hovertemplate="Wert %{x:.1f}: Gebot %{y:.2f}<extra>Erstpreis</extra>"))
    fig.add_trace(go.Scatter(x=view.vs_allpay[keep_a][::step_a], y=view.bs_allpay[keep_a][::step_a], mode="lines", name="All-pay", line=dict(color=C.COLOR_ALLPAY, width=3),
                             hovertemplate="Wert %{x:.1f}: Gebot %{y:.2f}<extra>All-pay</extra>"))
    fig.update_xaxes(title_text="Wert des Spediteurs (Euro je Slot)", range=[0, top])
    fig.update_yaxes(title_text="Gebot (Euro je Slot)", range=[0, top])
    return _base(fig)


def build_revenue_bars(view) -> go.Figure:
    """Erlöse der drei Regeln (Integral) mit den simulierten Mittelwerten und Standardfehlern."""
    names = ["Zweitpreis", "Erstpreis", "All-pay"]
    exact = [view.rev_second, view.rev_first, view.rev_allpay]
    sim = [view.sim["zweit"], view.sim["erst"], view.sim["allpay"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=exact, marker_color=[C.COLOR_SECOND, C.COLOR_FIRST, C.COLOR_ALLPAY], name="Integral",
                         hovertemplate="%{x}: %{y:.3f} Euro<extra>Integral</extra>"))
    fig.add_trace(go.Scatter(x=names, y=[s[0] for s in sim], mode="markers", name="Simulation (4 Blöcke)", marker=dict(color="black", size=9, symbol="diamond"),
                             error_y=dict(type="data", array=[2 * s[1] for s in sim], visible=True), hovertemplate="%{x}: %{y:.3f} Euro<extra>Simulation</extra>"))
    fig.update_yaxes(title_text="Erlös (Euro je Slot)")
    return _base(fig, 320)


def build_risk(n: int, rho: float) -> go.Figure:
    """Verhältnis Erstpreis/Zweitpreis über dem Risikoexponenten für n und zum Vergleich n = 2, 5, 10 (Gleichverteilung)."""
    rhos = np.linspace(C.RHO_MIN, C.RHO_MAX, 61)
    fig = go.Figure()
    shown = sorted({2, 5, 10, n} - {1})
    for k in shown:
        lam = (k - 1) / (k - 1 + rhos)
        ratio = lam * k / (k - 1)           # Erstpreis-Erlös λ·100n/(n+1) geteilt durch Zweitpreis 100(n−1)/(n+1)
        fig.add_trace(go.Scatter(x=rhos, y=ratio, mode="lines", name=f"n = {k}", line=dict(width=4 if k == n else 2)))
    if n > 1:
        lam_now = (n - 1) / (n - 1 + rho)
        fig.add_trace(go.Scatter(x=[rho], y=[lam_now * n / (n - 1)], mode="markers", name="gewählt", marker=dict(color="black", size=11, symbol="diamond")))
    fig.add_hline(y=1.0, line_dash="dot", line_color=C.COLOR_NEUTRAL)
    fig.update_xaxes(title_text="Risikoexponent ρ (1 = risikoneutral)")
    fig.update_yaxes(title_text="Erlös Erstpreis / Erlös Zweitpreis")
    return _base(fig, 320)


def build_reserve(view) -> go.Figure:
    """Nutzen des Verkäufers über dem Mindestgebot r mit dem besten r, der Reserve aus φ(r) = v0 und dem Mindestgebot der Seitenleiste."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.rs, y=view.us, mode="lines", name="Nutzen U(r)", line=dict(color=C.COLOR_SECOND, width=3), hovertemplate="r = %{x:.1f}: %{y:.2f} Euro<extra></extra>"))
    fig.add_hline(y=view.u_plain, line_dash="dot", line_color=C.COLOR_NEUTRAL, annotation_text="ohne Mindestgebot über dem Eigenwert", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=[view.r_grid], y=[view.u_grid], mode="markers", name="Maximum auf dem Gitter", marker=dict(color=C.COLOR_BEST, size=12, symbol="diamond")))
    if view.r_star is not None:
        fig.add_trace(go.Scatter(x=[view.r_star], y=[view.u_star], mode="markers", name="Reserve aus φ(r) = v0", marker=dict(color="black", size=9, symbol="x")))
    fig.add_vline(x=view.r, line_dash="dash", line_color=C.COLOR_FIRST, annotation_text="gewählt", annotation_position="top")
    fig.update_xaxes(title_text="Mindestgebot r (Euro je Slot)")
    fig.update_yaxes(title_text="Nutzen des Verkäufers (Euro je Slot)")
    return _base(fig)


def build_bulow(view) -> go.Figure:
    """Bestes Verfahren mit n Bietern gegen einfache Auktion mit n + 1 Bietern über n."""
    ns = [r["n"] for r in view.rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["best"] for r in view.rows], mode="lines+markers", name="bestes Verfahren, n Bieter", line=dict(color=C.COLOR_BEST, width=3)))
    fig.add_trace(go.Scatter(x=ns, y=[r["plain_next"] for r in view.rows], mode="lines+markers", name="einfache Auktion, n + 1 Bieter", line=dict(color=C.COLOR_SECOND, width=3)))
    fig.add_vline(x=view.n, line_dash="dot", line_color=C.COLOR_NEUTRAL)
    fig.update_xaxes(title_text="Zahl der Bieter n", dtick=1)
    fig.update_yaxes(title_text="Nutzen des Verkäufers (Euro je Slot)")
    return _base(fig)


def build_threshold(view) -> go.Figure:
    """Teilnahmeschwelle p* des zusätzlichen Bieters über n (Werte über 1: ein zusätzlicher Bieter reicht nicht)."""
    ns = [r["n"] for r in view.rows]
    ps = [min(r["p_star"], 10.0) for r in view.rows]
    fig = go.Figure(go.Bar(x=ns, y=ps, marker_color=[C.COLOR_SECOND if r["p_star"] <= 1.0 else C.COLOR_FIRST for r in view.rows],
                           hovertemplate="n = %{x}: p* = %{y:.4f}<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dot", line_color=C.COLOR_NEUTRAL, annotation_text="p* = 1", annotation_position="top right")
    fig.update_xaxes(title_text="Zahl der Bieter n", dtick=1)
    fig.update_yaxes(title_text="Teilnahmeschwelle p* des zusätzlichen Bieters", type="log")
    return _base(fig, 300)


def build_asym(view) -> go.Figure:
    """Gleichgewichtsgebote des starken und des schwachen Bieters über dem eigenen Wert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Zweitpreis (Gebot = Wert)", line=dict(color=C.COLOR_SECOND, width=3, dash="dot")))
    fig.add_trace(go.Scatter(x=view.v1, y=view.b_strong, mode="lines", name="Erstpreis, starker Bieter (Werte bis 100)", line=dict(color=C.COLOR_FIRST, width=3)))
    fig.add_trace(go.Scatter(x=view.v2, y=view.b_weak, mode="lines", name=f"Erstpreis, schwacher Bieter (Werte bis {100 * view.w:.0f})", line=dict(color=C.COLOR_ALLPAY, width=3)))
    fig.update_xaxes(title_text="Eigener Wert (Euro je Slot)", range=[0, 100])
    fig.update_yaxes(title_text="Gebot (Euro je Slot)", range=[0, 100])
    return _base(fig)


def build_learning(res: dict) -> go.Figure:
    """Verlauf des mittleren Gebotsfaktors α über dem Lauf, mit dem Gleichgewicht."""
    traj = res["trajectory"]
    xs = np.linspace(0, res.get("rounds", len(traj)) if res.get("rounds") else len(traj), len(traj))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=traj, mode="lines", name=f"{res['rule']} (Mittel der Bieter)", line=dict(color=C.COLOR_FIRST, width=3)))
    fig.add_hline(y=res["alpha_eq"], line_dash="dot", line_color=C.COLOR_SECOND, annotation_text="Gleichgewicht", annotation_position="top right")
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Gebotsfaktor α (Gebot = α · Wert)", range=[0, 1.02])
    return _base(fig, 320)


def build_learning_cells(cells: list[dict]) -> go.Figure:
    """Letzter Gebotsfaktor je Zelle (Regel, Bieter, Format) mit Standardfehler, gegen das Gleichgewicht."""
    labels = [f"{c['rule']}, n = {c['n']}, {c['fmt']}" for c in cells]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[c["alpha_last"]["mean"] for c in cells], marker_color=[C.COLOR_FIRST if c["fmt"] == "Erstpreis" else C.COLOR_SECOND for c in cells],
                         error_y=dict(type="data", array=[2 * c["alpha_last"]["se"] for c in cells], visible=True), name="letzter Iterierter"))
    fig.add_trace(go.Scatter(x=labels, y=[c["alpha_eq"] for c in cells], mode="markers", name="Gleichgewicht", marker=dict(color="black", size=11, symbol="line-ew-open", line=dict(width=3))))
    fig.update_yaxes(title_text="Gebotsfaktor α", range=[0, 1.05])
    fig.update_xaxes(tickangle=-35)
    return _base(fig, 420)


def build_main(view) -> go.Figure:
    """Gebote im Erstpreis bei Mindestgebot r (rot) und im Zweitpreis (Gebot = Wert, grün) über dem Wert; unter r wird nicht geboten."""
    d = get_dist(view.kind)
    top = _x_limit(view.kind)
    keep = view.vs <= top
    step = max(1, int(keep.sum()) // 400)
    start = max(view.r, 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[start, top], y=[start, top], mode="lines", name="Zweitpreis (Gebot = Wert)", line=dict(color=C.COLOR_SECOND, width=3, dash="dot")))
    fig.add_trace(go.Scatter(x=view.vs[keep][::step], y=view.bs[keep][::step], mode="lines", name="Erstpreis", line=dict(color=C.COLOR_FIRST, width=3),
                             hovertemplate="Wert %{x:.1f}: Gebot %{y:.2f}<extra>Erstpreis</extra>"))
    if view.r > d.lo:
        fig.add_vline(x=view.r, line_dash="dash", line_color=C.COLOR_NEUTRAL, annotation_text="Mindestgebot", annotation_position="top")
    if view.v0 > 0:
        fig.add_vline(x=view.v0, line_dash="dot", line_color=C.COLOR_BEST, annotation_text="Eigenwert", annotation_position="bottom")
    fig.update_xaxes(title_text="Wert des Spediteurs (Euro je Slot)", range=[0, top])
    fig.update_yaxes(title_text="Gebot (Euro je Slot)", range=[0, top])
    return _base(fig)
