
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from mccabe import stair_stepper, rectifying_line, q_line, break_point, stripping_line
from thermo_simple import yfxeq, xfyeq


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized wrappers (so scalar yfxeq/xfyeq accept arrays)
# ──────────────────────────────────────────────────────────────────────────────
def yfxeq_vec(x_array, alpha):
    x_array = np.atleast_1d(x_array)
    return np.array([yfxeq(float(x), float(alpha)) for x in x_array])

def xfyeq_vec(y_array, alpha):
    y_array = np.atleast_1d(y_array)
    return np.array([xfyeq(float(y), float(alpha)) for y in y_array])


def create_equilibrium_curve(alpha: float, n_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    x_eq = np.linspace(0.0, 1.0, n_points)
    y_eq = yfxeq_vec(x_eq, alpha)
    return x_eq, y_eq
def plot_mccabe_thiele(
    result: Dict, xD: float, xB: float, xF: float, q: float, R: float, alpha: float,
    show_numbers: bool = True,
    show_feed_arrow: bool = True
) -> "go.Figure":
    
    theme: str = "plotly_white"  # or "plotly_dark", "simple_white"

    
    colors = {
        'equilibrium': '#2E86AB',
        'diagonal': '#A23B72', 
        'rectifying': '#F18F01',
        'stripping': '#C73E1D',
        'qline': '#592E83',
        'stages': '#4CAF50',
        'feed_point': '#FF6B6B'
    }
    # ── Equilibrium + y=x
    x_eq, y_eq = create_equilibrium_curve(alpha)

    fig = go.Figure()

    # Equilibrium curve
    fig.add_trace(go.Scatter(
        x=x_eq, y=y_eq, mode="lines", name="Equilibrium Curve", line=dict(width=2)
    ))

    # Diagonal y=x
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="y = x",
        line=dict(width=2, dash="dash")
    ))

    # ── Operating lines via your functions
    mR, bR = rectifying_line(R, xD)
    qline = q_line(xF, q)
    xstar, ystar = break_point(mR, bR, qline)
    ms, bs = stripping_line(xB, xstar, ystar)

    # Rectifying line segment (x between x* and xD)
    x_rect = np.linspace(max(0, min(xstar, xD)), min(1, max(xstar, xD)), 200)
    y_rect = mR * x_rect + bR
    fig.add_trace(go.Scatter(
        x=x_rect, y=y_rect, mode="lines", name="Rectifying Line", line=dict(width=2)
    ))

    # Stripping line segment (x between xB and x*)
    x_strip = np.linspace(max(0, min(xB, xstar)), min(1, max(xB, xstar)), 200)
    y_strip = ms * x_strip + bs
    fig.add_trace(go.Scatter(
        x=x_strip, y=y_strip, mode="lines", name="Stripping Line", line=dict(width=2)
    ))

    # q-line
    if qline.get("vertical", False):
        # vertical shape x = xF
        fig.add_shape(
            type="line", x0=xF, x1=xF, y0=0, y1=1,
            line=dict(width=2, dash="dash"),
            name="q-line (vertical)"
        )
        # Add a dummy legend entry for visibility
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name="q-line (vertical)",
            line=dict(width=2, dash="dash")
        ))
    elif qline.get("horizontal", False):
        # horizontal shape y = xF
        fig.add_shape(
            type="line", x0=0, x1=1, y0=xF, y1=xF,
            line=dict(width=2, dash="dash"),
            name="q-line (horizontal)"
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name="q-line (horizontal)",
            line=dict(width=2, dash="dash")
        ))
    else:
        mq, bq = qline["mq"], qline["bq"]
        x_q = np.linspace(0, 1, 400)
        y_q = mq * x_q + bq
        mask = (y_q >= 0) & (y_q <= 1)
        fig.add_trace(go.Scatter(
            x=x_q[mask], y=y_q[mask], mode="lines", name="q-line", line=dict(width=2, dash="dash")
        ))

    # ── STAIRS
    vertices = result.get("vertices", [])
    annotations = []

    if vertices:
        vx = [v[0] for v in vertices]
        vy = [v[1] for v in vertices]

        # Ensure clean finish at (xB, xB) if last vertex isn't exactly there
        if abs(vx[-1] - xB) > 1e-12 or abs(vy[-1] - xB) > 1e-12:
            vx.append(xB); vy.append(xB)

        # Plotly can draw HV stairs with line_shape="hv"
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode="lines+markers", name="Stages",
            line=dict(width=2), marker=dict(size=6),
            line_shape="hv"
        ))

        # ---------- Stage numbers (ONLY vertical steps = true trays) ----------
        if show_numbers:
            stage_num = 1
            for i in range(1, len(vertices)+1):
                x_prev, y_prev = vertices[i-1]
                x_i,   y_i     = vertices[i]
                # vertical move → x unchanged
                if abs(x_i - x_prev) < 1e-6:
                    annotations.append(dict(
                        x=x_i + 0.01, y=y_i + 0.03,
                        text=str(stage_num), showarrow=False,
                        font=dict(size=16, color="black"), xanchor="left", yanchor="bottom"
                    ))
                    stage_num += 1

        # Enhanced feed stage arrow
        if show_feed_arrow:
            feed_idx = result.get("feed_stage_index", -1)
            if isinstance(feed_idx, int) and feed_idx >= 0:
                feed_vertex_idx = 2 * (feed_idx )
                if 0 <= feed_vertex_idx < len(vertices):
                    fx, fy = vertices[feed_vertex_idx]
                    annotations.append(dict(
                        x=fx, y=fy, 
                        ax=fx + 0.12, ay=fy + 0.08,
                        text=f"<b>Feed Stage {feed_idx}</b>",
                        showarrow=True, 
                        arrowhead=3, 
                        arrowsize=1.5, 
                        arrowwidth=3,
                        arrowcolor=colors['feed_point'], 
                        font=dict(color=colors['feed_point'], size=13, family="Arial"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=colors['feed_point'],
                        borderwidth=2
                    ))

    points_data = [
        (xD, xD, f"Distillate<br>xD = {xD:.4f}", "diamond", "#FF6B6B"),
        (xB, xB, f"Bottoms<br>xB = {xB:.4f}", "diamond", "#4ECDC4"),
        (xF, xF, f"Feed<br>xF = {xF:.4f}", "star", colors['feed_point']),
        (xstar, ystar, f"Break Point<br>({xstar:.4f}, {ystar:.4f})", "x", "#9B59B6")
    ]
    
    for x, y, name, symbol, color in points_data:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], 
            mode="markers", 
            name=name.replace('<br>', ' '),
            marker=dict(size=12, color=color, symbol=symbol, line=dict(width=2, color="white")),
            hovertemplate=f"<b>{name}</b><extra></extra>"
        ))
    
    # Enhanced layout with better styling
    fig.update_layout(
        template=theme,
        width=1000, height=750,
        title=dict(
            text="<b>McCabe-Thiele Distillation Column Design</b>",
            x=0.5,
            font=dict(size=18, family="Arial")
        ),
        xaxis=dict(
            range=[0, 1], 
            title=dict(text="<b>x (liquid mole fraction of light component)</b>", font=dict(size=14)),
            showgrid=True, 
            gridwidth=1, 
            gridcolor="rgba(128,128,128,0.3)",
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor="rgba(0,0,0,0.5)"
        ),
        yaxis=dict(
            range=[0, 1], 
            title=dict(text="<b>y (vapor mole fraction of light component)</b>", font=dict(size=14)),
            showgrid=True, 
            gridwidth=1, 
            gridcolor="rgba(128,128,128,0.3)",
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor="rgba(0,0,0,0.5)"
        ),
        legend=dict(
            x=0.02, y=0.02, 
            xanchor="right", yanchor="bottom", 
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        annotations=annotations,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # Add efficiency information as subtitle
    total_stages = result.get("stage_counter", 0) + 1
    theoretical_min = f"Theoretical stages: {total_stages}"
    fig.add_annotation(
        text=f"<i>{theoretical_min} | α = {alpha:.2f} | R = {R:.2f}</i>",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, color="gray"),
        xanchor="center"
    )

    return fig
