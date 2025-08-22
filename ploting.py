
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
    show_feed_arrow: bool = True,
    theme: str = "plotly_white"  # or "plotly_dark", "simple_white"
) -> "go.Figure":
    """Enhanced McCabe-Thiele diagram with better styling and interactivity"""
    
    # Color scheme
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
    
    # Equilibrium curve with enhanced styling
    fig.add_trace(go.Scatter(
        x=x_eq, y=y_eq, 
        mode="lines", 
        name="Equilibrium Curve", 
        line=dict(width=3, color=colors['equilibrium']),
        hovertemplate="<b>Equilibrium</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>"
    ))
    
    # Diagonal y=x with enhanced styling
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode="lines", 
        name="y = x",
        line=dict(width=2, dash="dash", color=colors['diagonal']),
        hovertemplate="<b>45° Line</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>"
    ))
    
    # ── Operating lines via your functions
    mR, bR = rectifying_line(R, xD)
    qline = q_line(xF, q)
    xstar, ystar = break_point(mR, bR, qline)
    ms, bs = stripping_line(xB, xstar, ystar)
    
    # Rectifying line with enhanced styling
    x_rect = np.linspace(max(0, min(xstar, xD)), min(1, max(xstar, xD)), 200)
    y_rect = mR * x_rect + bR
    fig.add_trace(go.Scatter(
        x=x_rect, y=y_rect, 
        mode="lines", 
        name=f"Rectifying Line (R={R:.2f})", 
        line=dict(width=3, color=colors['rectifying']),
        hovertemplate="<b>Rectifying Line</b><br>x: %{x:.4f}<br>y: %{y:.4f}<br>Slope: " + f"{mR:.4f}<extra></extra>"
    ))
    
    # Stripping line with enhanced styling
    x_strip = np.linspace(max(0, min(xB, xstar)), min(1, max(xB, xstar)), 200)
    y_strip = ms * x_strip + bs
    fig.add_trace(go.Scatter(
        x=x_strip, y=y_strip, 
        mode="lines", 
        name="Stripping Line", 
        line=dict(width=3, color=colors['stripping']),
        hovertemplate="<b>Stripping Line</b><br>x: %{x:.4f}<br>y: %{y:.4f}<br>Slope: " + f"{ms:.4f}<extra></extra>"
    ))
    
    # Enhanced q-line handling
    if qline.get("vertical", False):
        fig.add_shape(
            type="line", x0=xF, x1=xF, y0=0, y1=1,
            line=dict(width=3, dash="dash", color=colors['qline']),
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", 
            name=f"q-line (q={q:.2f}, vertical)",
            line=dict(width=3, dash="dash", color=colors['qline'])
        ))
    elif qline.get("horizontal", False):
        fig.add_shape(
            type="line", x0=0, x1=1, y0=xF, y1=xF,
            line=dict(width=3, dash="dash", color=colors['qline']),
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", 
            name=f"q-line (q={q:.2f}, horizontal)",
            line=dict(width=3, dash="dash", color=colors['qline'])
        ))
    else:
        mq, bq = qline["mq"], qline["bq"]
        x_q = np.linspace(0, 1, 400)
        y_q = mq * x_q + bq
        mask = (y_q >= 0) & (y_q <= 1)
        fig.add_trace(go.Scatter(
            x=x_q[mask], y=y_q[mask], 
            mode="lines", 
            name=f"q-line (q={q:.2f})", 
            line=dict(width=3, dash="dash", color=colors['qline']),
            hovertemplate="<b>q-line</b><br>x: %{x:.4f}<br>y: %{y:.4f}<br>Slope: " + f"{mq:.4f}<extra></extra>"
        ))
    
    # ── ENHANCED STAIRS with better visualization
    vertices = result.get("vertices", [])
    annotations = []
    
    if vertices:
        vx = [v[0] for v in vertices]
        vy = [v[1] for v in vertices]
        
        # Ensure clean finish at (xB, xB)
        if abs(vx[-1] - xB) > 1e-12 or abs(vy[-1] - xB) > 1e-12:
            vx.append(xB); vy.append(xB)
        
        # Enhanced stairs with better styling
        fig.add_trace(go.Scatter(
            x=vx, y=vy, 
            mode="lines+markers", 
            name=f"Stages (Total: {result.get('stage_counter', 0) + 1})",
            line=dict(width=2.5, color=colors['stages']), 
            marker=dict(size=8, color=colors['stages'], symbol='circle'),
            line_shape="hv",
            hovertemplate="<b>Stage Point</b><br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>"
        ))
        
        # Enhanced stage numbering
        if show_numbers:
            stage_num = 1
            for i in range(1, len(vertices)):
                x_prev, y_prev = vertices[i-1]
                x_i, y_i = vertices[i]
                # vertical move → x unchanged
                if abs(x_i - x_prev) < 1e-6:
                    annotations.append(dict(
                        x=x_i + 0.015, y=y_i + 0.02,
                        text=f"<b>{stage_num}</b>", 
                        showarrow=False,
                        font=dict(size=14, color="white", family="Arial Black"), 
                        bgcolor=colors['stages'],
                        bordercolor="white",
                        borderwidth=1,
                        borderpad=4,
                        xanchor="left", yanchor="bottom"
                    ))
                    stage_num += 1
        
        # Enhanced feed stage arrow
        if show_feed_arrow:
            feed_idx = result.get("feed_stage_index", -1)
            if isinstance(feed_idx, int) and feed_idx >= 0:
                feed_vertex_idx = 2 * (feed_idx + 1)
                if 0 <= feed_vertex_idx < len(vertices):
                    fx, fy = vertices[feed_vertex_idx]
                    annotations.append(dict(
                        x=fx, y=fy, 
                        ax=fx + 0.12, ay=fy + 0.08,
                        text=f"<b>Feed Stage {feed_idx+1}</b>",
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
    
    # Enhanced key points with better styling
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
            xanchor="left", yanchor="bottom", 
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
