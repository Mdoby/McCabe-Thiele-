
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # ── Equilibrium + y=x
    x_eq, y_eq = create_equilibrium_curve(alpha)
    ax.plot(x_eq, y_eq, linewidth=2, label="Equilibrium Curve")
    ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.6, label="y = x")

    # ── Operating lines via your functions
    mR, bR = rectifying_line(R, xD)
    qline = q_line(xF, q)
    xstar, ystar = break_point(mR, bR, qline)
    ms, bs = stripping_line(xB, xstar, ystar)

    # Rectifying line segment (x between x* and xD)
    x_rect = np.linspace(max(0, min(xstar, xD)), min(1, max(xstar, xD)), 200)
    ax.plot(x_rect, mR * x_rect + bR, linewidth=2, label="Rectifying Line")

    # Stripping line segment (x between xB and x*)
    x_strip = np.linspace(max(0, min(xB, xstar)), min(1, max(xB, xstar)), 200)
    ax.plot(x_strip, ms * x_strip + bs, linewidth=2, label="Stripping Line")

    # q-line
    if qline.get("vertical", False):
        ax.axvline(x=xF, linestyle="--", linewidth=2, label="q-line (vertical)")
    elif qline.get("horizontal", False):
        ax.axhline(y=xF, linestyle="--", linewidth=2, label="q-line (horizontal)")
    else:
        mq, bq = qline["mq"], qline["bq"]
        x_q = np.linspace(0, 1, 400)
        y_q = mq * x_q + bq
        mask = (y_q >= 0) & (y_q <= 1)
        ax.plot(x_q[mask], y_q[mask], linestyle="--", linewidth=2, label="q-line")

    # ── STAIRS (draw first, then we can number / add arrow)
    vertices = result.get("vertices", [])
    if vertices:
        vx = [v[0] for v in vertices]
        vy = [v[1] for v in vertices]

        # Ensure clean finish at (xB, xB) if last vertex isn't exactly there
        if abs(vx[-1] - xB) > 1e-12 or abs(vy[-1] - xB) > 1e-12:
            vx.append(xB); vy.append(xB)

        ax.step(vx, vy, where="post", linewidth=1.8, label="Stages")
        ax.scatter(vx, vy, s=12)

            # ---------- Stage numbers (ONLY vertical steps = true trays) ----------
        # Use a true step plot and explicitly end at (xB, xB)
        if show_numbers:    
            vertices = result.get("vertices", [])
            if vertices:
                vx = [v[0] for v in vertices]
                vy = [v[1] for v in vertices]

                # Force final point at bottoms if needed (fixes the square-marked issue)
                if abs(vx[-1] - xB) > 1e-12 or abs(vy[-1] - xB) > 1e-12:
                    vx.append(xB)
                    vy.append(xB)

                # Draw proper stairs (horizontal then vertical) — removes diagonal artifacts
                ax.step(vx, vy, where="post", linewidth=1.8, label="Stages")
                ax.scatter(vx, vy, s=12)  # optional: show corners
                # ===== Stage numbering: label every vertical step endpoint =====
                vertices = result.get("vertices", [])
                stage_counter = int(result.get("stage_counter") or 0)

                # Vertical step endpoints are at indices 1, 3, 5, ...
                stage_vertex_indices = [i for i in range(1, len(vertices), 2)]

                for s, idx in enumerate(stage_vertex_indices, start=1):
                    xx, yy = vertices[idx]
                    ax.text(xx + 0.008, yy + 0.008, str(s),
                            fontsize=15, ha="left", va="bottom",
                            alpha=0.95, zorder=6)

        # ---------- Feed stage arrow ----------
        if show_feed_arrow:
            feed_idx = result.get("feed_stage_index", -1)  # 0-based stage index
            if isinstance(feed_idx, int) and feed_idx >= 0:
                # vertical endpoint vertex for stage S is 2*(S) in 1-based,
                # so for 0-based feed_idx → vertex index = 2*(feed_idx+1)
                feed_vertex_idx = 2 * (feed_idx )
                if 0 <= feed_vertex_idx < len(vertices):
                    fx, fy = vertices[feed_vertex_idx]
                    ax.annotate(
                        f"Feed stage = {feed_idx}",
                        xy=(fx, fy), xytext=(fx + 0.08, fy + 0.05),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="red"),
                        fontsize=10, color="red", weight="bold"
                    )

    # ── Points & styling
    ax.plot([xD],   [xD],   "o", label=f"Distillate (xD={xD:.3f})")
    ax

    return fig