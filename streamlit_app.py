# src/streamlit_app.py
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Import your existing modules (NO algorithm changes)
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="McCabe–Thiele Distillation Simulator", page_icon="🧪", layout="wide")
    st.title("🧪 McCabe–Thiele Distillation Column Simulator ")

# ── Display options
    show_numbers = st.sidebar.checkbox("Show stage numbers", value=True)
    show_feed_arrow = st.sidebar.checkbox("Show feed arrow", value=True)

    # Sidebar inputs — typed values
    st.sidebar.header("📊 Input Parameters")

    st.sidebar.subheader("🎯 Compositions (Mole Fractions)")
    xD = st.sidebar.number_input("Distillate Composition (xD)", min_value=0.0, max_value=1.0, value=0.923, step=0.001, format="%.6f")
    xB = st.sidebar.number_input("Bottoms Composition (xB)",   min_value=0.0, max_value=1.0, value=0.071, step=0.001, format="%.6f")
    xF = st.sidebar.number_input("Feed Composition (xF)",      min_value=0.0, max_value=1.0, value=0.584, step=0.001, format="%.6f")

    st.sidebar.subheader("⚙️ Operating Conditions")
    R     = st.sidebar.number_input("Reflux Ratio (R)",   min_value=0.1,  max_value=50.0,  value=2.67, step=0.01,  format="%.6f")
    q     = st.sidebar.number_input("Feed Quality (q)",   min_value=-5.0, max_value=5.0,   value=1.0,  step=0.1,   format="%.6f")
    alpha = st.sidebar.number_input("Relative Volatility (α)", min_value=1.001, max_value=100.0, value=3.04, step=0.01, format="%.6f")

    st.sidebar.subheader("🔧 Advanced Settings")
    max_iterations = st.sidebar.number_input("Max Iterations", min_value=10, max_value=5000, value=500, step=10)
    tolerance      = st.sidebar.number_input("Tolerance", min_value=1e-15, max_value=1e-6, value=1e-10, step=1e-12, format="%.1e")
    N_max      = st.sidebar.number_input("Maximum number of stages", min_value=1, max_value=500, value=25, step=1)
    # Validate compositions
    if not (0.0 <= xB < xF < xD <= 1.0):
        st.error("⚠️ Invalid compositions! They must satisfy: 0 ≤ xB < xF < xD ≤ 1.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("📈 McCabe–Thiele Diagram")
        try:
            with st.spinner("Running McCabe–Thiele stepping..."):
                result = stair_stepper(
                    xD=xD, xB=xB, xF=xF, q=q, R=R, alpha=alpha,
                    max_iterations=int(max_iterations), tolerance=float(tolerance), N_max=N_max
                )
            fig = plot_mccabe_thiele(
                result, xD, xB, xF, q, R, alpha,
                show_numbers=show_numbers,
                show_feed_arrow=show_feed_arrow
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
            st.stop()

    with col2:
        st.header("📋 Results Summary")
        st.metric("🔢 Number of Stage:", result.get("stage_counter", "—")+1)
        feed_idx = result.get("feed_stage_index", -1)
        st.metric("🍽️ Feed Stage Index (0-based)", feed_idx if feed_idx != -1 else "Not determined")

        message = result.get("message", "No message")
        ml = str(message).lower()
        if "error" in ml or "unphysical" in ml:
            st.error(message)
        elif "warn" in ml or "near" in ml:
            st.warning(message)
        else:
            st.info(message)

        st.subheader("📏 Operating Lines / Break Point")
        mR, bR = rectifying_line(R, xD)
        qline = q_line(xF, q)
        xstar, ystar = break_point(mR, bR, qline)
        ms, bs = stripping_line(xB, xstar, ystar)

        st.write(f"- Rectifying: **y = {mR:.6f} x + {bR:.6f}**")
        st.write(f"- Stripping: **y = {ms:.6f} x + {bs:.6f}**")
        if qline.get("vertical", False):
            st.write(f"- q-line: **x = {xF:.6f}** (vertical; q=1)")
        elif qline.get("horizontal", False):
            st.write(f"- q-line: **y = {xF:.6f}** (horizontal; q=0)")
        else:
            st.write(f"- q-line: **y = {qline['mq']:.6f} x + {qline['bq']:.6f}**")
            
        st.write(f"- Break point: **( x, y )** = **({xstar:.6f}, {ystar:.6f})**")

    st.markdown("---")
    st.header("📊 Detailed Results")

    tab1, tab2, tab3 = st.tabs(["📈 Stage Data", "🔍 Q-line Info", "📚 Theory"])

    with tab1:
        vertices = result.get("vertices", [])
        if vertices:
            df_vertices = pd.DataFrame(vertices, columns=["x", "y"])
            df_vertices.index = df_vertices.index + 1
            df_vertices.index.name = "Step"
            st.dataframe(df_vertices, use_container_width=True)
            st.download_button(
                "📥 Download Stage Data (CSV)",
                df_vertices.to_csv(index=True),
                file_name="mccabe_thiele_stages.csv",
                mime="text/csv",
            )
        else:
            st.info("No stage vertices returned.")

    with tab2:
        st.subheader("Q-line Information")
        if qline.get("vertical", False):
            st.write("- **Type:** Vertical (q = 1; saturated liquid)")
            st.write(f"- **Equation:** x = {xF:.6f}")
        elif qline.get("horizontal", False):
            st.write("- **Type:** Horizontal (q = 0; saturated vapor)")
            st.write(f"- **Equation:** y = {xF:.6f}")
        else:
            mq, bq = qline["mq"], qline["bq"]
            st.write("- **Type:** General q-line")
            st.write(f"- **Slope (mq):** {mq:.6f}")
            st.write(f"- **Intercept (bq):** {bq:.6f}")
            st.write(f"- **Equation:** y = {mq:.6f} x + {bq:.6f}")

        if q == 1:
            st.info("🌡️ Feed condition: **Saturated liquid**")
        elif q == 0:
            st.info("💨 Feed condition: **Saturated vapor**")
        elif q > 1:
            st.info("❄️ Feed condition: **Subcooled liquid**")
        elif q < 0:
            st.info("🔥 Feed condition: **Superheated vapor**")
        else:
            st.info("🌫️ Feed condition: **Two-phase mixture**")

    with tab3:
        st.header("McCabe–Thiele Method (Quick Notes)")
        st.markdown("""
    **Key Points:**
    1. **Assumptions**: Constant molar overflow (CMO), total condenser, constant relative volatility
    2. **Operating lines**:  
    - Rectifying: y = R/(R+1) × x + x_D/(R+1)  
    - Stripping: passes through (x*, y*) and (x_B, x_B)
    3. **Stepping procedure**: horizontal to equilibrium curve, vertical to operating line  
    4. **Termination**: when x approaches x_B or maximum iterations reached
            """)

if __name__ == "__main__":
    main()
