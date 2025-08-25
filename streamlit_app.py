# src/streamlit_app.py
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Import your existing modules 
# ──────────────────────────────────────────────────────────────────────────────

from mccabe import stair_stepper, rectifying_line, q_line, break_point, stripping_line
from thermo_simple import yfxeq, xfyeq
from ploting import *



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
    N_max      = st.sidebar.number_input("Maximum number of stages", min_value=1, max_value=500, value=500, step=1)
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
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
            st.stop()

    with col2:
        st.header("📋 Results Summary")
        st.metric("🔢 Number of Stage:", result.get("stage_counter", "—")+1)
        feed_idx = result.get("feed_stage_index", -1)
        st.metric("Feed Stage:", feed_idx if feed_idx != -1 else "Not determined")

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
