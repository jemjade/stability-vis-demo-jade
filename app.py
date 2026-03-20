"""
Stability-Aware Adaptive Visualization Framework — Interactive Demo
Run: streamlit run app.py

This demo imports the synthetic session generator and knowledge-graph logic from
simulation.py so that the app and analysis share a single source of truth.

The interface is intended for feasibility-oriented illustration in a synthetic-session
setting and should not be interpreted as externally validated evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from simulation import (
    KnowledgeGraphSimulator,
    generate_sessions,
    INTENTS,
    UIS,
    SEED,
    N_SESSIONS,
    GATE_DEFAULT,
    MIN_CONSEC_HIGH_WINDOWS,
    WINDOWS_PER_SESSION,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stability-Aware Adaptive Visualization",
    page_icon="🧠",
    layout="wide",
)

# ── display helpers ────────────────────────────────────────────────────────────
UI_LABELS = {
    "static": "Static",
    "immediate_adaptive": "Immediate Adaptive",
    "stability_aware_adaptive": "Stability-Aware",
}
UI_COLORS = {
    "static": "#636EFA",
    "immediate_adaptive": "#EF553B",
    "stability_aware_adaptive": "#00CC96",
}


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    high = df[df["cognitive_load"] == "high"]
    return high.groupby("ui_type").agg(
        task_time=("task_completion_time_sec", "mean"),
        comprehension=("comprehension_score_0_3", "mean"),
        filter_cnt=("filter_count", "mean"),
        backtrack=("backtrack_count", "mean"),
        click_cnt=("click_count", "mean"),
        gate_open_rate=("gate_reason", lambda s: float((s == "ICM_semantic_justified").mean())),
    ).reset_index()


@st.cache_data
def load_data(gate_threshold: float) -> tuple[pd.DataFrame, dict]:
    df, kg = generate_sessions(seed=SEED, gate_threshold=gate_threshold)
    edges_im = [
        (u, v, d["weight"])
        for u, v, d in kg.G.edges(data=True)
        if d.get("etype") == "I→M"
    ]
    edges_mc = [
        (u, v, d["weight"])
        for u, v, d in kg.G.edges(data=True)
        if d.get("etype") == "M→C"
    ]
    return df, {"edges_im": edges_im, "edges_mc": edges_mc, "kg": kg}


# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Parameters")
threshold = st.sidebar.slider(
    "Semantic Gate Threshold (θₛ)",
    0.30,
    0.90,
    float(GATE_DEFAULT),
    0.01,
    help="The ICM semantic gate score must exceed this threshold to justify adaptation.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Live Gate Inspector")
sel_intent = st.sidebar.selectbox("User Intent", INTENTS, index=1)
sel_load = st.sidebar.selectbox("Cognitive Load", ["high", "low"], index=0)
sel_consecutive = st.sidebar.slider(
    "Consecutive High-Load Windows",
    1,
    WINDOWS_PER_SESSION,
    MIN_CONSEC_HIGH_WINDOWS,
    1,
    help="The paper framing emphasizes cognitively necessary adaptation; this demo only allows adaptation when high cognitive load persists across at least two consecutive windows.",
)
st.sidebar.markdown("**Metric Signals (0–1)**")
sig_dwell = st.sidebar.slider("dwell_time", 0.0, 1.0, 0.70, 0.05)
sig_click = st.sidebar.slider("click_rate", 0.0, 1.0, 0.65, 0.05)
sig_filt = st.sidebar.slider("filtering", 0.0, 1.0, 0.55, 0.05)
sig_back = st.sidebar.slider("backtracking", 0.0, 1.0, 0.60, 0.05)
sig_eff = st.sidebar.slider("efficiency", 0.0, 1.0, 0.50, 0.05)
sig_comp = st.sidebar.slider("comprehension", 0.0, 1.0, 0.70, 0.05)

live_sigs = {
    "dwell_time": sig_dwell,
    "click_rate": sig_click,
    "filtering": sig_filt,
    "backtracking": sig_back,
    "efficiency": sig_eff,
    "comprehension": sig_comp,
}

# ── load data and live gate evaluation ────────────────────────────────────────
df, kg_meta = load_data(gate_threshold=threshold)
kg_live = KnowledgeGraphSimulator(rng=np.random.default_rng(SEED), gate_threshold=threshold)
adapt_live, reason_live, score_live, debug_live = kg_live.should_adapt(
    sel_intent,
    sel_load,
    live_sigs,
    consecutive_high_windows=sel_consecutive,
)
actions_live = kg_live.get_action_recommendations(live_sigs, topk=2)
summ = summary_table(df)

# ── header ─────────────────────────────────────────────────────────────────────
st.title("🧠 Stability-Aware Adaptive Visualization Framework")
st.markdown(
    f"**HCII 2026 Poster Demo** · Suhyun Park · Graduate School of SW and AI Convergence, Korea University  \n"
    f"Synthetic sessions: **N = {N_SESSIONS}** · Seed: {SEED} · Preliminary synthetic-session consistency: **>90%**"
)
st.caption(
    "This demo is intended for feasibility-oriented illustration in a synthetic-session setting. "
    "The displayed values reflect simulated trends and should not be interpreted as externally validated effects."
)
st.markdown("---")

# ── tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 UI Comparison",
        "🔍 Live Gate Inspector",
        "🕸️ KG Subgraph",
        "📈 Illustrative Threshold Sweep",
        "🏃 Fitness Dashboard",
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 · UI Comparison
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Performance by UI Type — High Cognitive Load Sessions")
    st.caption(
        "All values shown below are derived from synthetic sessions and are intended "
        "for feasibility-oriented illustration rather than external validation."
    )

    ui_order = ["static", "immediate_adaptive", "stability_aware_adaptive"]

    def get(col: str) -> list[float]:
        return [summ[summ.ui_type == u][col].values[0] for u in ui_order]

    tct_vals = get("task_time")
    comp_vals = get("comprehension")
    static_tct = tct_vals[0]
    static_comp = comp_vals[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        delta_str = f"-{(static_tct - tct_vals[2]) / static_tct * 100:.1f}% vs static"
        st.metric("Task Time · Stability-Aware", f"{tct_vals[2]:.1f}s", delta_str, delta_color="inverse")
    with c2:
        delta_comp = f"+{(comp_vals[2] - static_comp) / static_comp * 100:.1f}% vs static"
        st.metric("Comprehension · Stability-Aware", f"{comp_vals[2]:.2f}", delta_comp)
    with c3:
        gate_row = summ[summ.ui_type == "stability_aware_adaptive"].iloc[0]
        st.metric(
            "Gate Open Rate",
            f"{gate_row.gate_open_rate * 100:.1f}%",
            help="Computed among high-load synthetic sessions that reached semantic evaluation.",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        for ui, color in UI_COLORS.items():
            val = summ[summ.ui_type == ui]["task_time"].values[0]
            fig.add_bar(
                x=[UI_LABELS[ui]],
                y=[val],
                name=UI_LABELS[ui],
                marker_color=color,
                showlegend=False,
                text=[f"{val:.1f}s"],
                textposition="outside",
            )
        fig.update_layout(
            title="Task Completion Time (sec)",
            yaxis_title="seconds",
            height=350,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        for ui, color in UI_COLORS.items():
            val = summ[summ.ui_type == ui]["comprehension"].values[0]
            fig.add_bar(
                x=[UI_LABELS[ui]],
                y=[val],
                name=UI_LABELS[ui],
                marker_color=color,
                showlegend=False,
                text=[f"{val:.2f}"],
                textposition="outside",
            )
        fig.update_layout(
            title="Comprehension Score (0–3)",
            yaxis_title="score",
            yaxis_range=[0, 3.2],
            height=350,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Interaction Burden (High Load)")
    burden_metrics = ["filter_cnt", "backtrack", "click_cnt"]
    burden_labels = ["Filter Count", "Backtrack Count", "Click Count"]
    fig2 = go.Figure()
    for ui, color in UI_COLORS.items():
        row = summ[summ.ui_type == ui].iloc[0]
        fig2.add_bar(
            name=UI_LABELS[ui],
            x=burden_labels,
            y=[row[m] for m in burden_metrics],
            marker_color=color,
        )
    fig2.update_layout(
        barmode="group",
        height=350,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Gate Open Rate by Intent (Stability-Aware)")
    st.caption("Computed among high-load synthetic sessions that reached semantic evaluation.")
    high_stab = df[
        (df["cognitive_load"] == "high")
        & (df["ui_type"] == "stability_aware_adaptive")
    ].copy()
    high_stab["gate_open"] = (high_stab["gate_reason"] == "ICM_semantic_justified").astype(int)
    gor = high_stab.groupby("intent")["gate_open"].mean().reindex(INTENTS) * 100

    fig3 = px.bar(
        x=gor.index,
        y=gor.values,
        text=[f"{v:.1f}%" for v in gor.values],
        color_discrete_sequence=["#00CC96"],
        labels={"x": "Intent", "y": "Gate Open Rate (%)"},
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        height=320,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 · Live Gate Inspector
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Live Semantic Gate Inspector")
    st.caption(
        "This inspector illustrates paper-aligned semantic gating in a synthetic-session setting. "
        "Adaptation is only allowed when cognitive load is high and persists across consecutive windows."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if adapt_live:
            st.success(
                f"✅ **GATE OPEN** — Adaptation triggered  \n"
                f"ICM Score: **{score_live:.3f}** ≥ θ={threshold:.2f}"
            )
        else:
            st.error(
                f"🔒 **GATE CLOSED** — No adaptation  \n"
                f"ICM Score: **{score_live:.3f}**"
            )

        st.markdown(f"**Intent:** `{sel_intent}`")
        st.markdown(f"**Cognitive Load:** `{sel_load}`")
        st.markdown(f"**Consecutive High-Load Windows:** `{sel_consecutive}`")

        if adapt_live and actions_live:
            st.markdown("**Recommended Actions:**")
            for a, s in actions_live:
                st.write(f"• `{a}` ({s:.3f})")

        st.markdown("**Gate Reason:**")
        st.code(reason_live)

    with col2:
        contribs = debug_live.copy()
        contribs.pop("consecutive_high_windows", None)
        contribs.pop("min_required_high_windows", None)

        if contribs:
            contrib_df = pd.DataFrame([{"metric": k, **v} for k, v in contribs.items()])
            fig = px.bar(
                contrib_df.sort_values("contrib", ascending=False),
                x="metric",
                y="contrib",
                text="contrib",
                color="contrib",
                color_continuous_scale="Tealgrn",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                title="Metric Contribution to ICM Score",
                height=360,
                coloraxis_showscale=False,
                plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                contrib_df[
                    ["metric", "signal", "w(I→M)", "w(M→C)", "contrib"]
                ].sort_values("contrib", ascending=False),
                use_container_width=True,
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 · KG Subgraph
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🕸️ ICM Semantic Gate Subgraph")
    st.markdown("Intent → Metric → Cognitive-State edges used for semantic justification in the current prototype.")

    edges_im = kg_meta["edges_im"]
    edges_mc = kg_meta["edges_mc"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Intent → Metric (I→M)**")
        im_df = pd.DataFrame(edges_im, columns=["from", "to", "weight"])
        st.dataframe(im_df, use_container_width=True, height=300)

    with col2:
        st.markdown("**Metric → Cognitive State (M→C)**")
        mc_df = pd.DataFrame(edges_mc, columns=["from", "to", "weight"])
        st.dataframe(mc_df, use_container_width=True, height=300)

    st.caption(
        "The full knowledge graph contains 24 nodes and 49 typed edges across "
        "User, Intent, Event, Cognitive State, Metric, and Action nodes."
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 · Illustrative Threshold Sweep
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 Illustrative Threshold Sweep")
    st.caption(
        "This synthetic threshold sweep is intended to illustrate the trade-off between "
        "gate selectivity, task-time improvement, and volatility trends. It should not be "
        "interpreted as a full externally validated performance analysis."
    )

    thresholds = np.round(np.arange(0.30, 0.91, 0.05), 2)
    sweep_rows = []

    base_static = df[
        (df["ui_type"] == "static") & (df["cognitive_load"] == "high")
    ]
    static_mean_tct = base_static["task_completion_time_sec"].mean()

    for th in thresholds:
        dft, _ = generate_sessions(seed=SEED, gate_threshold=float(th))

        high_stab = dft[
            (dft["ui_type"] == "stability_aware_adaptive")
            & (dft["cognitive_load"] == "high")
        ]
        high_immed = dft[
            (dft["ui_type"] == "immediate_adaptive")
            & (dft["cognitive_load"] == "high")
        ]

        gate_open_rate = (high_stab["gate_reason"] == "ICM_semantic_justified").mean()
        tct_mean = high_stab["task_completion_time_sec"].mean()
        tct_improve = (static_mean_tct - tct_mean) / static_mean_tct * 100.0

        # Proxy used only to visualize qualitative volatility trends
        volatility_proxy = gate_open_rate * 100.0

        sweep_rows.append(
            {
                "threshold": th,
                "gate_open_rate": gate_open_rate * 100.0,
                "tct_improvement_pct": tct_improve,
                "volatility_proxy": volatility_proxy,
                "immediate_tct": high_immed["task_completion_time_sec"].mean(),
                "stability_tct": tct_mean,
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)

    st.markdown(
        "As θₛ increases, fewer sessions qualify for adaptation, yielding greater interface stability "
        "but potentially smaller task-time gains. In this synthetic analysis, gate-closed sessions are "
        "treated as behaving comparably to the immediate-response adaptation condition for illustrative comparison."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(
            x=sweep_df["threshold"],
            y=sweep_df["gate_open_rate"],
            mode="lines+markers",
            name="Gate Open Rate",
        )
        fig.update_layout(
            title="Gate Open Rate vs. Threshold",
            xaxis_title="θₛ",
            yaxis_title="Gate Open Rate (%)",
            height=340,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            xaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_scatter(
            x=sweep_df["threshold"],
            y=sweep_df["tct_improvement_pct"],
            mode="lines+markers",
            name="Task-Time Improvement",
        )
        fig.update_layout(
            title="Task-Time Improvement vs. Threshold",
            xaxis_title="θₛ",
            yaxis_title="Improvement vs Static (%)",
            height=340,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            xaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Volatility Trend (Proxy)")
    st.caption(
        "This demo visualizes a gate-open-rate-based proxy for volatility trend inspection; "
        "the formal UVI remains the conceptual system-level metric described in the paper."
    )
    fig = go.Figure()
    fig.add_scatter(
        x=sweep_df["threshold"],
        y=sweep_df["volatility_proxy"],
        mode="lines+markers",
        name="Volatility Proxy",
    )
    fig.update_layout(
        title="Volatility Proxy vs. Threshold",
        xaxis_title="θₛ",
        yaxis_title="Proxy Value",
        height=320,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        sweep_df[
            ["threshold", "gate_open_rate", "tct_improvement_pct"]
        ].rename(
            columns={
                "threshold": "θₛ",
                "gate_open_rate": "Gate Open Rate (%)",
                "tct_improvement_pct": "TCT Improvement (%)",
            }
        ),
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 · Fitness Dashboard
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🏃 Fitness-Domain Dashboard Instance")
    st.caption(
        "Illustrative dashboard instance used to demonstrate live layout adaptation in the paper."
    )

    if adapt_live:
        st.success("Current gate decision suggests a simplified, stability-aware layout.")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.metric("Weekly Training Load", "420", "+5%")
            st.metric("Recovery Status", "Moderate", "-")

            fig = go.Figure()
            fig.add_scatter(
                y=[65, 68, 66, 72, 70, 69, 71],
                mode="lines+markers",
                name="Readiness",
            )
            fig.update_layout(
                height=260,
                title="Key Trend: Readiness",
                plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**System Action**")
            st.write("• Simplify layout")
            st.write("• Progressive disclosure")
            st.write("• Stabilize layout")
            st.caption(
                "The layout prioritizes key metrics and reduces interactive burden under high cognitive load."
            )
    else:
        st.info("Current gate decision keeps the fuller dashboard layout.")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Weekly Training Load", "420", "+5%")
            st.metric("Recovery Status", "Moderate", "-")
        with col2:
            st.metric("Sleep Score", "78", "+2")
            st.metric("HRV", "61 ms", "+4")
        with col3:
            st.metric("Run Volume", "42 km", "+6%")
            st.metric("Strength Sessions", "3", "+1")

        fig = go.Figure()
        fig.add_scatter(y=[42, 38, 45, 50, 47, 44, 42], mode="lines+markers", name="Run Volume")
        fig.add_scatter(y=[3, 2, 3, 4, 3, 3, 3], mode="lines+markers", name="Strength")
        fig.update_layout(
            height=280,
            title="Multi-Metric Dashboard View",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The fuller layout remains stable when semantic justification for adaptation is not sufficient."
        )
