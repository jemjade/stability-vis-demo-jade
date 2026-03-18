"""
Stability-Aware Adaptive Visualization Framework — Interactive Demo
Run: streamlit run app.py

Imports KnowledgeGraphSimulator and generate_sessions from simulation.py
(single source of truth — no duplicated logic).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── import from simulation.py (DRY: no duplicated KG/generation logic) ────────
from simulation import (
    KnowledgeGraphSimulator,
    generate_sessions,
    INTENTS, UIS, SEED, N_SESSIONS, GATE_DEFAULT,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stability-Aware Adaptive Visualization",
    page_icon="🧠",
    layout="wide",
)

# ── helpers ────────────────────────────────────────────────────────────────────
UI_LABELS = {
    "static":                   "Static",
    "immediate_adaptive":       "Immediate Adaptive",
    "stability_aware_adaptive": "Stability-Aware",
}
UI_COLORS = {
    "static":                   "#636EFA",
    "immediate_adaptive":       "#EF553B",
    "stability_aware_adaptive": "#00CC96",
}


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    high = df[df["cognitive_load"] == "high"]
    return high.groupby("ui_type").agg(
        task_time    =("task_completion_time_sec",  "mean"),
        comprehension=("comprehension_score_0_3",   "mean"),
        filter_cnt   =("filter_count",              "mean"),
        backtrack    =("backtrack_count",           "mean"),
        click_cnt    =("click_count",               "mean"),
        gate_open_rate=("gate_reason",
                        lambda s: float((s == "ICM_semantic_justified").mean())),
    ).reset_index()


@st.cache_data
def load_data(gate_threshold: float) -> tuple[pd.DataFrame, dict]:
    """Single call to simulation.py — no duplicated generation code."""
    df, kg = generate_sessions(seed=SEED, gate_threshold=gate_threshold)
    edges_im = [(u, v, d["weight"]) for u, v, d in kg.G.edges(data=True) if d.get("etype") == "I→M"]
    edges_mc = [(u, v, d["weight"]) for u, v, d in kg.G.edges(data=True) if d.get("etype") == "M→C"]
    return df, {"edges_im": edges_im, "edges_mc": edges_mc, "kg": kg}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Parameters")
threshold = st.sidebar.slider(
    "Semantic Gate Threshold (θₛ)", 0.30, 0.90, 0.51, 0.05,
    help="ICM subgraph score must exceed this to allow adaptation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Live Gate Inspector")
sel_intent = st.sidebar.selectbox("User Intent", INTENTS, index=1)
sel_load   = st.sidebar.selectbox("Cognitive Load", ["high", "low"], index=0)
st.sidebar.markdown("**Metric Signals (0–1)**")
sig_dwell  = st.sidebar.slider("dwell_time",    0.0, 1.0, 0.70, 0.05)
sig_click  = st.sidebar.slider("click_rate",    0.0, 1.0, 0.65, 0.05)
sig_filt   = st.sidebar.slider("filtering",     0.0, 1.0, 0.55, 0.05)
sig_back   = st.sidebar.slider("backtracking",  0.0, 1.0, 0.60, 0.05)
sig_eff    = st.sidebar.slider("efficiency",    0.0, 1.0, 0.50, 0.05)
sig_comp   = st.sidebar.slider("comprehension", 0.0, 1.0, 0.70, 0.05)

live_sigs = {
    "dwell_time":    sig_dwell, "click_rate":   sig_click,
    "filtering":     sig_filt,  "backtracking": sig_back,
    "efficiency":    sig_eff,   "comprehension": sig_comp,
}

# ── load data & live gate ──────────────────────────────────────────────────────
df, kg_meta = load_data(gate_threshold=threshold)
kg_live = KnowledgeGraphSimulator(rng=np.random.default_rng(SEED), gate_threshold=threshold)
adapt_live, _, score_live, debug_live = kg_live.should_adapt(sel_intent, sel_load, live_sigs)
actions_live = kg_live.get_action_recommendations(live_sigs, topk=2)
summ = summary_table(df)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Stability-Aware Adaptive Visualization Framework")

st.markdown(
    f"**HCII 2026 Poster Demo** · Suhyun Park · Graduate School of SW and AI Convergence, Korea University  \n"
    f"Synthetic sessions: **N = {N_SESSIONS}** · Seed: {SEED} · Preliminary ML consistency: **~94%**"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 UI Comparison",
    "🔍 Live Gate Inspector",
    "🕸️ KG Subgraph",
    "📈 Threshold Sweep",
    "🏃 Fitness Dashboard",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · UI Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Performance by UI Type — High Cognitive Load Sessions")

    ui_order  = ["static", "immediate_adaptive", "stability_aware_adaptive"]

    def get(col): return [summ[summ.ui_type == u][col].values[0] for u in ui_order]

    tct_vals  = get("task_time")
    comp_vals = get("comprehension")
    static_tct = tct_vals[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        delta_str = f"-{(static_tct - tct_vals[2]) / static_tct * 100:.1f}% vs static"
        st.metric("Task Time · Stability-Aware", f"{tct_vals[2]:.1f}s", delta_str, delta_color="inverse")
    with c2:
        delta_comp = f"+{(comp_vals[2] - comp_vals[0]) / comp_vals[0] * 100:.1f}% vs static"
        st.metric("Comprehension · Stability-Aware", f"{comp_vals[2]:.2f}", delta_comp)
    with c3:
        gate_row = summ[summ.ui_type == "stability_aware_adaptive"].iloc[0]
        st.metric("Gate Open Rate", f"{gate_row.gate_open_rate * 100:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        for ui, color in UI_COLORS.items():
            val = summ[summ.ui_type == ui]["task_time"].values[0]
            fig.add_bar(x=[UI_LABELS[ui]], y=[val], name=UI_LABELS[ui],
                        marker_color=color, showlegend=False,
                        text=[f"{val:.1f}s"], textposition="outside")
        fig.update_layout(title="Task Completion Time (sec)", yaxis_title="seconds",
                          height=350, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        for ui, color in UI_COLORS.items():
            val = summ[summ.ui_type == ui]["comprehension"].values[0]
            fig.add_bar(x=[UI_LABELS[ui]], y=[val], name=UI_LABELS[ui],
                        marker_color=color, showlegend=False,
                        text=[f"{val:.2f}"], textposition="outside")
        fig.update_layout(title="Comprehension Score (0–3)", yaxis_title="score",
                          yaxis_range=[0, 3.2], height=350, plot_bgcolor="white",
                          yaxis=dict(gridcolor="#eee"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Interaction Burden (High Load)")
    burden_metrics = ["filter_cnt", "backtrack", "click_cnt"]
    burden_labels  = ["Filter Count", "Backtrack Count", "Click Count"]
    fig2 = go.Figure()
    for ui, color in UI_COLORS.items():
        row = summ[summ.ui_type == ui].iloc[0]
        fig2.add_bar(name=UI_LABELS[ui], x=burden_labels,
                     y=[row[m] for m in burden_metrics], marker_color=color)
    fig2.update_layout(barmode="group", height=350, plot_bgcolor="white",
                       yaxis=dict(gridcolor="#eee"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Gate Open Rate by Intent (Stability-Aware)")
    high_stab = df[(df["cognitive_load"] == "high") &
                   (df["ui_type"] == "stability_aware_adaptive")].copy()
    high_stab["gate_open"] = (high_stab["gate_reason"] == "ICM_semantic_justified").astype(int)
    gor = high_stab.groupby("intent")["gate_open"].mean().reindex(INTENTS) * 100
    fig3 = px.bar(x=gor.index, y=gor.values,
                  text=[f"{v:.1f}%" for v in gor.values],
                  color_discrete_sequence=["#00CC96"],
                  labels={"x": "Intent", "y": "Gate Open Rate (%)"})
    fig3.update_traces(textposition="outside")
    fig3.update_layout(height=320, plot_bgcolor="white", yaxis=dict(gridcolor="#eee"))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Live Gate Inspector
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Live Semantic Gate Inspector")
    st.markdown("Adjust metric signals in the sidebar to see real-time gate decisions.")

    col1, col2 = st.columns([1, 2])
    with col1:
        if adapt_live:
            st.success(f"✅ **GATE OPEN** — Adaptation triggered  \nICM Score: **{score_live:.3f}** ≥ θ={threshold:.2f}")
        else:
            st.error(f"🔒 **GATE CLOSED** — No adaptation  \nICM Score: **{score_live:.3f}** < θ={threshold:.2f}")
        st.markdown(f"**Intent:** `{sel_intent}`  \n**Cognitive Load:** `{sel_load}`")
        if adapt_live and actions_live:
            st.markdown("**Recommended Actions:**")
            for action, score in actions_live:
                st.markdown(f"- `{action}` (score: {score:.3f})")

    with col2:
        if debug_live:
            rows_debug = [{"Metric": m, "Signal": info["signal"], "w(I→M)": info["w(I→M)"],
                           "w(M→C)": info["w(M→C)"], "Contribution": info["contrib"]}
                          for m, info in debug_live.items()]
            df_debug = pd.DataFrame(rows_debug).sort_values("Contribution", ascending=False)
            fig_bar = px.bar(df_debug, x="Metric", y="Contribution",
                             color="Contribution", color_continuous_scale="Teal",
                             text=df_debug["Contribution"].apply(lambda x: f"{x:.4f}"),
                             title="Per-Metric ICM Contribution")
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(height=340, plot_bgcolor="white",
                                  coloraxis_showscale=False, yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(
                df_debug.style.format({"Signal":"{:.3f}","w(I→M)":"{:.3f}",
                                       "w(M→C)":"{:.3f}","Contribution":"{:.4f}"}),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · KG Subgraph
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🕸️ ICM Subgraph Visualization")
    st.markdown("Intent–Cognitive State–Metric subgraph for the selected intent.")

    kg_viz     = KnowledgeGraphSimulator(rng=np.random.default_rng(SEED), gate_threshold=threshold)
    viz_intent = st.selectbox("Intent to visualize", INTENTS, index=1, key="kg_intent")
    viz_cog    = st.radio("Cognitive State", ["high", "low"], horizontal=True)

    i_node  = f"I:{viz_intent}"; c_node = f"C:{viz_cog}"
    m_nodes = [v for u, v in kg_viz.G.out_edges(i_node)
               if kg_viz.G.nodes[v].get("ntype") == "M"]
    H = kg_viz.G.subgraph([i_node, c_node] + m_nodes).copy()

    pos = {i_node: (0, 2), c_node: (4, 2)}
    for idx, m in enumerate(sorted(m_nodes)):
        pos[m] = (2, idx * 0.8)

    edge_x, edge_y, edge_text_x, edge_text_y, edge_weights = [], [], [], [], []
    for u, v, d in H.edges(data=True):
        if d.get("etype") not in ("I→M", "M→C"): continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_text_x.append((x0+x1)/2); edge_text_y.append((y0+y1)/2)
        edge_weights.append(f"{d['weight']:.2f}")

    node_x = [pos[n][0] for n in H.nodes()]
    node_y = [pos[n][1] for n in H.nodes()]
    ntype_color = {"U": "#FFA15A", "I": "#636EFA", "E": "#19D3F3",
                   "C": "#EF553B", "M": "#00CC96", "A": "#AB63FA"}
    node_colors = [ntype_color.get(H.nodes[n].get("ntype","M"),"#aaa") for n in H.nodes()]
    node_labels = [n.split(":")[1].replace("_","\n") if ":" in n else n for n in H.nodes()]

    fig_kg = go.Figure()
    fig_kg.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(color="#aaa", width=2), hoverinfo="none"))
    fig_kg.add_trace(go.Scatter(x=edge_text_x, y=edge_text_y, mode="text",
                                text=edge_weights, textfont=dict(size=11, color="#555"),
                                hoverinfo="none"))
    fig_kg.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                marker=dict(size=38, color=node_colors,
                                            line=dict(color="black", width=1.5)),
                                text=node_labels, textposition="middle center",
                                textfont=dict(size=9, color="white"),
                                hovertext=list(H.nodes()), hoverinfo="text"))
    fig_kg.update_layout(
        title=f"ICM Subgraph · Intent={viz_intent} · Cog={viz_cog}",
        showlegend=False, height=420,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_kg, use_container_width=True)

    legend_cols = st.columns(6)
    for col, (_, color, label) in zip(legend_cols, [
        ("U","#FFA15A","User (U)"),
        ("I","#636EFA","Intent (I)"),
        ("E","#19D3F3","Event (E)"),
        ("M","#00CC96","Metric (M)"),
        ("C","#EF553B","Cognitive (C)"),
        ("A","#AB63FA","Action (A)"),
    ]):
        col.markdown(f'<span style="background:{color};padding:3px 10px;border-radius:4px;'
                     f'color:white;font-size:12px">{label}</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 Stability–Performance Tradeoff (Threshold Sweep)")
    st.markdown(
        "As θₛ increases, fewer sessions qualify for adaptation, yielding greater interface stability "
        "but potentially smaller task-time gains. In this simulation, gate-closed sessions revert to "
        "the immediate-adaptive baseline, consistent with the stability--performance trade-off discussed "
        "in prior adaptive interface research."
    )

    high_df = df[df["cognitive_load"] == "high"].copy()
    static_tct_mean   = float(high_df[high_df["ui_type"] == "static"]["task_completion_time_sec"].mean())
    immediate_tct_mean = float(high_df[high_df["ui_type"] == "immediate_adaptive"]["task_completion_time_sec"].mean())
    stab_df = high_df[high_df["ui_type"] == "stability_aware_adaptive"].dropna(subset=["gate_score"])

    thresholds = np.arange(0.30, 0.91, 0.05)
    curve_rows = []
    for t in thresholds:
        mask    = (stab_df["gate_score"] >= t)
        n_total = len(stab_df); n_open = int(mask.sum())
        open_rate  = float(mask.mean())
        tct_open   = stab_df["task_completion_time_sec"][mask].mean() if n_open > 0 else static_tct_mean
        # gate-closed sessions: no stability-aware adaptation → immediate_adaptive baseline
        tct_mean   = (n_open * tct_open + (n_total - n_open) * immediate_tct_mean) / n_total
        improve    = (static_tct_mean - tct_mean) / static_tct_mean * 100.0
        curve_rows.append({
            "threshold":         round(t, 2),
            "gate_open_rate_pct": round(open_rate * 100, 1),
            "tct_improve_pct":   round(improve, 1),
            "uvi_proxy":         round(open_rate * 100, 1),
        })
    curve_df = pd.DataFrame(curve_rows)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(
            x=curve_df["gate_open_rate_pct"], y=curve_df["tct_improve_pct"],
            mode="lines+markers+text",
            text=curve_df["threshold"].astype(str), textposition="top center",
            textfont=dict(size=10), line=dict(color="#00CC96", width=2.5), marker=dict(size=8),
        ))
        cur = curve_df[curve_df["threshold"] == round(threshold, 2)]
        if not cur.empty:
            fig_sw.add_vline(x=float(cur["gate_open_rate_pct"].values[0]),
                             line_dash="dash", line_color="red",
                             annotation_text=f"current θ={threshold:.2f}")
        fig_sw.update_layout(
            title="Gate Open Rate vs. Task Time Improvement",
            xaxis_title="Gate Open Rate (%) ← lower = more stable",
            yaxis_title="Task Time Improvement vs Static (%)",
            height=380, plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_sw, use_container_width=True)

    with col_b:
        fig_uvi = go.Figure()
        fig_uvi.add_trace(go.Scatter(
            x=curve_df["threshold"], y=curve_df["uvi_proxy"],
            mode="lines+markers", line=dict(color="#636EFA", width=2.5), marker=dict(size=7),
        ))
        fig_uvi.add_vline(x=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"θ={threshold:.2f}")
        fig_uvi.update_layout(
            title="UI Volatility Proxy vs. Threshold",
            xaxis_title="Threshold (θₛ)",
            yaxis_title="UVI proxy (gate open rate × 100)",
            height=380, plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_uvi, use_container_width=True)

    st.dataframe(curve_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · Fitness Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab5:

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                padding:28px 32px;border-radius:12px;margin-bottom:24px'>
        <div style='color:#e94560;font-size:11px;font-weight:700;letter-spacing:3px;
                    text-transform:uppercase;margin-bottom:6px'>STABILITY-AWARE ADAPTATION DEMO</div>
        <div style='color:#ffffff;font-size:26px;font-weight:800;margin-bottom:4px'>
            🏃 Fitness Performance Hub
        </div>
        <div style='color:#a8b2c1;font-size:13px'>
            Synthetic Strava-format data · 12-week training block · fitness-domain scenario
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── synthetic data ────────────────────────────────────────────────────────
    @st.cache_data
    def make_fitness_data(seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = 12
        weeks = [f"W{i+1}" for i in range(n)]
        distance_km   = rng.normal(45, 10, n).clip(20, 80)
        avg_hr        = rng.normal(148, 8,  n).clip(128, 175)
        training_load = distance_km * (avg_hr / 140)
        vo2max_proxy  = 55 - training_load * 0.04 + rng.normal(0, 0.8, n)
        sleep_hrs     = rng.normal(7.1, 0.6, n).clip(5.5, 9.0)
        hrv           = rng.normal(62, 9, n).clip(40, 90)
        return pd.DataFrame({
            "week": weeks,
            "distance_km":   distance_km.round(1),
            "avg_hr":        avg_hr.round(0).astype(int),
            "training_load": training_load.round(1),
            "vo2max":        vo2max_proxy.round(1),
            "sleep_hrs":     sleep_hrs.round(1),
            "hrv":           hrv.round(0).astype(int),
        })

    fdf = make_fitness_data()
    latest = fdf.iloc[-1]; prev = fdf.iloc[-2]

    def dstr(col, fmt=".1f", invert=False):
        d = latest[col] - prev[col]
        sign = ("+" if d > 0 else "") if not invert else ("-" if d > 0 else "+")
        return f"{sign}{abs(d):{fmt}} vs W{len(fdf)-1}"

    # ── independent controls ──────────────────────────────────────────────────
    st.markdown("##### 🎛️ Simulate User Context")
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    with cc1:
        fit_intent = st.selectbox("Monitoring Intent", INTENTS,
            format_func=lambda x: {
                "fast_concise":    "⚡ Fast · Concise",
                "fast_detailed":   "⚡ Fast · Detailed",
                "gradual_concise": "🔍 Gradual · Concise",
                "gradual_detailed":"🔍 Gradual · Detailed",
            }[x], key="fit_intent2")
    with cc2:
        fit_load = st.radio("Cognitive Load", ["low", "high"],
            horizontal=True, key="fit_load2",
            help="High = many clicks + long dwell + backtracking")
    with cc3:
        fit_focus = st.selectbox("Focus Metric", ["training_load","distance_km","avg_hr","vo2max","hrv"],
            format_func=lambda x: {
                "training_load":"🏋 Training Load","distance_km":"📏 Distance",
                "avg_hr":"❤️ Avg HR","vo2max":"🫁 VO₂max","hrv":"💤 HRV",
            }[x], key="fit_focus2")
    with cc4:
        fit_theta = st.slider("θ gate", 0.30, 0.80, float(GATE_DEFAULT), 0.01, key="fit_theta")

    # ── independent gate computation (no sidebar dependency) ─────────────────
    fit_dwell   = 10.5 if fit_load == "high" else 5.0
    fit_click   = 13   if fit_load == "high" else 6
    fit_filter  = 7    if fit_load == "high" else 3
    fit_back    = 3    if fit_load == "high" else 1
    fit_eff     = (fit_dwell * 10) / max(1, fit_click)

    kg_fit = KnowledgeGraphSimulator(rng=np.random.default_rng(SEED), gate_threshold=fit_theta)
    fit_sigs = {
        "dwell_time":    kg_fit._normalize(fit_dwell,  2.0, 16.0),
        "click_rate":    kg_fit._normalize(fit_click,  2.0, 22.0),
        "filtering":     kg_fit._normalize(fit_filter, 0.0, 14.0),
        "backtracking":  kg_fit._normalize(fit_back,   0.0,  8.0),
        "efficiency":    kg_fit._normalize(fit_eff,    3.0, 18.0),
        "comprehension": 1.0 - kg_fit._normalize(
            {"training_load":1.4,"distance_km":2.1,"avg_hr":1.8,
             "vo2max":1.2,"hrv":1.6}[fit_focus], 0.0, 3.0),
    }
    adapt_fit, _, score_fit, debug_fit = kg_fit.should_adapt(fit_intent, fit_load, fit_sigs)

    # ── gate status bar ───────────────────────────────────────────────────────
    st.markdown("---")
    sb1, sb2 = st.columns([1, 3])
    with sb1:
        if adapt_fit:
            st.markdown(f"""
            <div style='background:#0d5c3a;border:2px solid #00CC96;border-radius:10px;
                        padding:16px;text-align:center'>
                <div style='color:#00CC96;font-size:28px'>🟢</div>
                <div style='color:#00CC96;font-weight:800;font-size:16px'>GATE OPEN</div>
                <div style='color:#aaa;font-size:12px;margin-top:4px'>
                    score {score_fit:.3f} ≥ θ {fit_theta:.2f}</div>
                <div style='color:#ccc;font-size:11px;margin-top:6px'>Layout adapting ↓</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#1a2a4a;border:2px solid #636EFA;border-radius:10px;
                        padding:16px;text-align:center'>
                <div style='color:#636EFA;font-size:28px'>🔵</div>
                <div style='color:#636EFA;font-weight:800;font-size:16px'>GATE CLOSED</div>
                <div style='color:#aaa;font-size:12px;margin-top:4px'>
                    score {score_fit:.3f} &lt; θ {fit_theta:.2f}</div>
                <div style='color:#ccc;font-size:11px;margin-top:6px'>Layout stable</div>
            </div>""", unsafe_allow_html=True)
    with sb2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_fit,
            number={"font":{"size":32}, "valueformat":".3f"},
            gauge={
                "axis":  {"range":[0,1], "tickwidth":1, "tickcolor":"#555"},
                "bar":   {"color":"#00CC96" if adapt_fit else "#636EFA", "thickness":0.25},
                "bgcolor":"#1e1e2e",
                "steps": [
                    {"range":[0, fit_theta], "color":"#1a2a4a"},
                    {"range":[fit_theta, 1], "color":"#0d3a28"},
                ],
                "threshold":{"line":{"color":"#e94560","width":3},
                             "thickness":0.85,"value":fit_theta},
            },
            title={"text":"ICM Semantic Gate Score","font":{"size":13,"color":"#aaa"}},
        ))
        fig_g.update_layout(height=170, paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#ccc", margin=dict(t=35,b=5,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    # ── dashboard ─────────────────────────────────────────────────────────────
    st.markdown("---")

    METRIC_META = {
        "training_load": ("🏋 Training Load", "AU",       "training_load"),
        "distance_km":   ("📏 Distance",       "km",       "distance_km"),
        "avg_hr":        ("❤️ Avg HR",          "bpm",      "avg_hr"),
        "vo2max":        ("🫁 VO₂max",          "ml/kg/min","vo2max"),
        "hrv":           ("💤 HRV",             "ms",       "hrv"),
    }
    label_focus, unit_focus, col_focus = METRIC_META[fit_focus]

    if adapt_fit:
        # ── GATE OPEN: simplified layout ──────────────────────────────────────
        st.markdown(
            f"<div style='background:#0d3a28;border-left:4px solid #00CC96;"
            f"padding:10px 16px;border-radius:6px;margin-bottom:16px;color:#ccc'>"
            f"🟢 <b style='color:#00CC96'>Stability-Aware Adaptation Active</b> &nbsp;·&nbsp; "
            f"Layout simplified for <code>{fit_intent}</code> intent. "
            f"Non-critical panels hidden to reduce extraneous cognitive load.</div>",
            unsafe_allow_html=True)

        # 3 focused KPIs only
        km1, km2, km3 = st.columns(3)
        km1.metric(label_focus,          f"{latest[col_focus]} {unit_focus}", dstr(col_focus))
        km2.metric("🔋 HRV (Recovery)",  f"{latest.hrv} ms",    dstr("hrv"))
        km3.metric("😴 Sleep",           f"{latest.sleep_hrs}h", dstr("sleep_hrs"))

        # single focused trend chart
        accent = "#00CC96"
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=fdf["week"], y=fdf[col_focus],
            mode="lines+markers",
            line=dict(color=accent, width=3),
            marker=dict(size=9, color=accent,
                        line=dict(color="#fff", width=1.5)),
            fill="tozeroy", fillcolor="rgba(0,204,150,0.10)",
            name=label_focus,
        ))
        fig_f.update_layout(
            title=dict(text=f"{label_focus} — 12-Week Trend  <span style='font-size:12px;color:#aaa'>"
                            f"(Simplified · Gate Open)</span>",
                       font=dict(size=15)),
            height=300, plot_bgcolor="#0d1117", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#2a2a3e", color="#888"),
            yaxis=dict(gridcolor="#2a2a3e", color="#888"),
            font_color="#ccc", margin=dict(t=45,b=20,l=10,r=10),
        )
        st.plotly_chart(fig_f, use_container_width=True)
        st.caption("ℹ️ Non-critical panels hidden by stability-aware adaptation.")

    else:
        # ── GATE CLOSED: full dashboard ────────────────────────────────────────
        st.markdown(
            f"<div style='background:#111827;border-left:4px solid #636EFA;"
            f"padding:10px 16px;border-radius:6px;margin-bottom:16px;color:#ccc'>"
            f"🔵 <b style='color:#636EFA'>Full Dashboard — Stable Layout</b> &nbsp;·&nbsp; "
            f"Semantic score {score_fit:.3f} below θ={fit_theta:.2f}. "
            f"No adaptation triggered — layout unchanged.</div>",
            unsafe_allow_html=True)

        # 5 KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📏 Distance",      f"{latest.distance_km} km", dstr("distance_km"))
        k2.metric("❤️ Avg HR",        f"{latest.avg_hr} bpm",     dstr("avg_hr", invert=True))
        k3.metric("🏋 Training Load", f"{latest.training_load}",  dstr("training_load"))
        k4.metric("🫁 VO₂max",        f"{latest.vo2max}",         dstr("vo2max"))
        k5.metric("💤 HRV",           f"{latest.hrv} ms",         dstr("hrv"))

        BG = "#0d1117"; GRID = "#2a2a3e"

        # row 1: Distance bar + HR/HRV dual axis
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fd = go.Figure()
            fd.add_bar(x=fdf["week"], y=fdf["distance_km"],
                       marker=dict(color=fdf["distance_km"],
                                   colorscale="Blues", showscale=False),
                       name="Distance (km)")
            fd.update_layout(title="Weekly Distance (km)", height=260,
                             plot_bgcolor=BG, paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(gridcolor=GRID,color="#888"),
                             yaxis=dict(gridcolor=GRID,color="#888"),
                             font_color="#ccc", margin=dict(t=40,b=10))
            st.plotly_chart(fd, use_container_width=True)

        with r1c2:
            fh = go.Figure()
            fh.add_trace(go.Scatter(x=fdf["week"], y=fdf["avg_hr"],
                                    mode="lines+markers", name="Avg HR",
                                    line=dict(color="#EF553B", width=2.5),
                                    marker=dict(size=6)))
            fh.add_trace(go.Scatter(x=fdf["week"], y=fdf["hrv"],
                                    mode="lines+markers", name="HRV (ms)",
                                    line=dict(color="#AB63FA", width=2.5, dash="dot"),
                                    marker=dict(size=6), yaxis="y2"))
            fh.update_layout(
                title="Heart Rate & HRV", height=260,
                plot_bgcolor=BG, paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=GRID, color="#888"),
                yaxis=dict(title="HR (bpm)", gridcolor=GRID, color="#EF553B"),
                yaxis2=dict(title="HRV (ms)", overlaying="y", side="right", color="#AB63FA"),
                legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
                font_color="#ccc", margin=dict(t=40,b=10),
            )
            st.plotly_chart(fh, use_container_width=True)

        # row 2: Training Load area + Sleep bar
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fl = go.Figure()
            fl.add_trace(go.Scatter(
                x=fdf["week"], y=fdf["training_load"],
                mode="lines+markers",
                line=dict(color="#00CC96", width=2.5),
                marker=dict(size=7, color="#00CC96"),
                fill="tozeroy", fillcolor="rgba(0,204,150,0.12)",
                name="Training Load",
            ))
            fl.update_layout(title="Training Load (AU)", height=240,
                             plot_bgcolor=BG, paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(gridcolor=GRID,color="#888"),
                             yaxis=dict(gridcolor=GRID,color="#888"),
                             font_color="#ccc", margin=dict(t=40,b=10))
            st.plotly_chart(fl, use_container_width=True)

        with r2c2:
            sleep_colors = ["#EF553B" if h < 6.5 else "#00CC96" if h >= 7.5 else "#FFA15A"
                            for h in fdf["sleep_hrs"]]
            fs = go.Figure()
            fs.add_bar(x=fdf["week"], y=fdf["sleep_hrs"],
                       marker_color=sleep_colors, name="Sleep (hrs)")
            fs.add_hline(y=7.5, line_dash="dash", line_color="#636EFA",
                         annotation_text="Target 7.5h",
                         annotation_font_color="#636EFA")
            fs.update_layout(title="Sleep Duration (hrs)", height=240,
                             plot_bgcolor=BG, paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(gridcolor=GRID,color="#888"),
                             yaxis=dict(gridcolor=GRID,color="#888"),
                             font_color="#ccc", margin=dict(t=40,b=10))
            st.plotly_chart(fs, use_container_width=True)

    # ── gate logic explainer ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("💡 What just happened? (Gate logic explained)"):
        st.markdown(f"""
**Gate score:** `{score_fit:.3f}` &nbsp;vs&nbsp; threshold `θ = {fit_theta:.2f}`

**Decision:** `{"ADAPT — layout simplified" if adapt_fit else "STABLE — full layout retained"}`

The ICM subgraph traverses **Intent → Metric → Cognitive State** paths in the KG.
A score ≥ θ means high-load signals are semantically aligned with monitoring intent → adaptation warranted.
A score < θ means signals may reflect transient exploration → layout change suppressed.

**Metric signal contributions:**
        """)
        if debug_fit:
            rows_d = [{"Metric": m, "Signal": v["signal"],
                       "w(I→M)": v["w(I→M)"], "Contribution": v["contrib"]}
                      for m, v in debug_fit.items()]
            st.dataframe(
                pd.DataFrame(rows_d).sort_values("Contribution", ascending=False)
                  .style.format({"Signal":"{:.3f}","w(I→M)":"{:.3f}","Contribution":"{:.4f}"}),
                use_container_width=True)


st.markdown("---")
st.caption(
    f"This demo is based entirely on synthetic interaction sessions (N={N_SESSIONS}). "
    "No real participants or personal data were involved. "
    "Human-subject evaluation is planned as future work. "
    "HCII 2026 · Suhyun Park · Graduate School of SW and AI Convergence, Korea University"
)
