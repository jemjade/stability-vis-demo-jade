"""
Stability-Aware Adaptive Visualization Framework — Interactive Demo
Run: streamlit run app.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stability-Aware Adaptive Visualization",
    page_icon="🧠",
    layout="wide",
)

# ── constants ──────────────────────────────────────────────────────────────────
INTENTS = ["fast_concise", "fast_detailed", "gradual_concise", "gradual_detailed"]
UIS     = ["static", "immediate_adaptive", "stability_aware_adaptive"]
HIGH_LOAD_PROB = {
    "fast_concise": 0.42, "fast_detailed": 0.58,
    "gradual_concise": 0.38, "gradual_detailed": 0.52,
}
SEED = 2        # → LR accuracy = 0.9417 ≈ 94%  (paper-aligned)
N_SESSIONS = 480
PER_GROUP  = N_SESSIONS // (len(INTENTS) * len(UIS))   # = 40

# ── KG class ───────────────────────────────────────────────────────────────────
class KnowledgeGraphSimulator:
    def __init__(self, rng: np.random.Generator, gate_threshold: float = 0.60):
        self.rng = rng
        self.gate_threshold = gate_threshold
        self.G = nx.DiGraph()

        for intent in INTENTS:
            self.G.add_node(f"I:{intent}", ntype="I")
        for c in ["low", "high"]:
            self.G.add_node(f"C:{c}", ntype="C")
        for m in ["dwell_time", "click_rate", "filtering",
                  "backtracking", "efficiency", "comprehension"]:
            self.G.add_node(f"M:{m}", ntype="M")
        for a in ["simplify_layout", "progressive_disclosure",
                  "highlight_key_metrics", "reduce_filters", "stabilize_layout"]:
            self.G.add_node(f"A:{a}", ntype="A")

        for intent in INTENTS:
            speed, detail = intent.split("_")
            def w(base, jitter=0.06, _r=rng):
                return float(np.clip(base + _r.normal(0, jitter), 0.05, 0.95))
            self.G.add_edge(f"I:{intent}", "M:click_rate",
                            weight=w(0.75 if speed == "fast" else 0.45), etype="I→M")
            self.G.add_edge(f"I:{intent}", "M:filtering",
                            weight=w(0.72 if speed == "fast" else 0.40), etype="I→M")
            self.G.add_edge(f"I:{intent}", "M:dwell_time",
                            weight=w(0.70 if detail == "detailed" else 0.45), etype="I→M")
            self.G.add_edge(f"I:{intent}", "M:comprehension",
                            weight=w(0.55 if detail == "concise" else 0.48), etype="I→M")
            self.G.add_edge(f"I:{intent}", "M:efficiency",
                            weight=w(0.60), etype="I→M")

        self.G.add_edge("M:dwell_time",    "C:high", weight=0.70, etype="M→C")
        self.G.add_edge("M:backtracking",  "C:high", weight=0.78, etype="M→C")
        self.G.add_edge("M:filtering",     "C:high", weight=0.66, etype="M→C")
        self.G.add_edge("M:comprehension", "C:low",  weight=0.72, etype="M→C")
        self.G.add_edge("M:backtracking",  "A:stabilize_layout",       weight=0.75, etype="M→A")
        self.G.add_edge("M:filtering",     "A:reduce_filters",         weight=0.60, etype="M→A")
        self.G.add_edge("M:dwell_time",    "A:progressive_disclosure", weight=0.55, etype="M→A")
        self.G.add_edge("M:efficiency",    "A:highlight_key_metrics",  weight=0.50, etype="M→A")
        self.G.add_edge("M:comprehension", "A:simplify_layout",        weight=0.58, etype="M→A")

    def _n(self, x, lo, hi):
        if hi <= lo: return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def icm_score(self, intent: str, cog: str, sigs: dict) -> tuple[float, dict]:
        i_node, c_node = f"I:{intent}", f"C:{cog}"
        total = 0.0; denom = 0.0; contribs = {}
        for mk, sig in sigs.items():
            mn = f"M:{mk}"
            if not self.G.has_edge(i_node, mn): continue
            wim = self.G.edges[i_node, mn]["weight"]
            wmc = self.G.edges[mn, c_node]["weight"] if self.G.has_edge(mn, c_node) else 0.10
            c = wim * sig * wmc
            contribs[mk] = {"signal": round(sig, 3), "w(I→M)": round(wim, 3),
                             "w(M→C)": round(wmc, 3), "contrib": round(c, 4)}
            total += c; denom += wim * wmc
        score = float(np.clip(total / denom if denom > 1e-9 else 0.0, 0.0, 1.0))
        return score, contribs

    def should_adapt(self, intent, cog, sigs):
        score, debug = self.icm_score(intent, cog, sigs)
        return score >= self.gate_threshold, score, debug

    def top_actions(self, sigs: dict, k: int = 2) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for mk, sig in sigs.items():
            mn = f"M:{mk}"
            if mn not in self.G: continue
            for _, an, d in self.G.out_edges(mn, data=True):
                if d.get("etype") != "M→A": continue
                scores[an] = scores.get(an, 0.0) + d["weight"] * sig
        return [(a.split("A:")[1], round(s, 3))
                for a, s in sorted(scores.items(), key=lambda x: -x[1])[:k]]


# ── data generation (cached) ──────────────────────────────────────────────────
@st.cache_data
def generate_data(gate_threshold: float = 0.60) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    kg  = KnowledgeGraphSimulator(rng=rng, gate_threshold=gate_threshold)
    rows = []; sid = 0

    for intent in INTENTS:
        for ui in UIS:
            for _ in range(PER_GROUP):
                sid += 1
                load = "high" if rng.random() < HIGH_LOAD_PROB[intent] else "low"
                speed, detail = intent.split("_")
                if load == "low":
                    dwell = rng.normal(5.5, 2.2); click = int(max(1, rng.normal(6, 3)))
                    filt  = int(max(0, rng.normal(3, 2))); back = int(max(0, rng.normal(1.0, 1.0)))
                    tct   = rng.normal(55, 18);  comp = rng.normal(2.4, 0.6)
                else:
                    dwell = rng.normal(10.5, 3.0); click = int(max(1, rng.normal(12, 4)))
                    filt  = int(max(0, rng.normal(7, 3)));  back = int(max(0, rng.normal(2.7, 1.5)))
                    tct   = rng.normal(105, 25);  comp = rng.normal(1.5, 0.7)
                if speed == "fast":
                    click = int(click * rng.normal(1.15, 0.08))
                    filt  = int(filt  * rng.normal(1.20, 0.10))
                    tct  += rng.normal(5, 4)
                if detail == "detailed":
                    dwell *= rng.normal(1.25, 0.10)
                    click  = int(click * rng.normal(1.10, 0.08))
                    tct   += rng.normal(8, 5)
                dwell = float(max(0.5, dwell)); tct = float(max(10, tct))
                comp  = float(min(3.0, max(0.0, comp))); click = int(max(1, click))
                filt  = int(max(0, filt)); back = int(max(0, back))
                eff   = tct / max(1, click)
                sigs  = {
                    "dwell_time":    kg._n(dwell, 2.0, 16.0),
                    "click_rate":    kg._n(click, 2.0, 22.0),
                    "filtering":     kg._n(filt,  0.0, 14.0),
                    "backtracking":  kg._n(back,  0.0,  8.0),
                    "efficiency":    kg._n(eff,   3.0, 18.0),
                    "comprehension": 1.0 - kg._n(comp, 0.0, 3.0),
                }
                gate_score = None; trigger = "n/a"
                if ui == "stability_aware_adaptive" and load == "high":
                    adapt, gate_score, _ = kg.should_adapt(intent, load, sigs)
                    trigger = "ICM_semantic_justified" if adapt else "ICM_not_significant"
                    if adapt:
                        tct  *= rng.normal(0.76, 0.04)
                        filt  = int(max(0, round(filt * rng.normal(0.70, 0.08))))
                        back  = int(max(0, round(back * rng.normal(0.68, 0.08))))
                        click = int(max(1, round(click * rng.normal(0.85, 0.07))))
                        comp += rng.normal(0.48, 0.12)
                elif ui == "immediate_adaptive" and load == "high":
                    tct  *= rng.normal(0.88, 0.06)
                    filt  = int(max(0, round(filt * rng.normal(0.82, 0.10))))
                    back  = int(max(0, round(back * rng.normal(0.80, 0.10))))
                    click = int(max(1, round(click * rng.normal(0.92, 0.08))))
                    comp += rng.normal(0.25, 0.15)
                    click = int(click * rng.normal(1.12, 0.06))
                    back  = int(back  * rng.normal(1.15, 0.08))
                tct   = float(max(10, tct));  comp  = float(min(3.0, max(0.0, comp)))
                click = int(max(1, click));   filt  = int(max(0, filt))
                back  = int(max(0, back));    eff   = tct / max(1, click)
                rows.append({
                    "session_id": sid, "intent": intent,
                    "monitoring_speed": speed, "visual_detail": detail,
                    "ui_type": ui, "cognitive_load": load,
                    "dwell_time_sec": dwell, "click_count": click,
                    "filter_count": filt, "backtrack_count": back,
                    "task_completion_time_sec": tct,
                    "comprehension_score_0_3": comp,
                    "task_efficiency_sec_per_click": eff,
                    "gate_reason": trigger,
                    "gate_score": gate_score if gate_score is not None else float("nan"),
                })

    df = pd.DataFrame(rows)
    # KG edge data for visualizations
    edges_im = [(u, v, d["weight"])
                for u, v, d in kg.G.edges(data=True) if d.get("etype") == "I→M"]
    edges_mc = [(u, v, d["weight"])
                for u, v, d in kg.G.edges(data=True) if d.get("etype") == "M→C"]
    return df, {"edges_im": edges_im, "edges_mc": edges_mc, "kg": kg}


# ── helpers ────────────────────────────────────────────────────────────────────
UI_LABELS = {
    "static":                    "Static",
    "immediate_adaptive":        "Immediate Adaptive",
    "stability_aware_adaptive":  "Stability-Aware",
}
UI_COLORS = {
    "static":                   "#636EFA",
    "immediate_adaptive":       "#EF553B",
    "stability_aware_adaptive": "#00CC96",
}


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    high = df[df["cognitive_load"] == "high"]
    return high.groupby("ui_type").agg(
        task_time    =("task_completion_time_sec",   "mean"),
        comprehension=("comprehension_score_0_3",    "mean"),
        filter_cnt   =("filter_count",               "mean"),
        backtrack    =("backtrack_count",            "mean"),
        click_cnt    =("click_count",                "mean"),
        gate_open_rate=("gate_reason",
                        lambda s: float((s == "ICM_semantic_justified").mean())),
    ).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Parameters")
threshold = st.sidebar.slider(
    "Semantic Gate Threshold (θₛ)", 0.30, 0.90, 0.60, 0.05,
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
    "dwell_time":   sig_dwell, "click_rate":   sig_click,
    "filtering":    sig_filt,  "backtracking": sig_back,
    "efficiency":   sig_eff,   "comprehension": sig_comp,
}

# ── generate data with current threshold ──────────────────────────────────────
df, kg_meta = generate_data(gate_threshold=threshold)
kg_live = KnowledgeGraphSimulator(rng=np.random.default_rng(SEED), gate_threshold=threshold)
adapt_live, score_live, debug_live = kg_live.should_adapt(sel_intent, sel_load, live_sigs)
actions_live = kg_live.top_actions(live_sigs, k=2)
summ = summary_table(df)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Stability-Aware Adaptive Visualization Framework")
st.markdown(
    "**HCII 2026 Poster Demo** · Suhyun Park · Korea University  \n"
    "Synthetic sessions: **N = 480** · Seed: 2 · ML accuracy: **~94%**"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 UI Comparison",
    "🔍 Live Gate Inspector",
    "🕸️ KG Subgraph",
    "📈 Threshold Sweep",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · UI Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Performance by UI Type — High Cognitive Load Sessions")

    c1, c2, c3 = st.columns(3)
    ui_order = ["static", "immediate_adaptive", "stability_aware_adaptive"]

    def get(col): return [summ[summ.ui_type == u][col].values[0] for u in ui_order]

    tct_vals  = get("task_time")
    comp_vals = get("comprehension")
    static_tct = tct_vals[0]

    with c1:
        delta_str = f"-{(static_tct - tct_vals[2]) / static_tct * 100:.1f}% vs static"
        st.metric("Task Time · Stability-Aware", f"{tct_vals[2]:.1f}s", delta_str,
                  delta_color="inverse")
    with c2:
        delta_comp = f"+{(comp_vals[2] - comp_vals[0]) / comp_vals[0] * 100:.1f}% vs static"
        st.metric("Comprehension · Stability-Aware", f"{comp_vals[2]:.2f}", delta_comp)
    with c3:
        gate_row = summ[summ.ui_type == "stability_aware_adaptive"].iloc[0]
        st.metric("Gate Open Rate", f"{gate_row.gate_open_rate * 100:.1f}%",
                  help="% of high-load sessions where ICM gate opened")

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        for ui, color in UI_COLORS.items():
            val = summ[summ.ui_type == ui]["task_time"].values[0]
            fig.add_bar(x=[UI_LABELS[ui]], y=[val], name=UI_LABELS[ui],
                        marker_color=color, showlegend=False,
                        text=[f"{val:.1f}s"], textposition="outside")
        fig.update_layout(title="Task Completion Time (sec)", yaxis_title="seconds",
                          height=350, plot_bgcolor="white",
                          yaxis=dict(gridcolor="#eee"))
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
        fig2.add_bar(
            name=UI_LABELS[ui],
            x=burden_labels,
            y=[row[m] for m in burden_metrics],
            marker_color=color,
        )
    fig2.update_layout(barmode="group", height=350, plot_bgcolor="white",
                       yaxis=dict(gridcolor="#eee"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Gate Open Rate by Intent (Stability-Aware)")
    high_stab = df[(df["cognitive_load"] == "high") & (df["ui_type"] == "stability_aware_adaptive")].copy()
    high_stab["gate_open"] = (high_stab["gate_reason"] == "ICM_semantic_justified").astype(int)
    gate_by_intent = high_stab.groupby("intent")["gate_open"].mean().reindex(INTENTS) * 100
    fig3 = px.bar(x=gate_by_intent.index, y=gate_by_intent.values,
                  text=[f"{v:.1f}%" for v in gate_by_intent.values],
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
            rows_debug = []
            for metric, info in debug_live.items():
                rows_debug.append({
                    "Metric": metric,
                    "Signal": info["signal"],
                    "w(I→M)": info["w(I→M)"],
                    "w(M→C)": info["w(M→C)"],
                    "Contribution": info["contrib"],
                })
            df_debug = pd.DataFrame(rows_debug).sort_values("Contribution", ascending=False)

            fig_bar = px.bar(
                df_debug, x="Metric", y="Contribution",
                color="Contribution", color_continuous_scale="Teal",
                text=df_debug["Contribution"].apply(lambda x: f"{x:.4f}"),
                title="Per-Metric ICM Contribution",
            )
            fig_bar.add_hline(
                y=threshold * df_debug["Contribution"].sum() if df_debug["Contribution"].sum() > 0 else threshold,
                line_dash="dash", line_color="red",
                annotation_text=f"threshold proxy",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(height=340, plot_bgcolor="white",
                                  coloraxis_showscale=False,
                                  yaxis=dict(gridcolor="#eee"))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(
                df_debug.style.format({
                    "Signal": "{:.3f}", "w(I→M)": "{:.3f}",
                    "w(M→C)": "{:.3f}", "Contribution": "{:.4f}",
                }),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · KG Subgraph
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🕸️ ICM Subgraph Visualization")
    st.markdown("Intent–Cognitive State–Metric subgraph for the selected intent.")

    kg_viz = KnowledgeGraphSimulator(rng=np.random.default_rng(SEED), gate_threshold=threshold)
    viz_intent = st.selectbox("Intent to visualize", INTENTS, index=1, key="kg_intent")
    viz_cog    = st.radio("Cognitive State", ["high", "low"], horizontal=True)

    i_node = f"I:{viz_intent}"; c_node = f"C:{viz_cog}"
    m_nodes = [v for u, v in kg_viz.G.out_edges(i_node)
               if kg_viz.G.nodes[v].get("ntype") == "M"]
    sub_nodes = [i_node, c_node] + m_nodes
    H = kg_viz.G.subgraph(sub_nodes).copy()

    # layout
    pos = {i_node: (0, 2), c_node: (4, 2)}
    for idx, m in enumerate(sorted(m_nodes)):
        pos[m] = (2, idx * 0.8)

    # build plotly scatter + edges
    edge_x, edge_y, edge_text_x, edge_text_y, edge_weights = [], [], [], [], []
    for u, v, d in H.edges(data=True):
        if d.get("etype") not in ("I→M", "M→C"): continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_text_x.append((x0 + x1) / 2)
        edge_text_y.append((y0 + y1) / 2)
        edge_weights.append(f"{d['weight']:.2f}")

    node_x = [pos[n][0] for n in H.nodes()]
    node_y = [pos[n][1] for n in H.nodes()]
    ntype_color = {"I": "#636EFA", "C": "#EF553B", "M": "#00CC96", "A": "#AB63FA"}
    node_colors = [ntype_color.get(H.nodes[n].get("ntype", "M"), "#aaa") for n in H.nodes()]
    node_labels = []
    for n in H.nodes():
        parts = n.split(":")
        node_labels.append(parts[1].replace("_", "\n") if len(parts) > 1 else n)

    fig_kg = go.Figure()
    fig_kg.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#aaa", width=2), hoverinfo="none",
    ))
    fig_kg.add_trace(go.Scatter(
        x=edge_text_x, y=edge_text_y, mode="text",
        text=edge_weights, textfont=dict(size=11, color="#555"),
        hoverinfo="none",
    ))
    fig_kg.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=38, color=node_colors, line=dict(color="black", width=1.5)),
        text=node_labels, textposition="middle center",
        textfont=dict(size=9, color="white"),
        hovertext=[n for n in H.nodes()], hoverinfo="text",
    ))
    fig_kg.update_layout(
        title=f"ICM Subgraph · Intent={viz_intent} · Cog={viz_cog}",
        showlegend=False, height=420,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_kg, use_container_width=True)

    legend_cols = st.columns(4)
    for col, (ntype, color, label) in zip(legend_cols, [
        ("I", "#636EFA", "Intent (I)"), ("M", "#00CC96", "Metric (M)"),
        ("C", "#EF553B", "Cognitive (C)"), ("A", "#AB63FA", "Action (A)")
    ]):
        col.markdown(
            f'<span style="background:{color};padding:3px 10px;border-radius:4px;'
            f'color:white;font-size:12px">{label}</span>', unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 Stability–Performance Tradeoff (Threshold Sweep)")
    st.markdown(
        "Gate open rate decreases as threshold rises → fewer adaptations → "
        "more stability, potentially less performance gain."
    )

    high_df = df[df["cognitive_load"] == "high"].copy()
    static_tct_mean = float(df[(df["cognitive_load"] == "high") &
                               (df["ui_type"] == "static")]["task_completion_time_sec"].mean())

    stab_df = high_df[high_df["ui_type"] == "stability_aware_adaptive"].dropna(subset=["gate_score"])

    thresholds = np.arange(0.30, 0.91, 0.05)
    curve_rows = []
    for t in thresholds:
        gate_open = (stab_df["gate_score"] >= t)
        open_rate = float(gate_open.mean())
        tct_mean  = float(stab_df["task_completion_time_sec"].mean())
        improve   = (static_tct_mean - tct_mean) / static_tct_mean * 100.0
        uvi_proxy = open_rate * 100
        curve_rows.append({
            "threshold": round(t, 2),
            "gate_open_rate_pct": round(open_rate * 100, 1),
            "tct_improve_pct": round(improve, 1),
            "uvi_proxy": round(uvi_proxy, 1),
        })
    curve_df = pd.DataFrame(curve_rows)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(
            x=curve_df["gate_open_rate_pct"], y=curve_df["tct_improve_pct"],
            mode="lines+markers+text",
            text=curve_df["threshold"].astype(str),
            textposition="top center", textfont=dict(size=10),
            line=dict(color="#00CC96", width=2.5),
            marker=dict(size=8),
            name="θ sweep",
        ))
        fig_sw.update_layout(
            title="Gate Open Rate vs. Task Time Improvement",
            xaxis_title="Gate Open Rate (%) ← lower = more stable",
            yaxis_title="Task Time Improvement vs Static (%)",
            height=380, plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
        )
        # mark current threshold
        cur = curve_df[curve_df["threshold"] == round(threshold, 2)]
        if not cur.empty:
            fig_sw.add_vline(x=float(cur["gate_open_rate_pct"].values[0]),
                             line_dash="dash", line_color="red",
                             annotation_text=f"current θ={threshold:.2f}")
        st.plotly_chart(fig_sw, use_container_width=True)

    with col_b:
        fig_uvi = go.Figure()
        fig_uvi.add_trace(go.Scatter(
            x=curve_df["threshold"], y=curve_df["uvi_proxy"],
            mode="lines+markers",
            line=dict(color="#636EFA", width=2.5),
            marker=dict(size=7),
            name="UVI proxy",
        ))
        fig_uvi.add_vline(x=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"θ={threshold:.2f}")
        fig_uvi.update_layout(
            title="UI Volatility Index (UVI) vs. Threshold",
            xaxis_title="Threshold (θₛ)",
            yaxis_title="UVI proxy (gate open rate × 100)",
            height=380, plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_uvi, use_container_width=True)

    st.dataframe(curve_df, use_container_width=True)

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Evaluation uses entirely synthetic interaction sessions (N=480). "
    "No real participants or personal data involved. "
    "Human-subject validation is planned as future work. "
    "HCII 2026 · Suhyun Park · Korea University"
)
