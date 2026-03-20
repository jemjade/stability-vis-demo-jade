"""
Stability-Aware Adaptive Visualization Framework — Synthetic Session Generator

This module serves as the single source of truth for:
- knowledge graph structure
- semantic gate logic
- synthetic session generation

The current implementation is designed for a synthetic feasibility-oriented
demonstration of stability-aware adaptive visualization. It should not be
interpreted as externally validated performance evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

# ──────────────────────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────────────────────
SEED = 6  # selected seed for a stable synthetic-session demonstration
N_SESSIONS = 480
PER_GROUP = 40  # 4 intents × 3 UI conditions × 40 = 480

INTENTS = [
    "fast_concise",
    "fast_detailed",
    "gradual_concise",
    "gradual_detailed",
]
UIS = ["static", "immediate_adaptive", "stability_aware_adaptive"]

INTENT_SHIFT_PROB = 0.18

WINDOWS_PER_SESSION = 5
MIN_CONSEC_HIGH_WINDOWS = 2

# Two-state Markov chain for synthetic cognitive-load windows
MARKOV_P_HH = 0.72
MARKOV_P_LL = 0.68

HIGH_LOAD_PROB = {
    "fast_concise": 0.42,
    "fast_detailed": 0.62,
    "gradual_concise": 0.35,
    "gradual_detailed": 0.56,
}

# Synthetic adaptation parameters used in the current feasibility-oriented simulation.
# These values produce illustrative improvements in task time and comprehension
# under high-load synthetic sessions, but should not be interpreted as externally validated effects.
TCT_MULT = 0.55
COMP_DELTA = 0.65
GATE_DEFAULT = 0.51


class KnowledgeGraphSimulator:
    """
    Build and query the knowledge graph used for semantic gating.

    The semantic gate uses the ICM subgraph (Intent → Metric → Cognitive State)
    to evaluate whether an adaptation is semantically justified in the current
    synthetic-session setting.

    The current parameterization is intended for a synthetic feasibility
    demonstration rather than for externally validated performance claims.
    """

    def __init__(self, rng: np.random.Generator, gate_threshold: float = GATE_DEFAULT):
        self.rng = rng
        self.gate_threshold = gate_threshold
        self.G = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        # ── nodes: total 24 ────────────────────────────────────────────────────
        self.G.add_node("U:user", ntype="User")

        for i in INTENTS:
            self.G.add_node(f"I:{i}", ntype="Intent")

        for e in [
            "dwell_time",
            "click_rate",
            "filtering",
            "backtracking",
            "efficiency",
            "comprehension",
        ]:
            self.G.add_node(f"E:{e}", ntype="Event")
            self.G.add_node(f"M:{e}", ntype="Metric")

        self.G.add_node("C:high", ntype="CognitiveState")
        self.G.add_node("C:low", ntype="CognitiveState")

        for a in [
            "simplify_layout",
            "progressive_disclosure",
            "stabilize_layout",
            "reduce_filters",
            "highlight_key_metrics",
        ]:
            self.G.add_node(f"A:{a}", ntype="Action")

        # U→I
        for i in INTENTS:
            self.G.add_edge("U:user", f"I:{i}", weight=1.0, etype="U→I")

        # I→M
        weights = {
            "fast_concise": {
                "click_rate": 0.85,
                "filtering": 0.80,
                "efficiency": 0.78,
                "dwell_time": 0.40,
                "comprehension": 0.35,
            },
            "fast_detailed": {
                "click_rate": 0.78,
                "filtering": 0.72,
                "dwell_time": 0.62,
                "efficiency": 0.65,
                "comprehension": 0.55,
            },
            "gradual_concise": {
                "dwell_time": 0.70,
                "comprehension": 0.75,
                "efficiency": 0.60,
                "filtering": 0.45,
                "backtracking": 0.40,
            },
            "gradual_detailed": {
                "dwell_time": 0.82,
                "comprehension": 0.86,
                "backtracking": 0.72,
                "filtering": 0.58,
                "click_rate": 0.42,
            },
        }
        for intent, mapping in weights.items():
            for m, w in mapping.items():
                self.G.add_edge(f"I:{intent}", f"M:{m}", weight=w, etype="I→M")

        # E→C
        self.G.add_edge("E:dwell_time", "C:high", weight=0.70, etype="E→C")
        self.G.add_edge("E:click_rate", "C:high", weight=0.62, etype="E→C")
        self.G.add_edge("E:filtering", "C:high", weight=0.66, etype="E→C")
        self.G.add_edge("E:backtracking", "C:high", weight=0.78, etype="E→C")
        self.G.add_edge("E:efficiency", "C:low", weight=0.55, etype="E→C")
        self.G.add_edge("E:comprehension", "C:low", weight=0.72, etype="E→C")

        # E→M
        self.G.add_edge("E:dwell_time", "M:dwell_time", weight=1.00, etype="E→M")
        self.G.add_edge("E:click_rate", "M:click_rate", weight=1.00, etype="E→M")
        self.G.add_edge("E:filtering", "M:filtering", weight=1.00, etype="E→M")
        self.G.add_edge("E:backtracking", "M:backtracking", weight=1.00, etype="E→M")
        self.G.add_edge("E:efficiency", "M:efficiency", weight=1.00, etype="E→M")
        self.G.add_edge("E:comprehension", "M:comprehension", weight=1.00, etype="E→M")

        # M→C (ICM gate path)
        self.G.add_edge("M:dwell_time", "C:high", weight=0.70, etype="M→C")
        self.G.add_edge("M:backtracking", "C:high", weight=0.78, etype="M→C")
        self.G.add_edge("M:filtering", "C:high", weight=0.66, etype="M→C")
        self.G.add_edge("M:comprehension", "C:low", weight=0.72, etype="M→C")

        # C→A
        self.G.add_edge("C:high", "A:simplify_layout", weight=0.80, etype="C→A")
        self.G.add_edge("C:high", "A:progressive_disclosure", weight=0.70, etype="C→A")
        self.G.add_edge("C:high", "A:stabilize_layout", weight=0.75, etype="C→A")
        self.G.add_edge("C:low", "A:highlight_key_metrics", weight=0.60, etype="C→A")

        # M→A
        self.G.add_edge("M:backtracking", "A:stabilize_layout", weight=0.75, etype="M→A")
        self.G.add_edge("M:filtering", "A:reduce_filters", weight=0.60, etype="M→A")
        self.G.add_edge("M:dwell_time", "A:progressive_disclosure", weight=0.55, etype="M→A")
        self.G.add_edge("M:efficiency", "A:highlight_key_metrics", weight=0.50, etype="M→A")
        self.G.add_edge("M:comprehension", "A:simplify_layout", weight=0.58, etype="M→A")

    def _normalize(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def icm_subgraph_score(
        self,
        intent: str,
        cognitive_load: str,
        metric_signals: dict,
    ) -> tuple[float, dict]:
        i_node = f"I:{intent}"
        c_node = f"C:{cognitive_load}"
        total = 0.0
        denom = 0.0
        contribs: dict = {}

        for m_key, sig in metric_signals.items():
            m_node = f"M:{m_key}"
            if not self.G.has_edge(i_node, m_node):
                continue

            w_im = self.G.edges[i_node, m_node]["weight"]
            w_mc = self.G.edges[m_node, c_node]["weight"] if self.G.has_edge(m_node, c_node) else 0.10
            c = w_im * sig * w_mc

            contribs[m_key] = {
                "signal": round(sig, 3),
                "w(I→M)": round(w_im, 3),
                "w(M→C)": round(w_mc, 3),
                "contrib": round(c, 4),
            }
            total += c
            denom += w_im * w_mc

        score = total / denom if denom > 1e-9 else 0.0
        return float(np.clip(score, 0.0, 1.0)), contribs

    def should_adapt(
        self,
        intent: str,
        cognitive_load: str,
        metric_signals: dict,
        consecutive_high_windows: int = 1,
    ) -> tuple[bool, str, float, dict]:
        score, debug = self.icm_subgraph_score(intent, cognitive_load, metric_signals)
        debug["consecutive_high_windows"] = consecutive_high_windows
        debug["min_required_high_windows"] = MIN_CONSEC_HIGH_WINDOWS

        # Paper-aligned safeguard: adaptation is only considered under high cognitive load.
        if cognitive_load != "high":
            return False, "cognitive_load_low", score, debug

        # Persistence-aware safeguard for the current synthetic-session prototype.
        if consecutive_high_windows < MIN_CONSEC_HIGH_WINDOWS:
            return False, "persistence_not_met", score, debug

        if score >= self.gate_threshold:
            return True, "ICM_semantic_justified", score, debug

        return False, "ICM_not_significant", score, debug

    def get_action_recommendations(self, metric_signals: dict, topk: int = 2) -> list[tuple[str, float]]:
        action_scores: dict[str, float] = {}

        for m_key, sig in metric_signals.items():
            m_node = f"M:{m_key}"
            if m_node not in self.G:
                continue
            for _, a_node, data in self.G.out_edges(m_node, data=True):
                if data.get("etype") != "M→A":
                    continue
                action_scores[a_node] = action_scores.get(a_node, 0.0) + data["weight"] * sig

        ranked = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [(a.split("A:")[1], float(s)) for a, s in ranked]


def simulate_load_windows(
    intent: str,
    rng: np.random.Generator,
    n_windows: int = WINDOWS_PER_SESSION,
) -> tuple[list[str], str, int, int]:
    """Simulate short cognitive-load windows using a two-state Markov chain."""
    state = "high" if rng.random() < HIGH_LOAD_PROB[intent] else "low"
    states: list[str] = []

    for i in range(n_windows):
        if i > 0:
            if state == "high":
                state = "high" if rng.random() < MARKOV_P_HH else "low"
            else:
                state = "low" if rng.random() < MARKOV_P_LL else "high"
        states.append(state)

    tail = 0
    for s in reversed(states):
        if s == "high":
            tail += 1
        else:
            break

    max_run = 0
    cur = 0
    for s in states:
        if s == "high":
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    return states, states[-1], tail, max_run


def generate_sessions(
    seed: int = SEED,
    gate_threshold: float = GATE_DEFAULT,
) -> tuple[pd.DataFrame, KnowledgeGraphSimulator]:
    """
    Generate N=480 synthetic sessions aligned to the current paper framing.

    The generated values support a feasibility-oriented illustration of
    stability-aware adaptive visualization under synthetic conditions.
    """
    rng = np.random.default_rng(seed)
    kg = KnowledgeGraphSimulator(rng=rng, gate_threshold=gate_threshold)
    rows: list[dict] = []
    session_id = 0

    for base_intent in INTENTS:
        for ui in UIS:
            for _ in range(PER_GROUP):
                session_id += 1

                intent = rng.choice(INTENTS) if rng.random() < INTENT_SHIFT_PROB else base_intent
                states, load, consecutive_high_windows, max_consecutive_high_windows = simulate_load_windows(intent, rng)
                speed, detail = intent.split("_")

                if load == "low":
                    dwell = rng.normal(5.5, 2.2)
                    click = int(max(1, rng.normal(6, 3)))
                    filt = int(max(0, rng.normal(3, 2)))
                    back = int(max(0, rng.normal(1.0, 1.0)))
                    tct = rng.normal(55, 18)
                    comp = rng.normal(2.4, 0.6)
                else:
                    dwell = rng.normal(10.5, 3.0)
                    click = int(max(1, rng.normal(12, 4)))
                    filt = int(max(0, rng.normal(7, 3)))
                    back = int(max(0, rng.normal(2.7, 1.5)))
                    tct = rng.normal(105, 25)
                    comp = rng.normal(1.5, 0.7)

                if speed == "fast":
                    click = int(click * rng.normal(1.15, 0.08))
                    filt = int(filt * rng.normal(1.20, 0.10))
                    tct += rng.normal(5, 4)

                if detail == "detailed":
                    dwell *= rng.normal(1.25, 0.10)
                    click = int(click * rng.normal(1.10, 0.08))
                    tct += rng.normal(8, 5)

                dwell = float(max(0.5, dwell))
                tct = float(max(10, tct))
                comp = float(min(3.0, max(0.0, comp)))
                click = int(max(1, click))
                filt = int(max(0, filt))
                back = int(max(0, back))
                eff = tct / max(1, click)

                metric_signals = {
                    "dwell_time": kg._normalize(dwell, 2.0, 16.0),
                    "click_rate": kg._normalize(click, 2.0, 22.0),
                    "filtering": kg._normalize(filt, 0.0, 14.0),
                    "backtracking": kg._normalize(back, 0.0, 8.0),
                    "efficiency": kg._normalize(eff, 3.0, 18.0),
                    "comprehension": 1.0 - kg._normalize(comp, 0.0, 3.0),
                }

                trigger_reason = "n/a"
                gate_score = None
                top_actions: list = []

                if ui == "stability_aware_adaptive" and load == "high":
                    should_adapt, trigger_reason, gate_score, _ = kg.should_adapt(
                        intent=intent,
                        cognitive_load=load,
                        metric_signals=metric_signals,
                        consecutive_high_windows=consecutive_high_windows,
                    )
                    if should_adapt:
                        top_actions = kg.get_action_recommendations(metric_signals, topk=2)
                        tct *= rng.normal(TCT_MULT, 0.04)
                        filt = int(max(0, round(filt * rng.normal(0.70, 0.08))))
                        back = int(max(0, round(back * rng.normal(0.68, 0.08))))
                        click = int(max(1, round(click * rng.normal(0.85, 0.07))))
                        comp += rng.normal(COMP_DELTA, 0.12)

                elif ui == "immediate_adaptive" and load == "high":
                    # Immediate adaptation is modeled as moderately helpful but more volatile.
                    tct *= rng.normal(0.88, 0.06)
                    filt = int(max(0, round(filt * rng.normal(0.82, 0.10))))
                    back = int(max(0, round(back * rng.normal(0.80, 0.10))))
                    click = int(max(1, round(click * rng.normal(0.92, 0.08))))
                    comp += rng.normal(0.25, 0.15)

                    # Interface churn effects
                    click = int(click * rng.normal(1.12, 0.06))
                    back = int(back * rng.normal(1.15, 0.08))

                tct = float(max(10, tct))
                comp = float(min(3.0, max(0.0, comp)))
                click = int(max(1, click))
                filt = int(max(0, filt))
                back = int(max(0, back))
                eff = tct / max(1, click)

                rows.append(
                    {
                        "session_id": session_id,
                        "intent": intent,
                        "monitoring_speed": speed,
                        "visual_detail": detail,
                        "ui_type": ui,
                        "cognitive_load": load,
                        "window_states": "|".join(states),
                        "consecutive_high_windows": consecutive_high_windows,
                        "max_consecutive_high_windows": max_consecutive_high_windows,
                        "dwell_time_sec": dwell,
                        "click_count": click,
                        "filter_count": filt,
                        "backtrack_count": back,
                        "efficiency_sec_per_click": eff,
                        "task_completion_time_sec": tct,
                        "comprehension_score_0_3": comp,
                        "gate_reason": trigger_reason,
                        "gate_score": gate_score,
                        "recommended_actions": ",".join(a for a, _ in top_actions) if top_actions else "",
                    }
                )

    df = pd.DataFrame(rows)
    return df, kg


if __name__ == "__main__":
    df, kg = generate_sessions()
    print("Synthetic sessions:", len(df))
    print("KG nodes / edges:", kg.G.number_of_nodes(), kg.G.number_of_edges())

    high = df[df["cognitive_load"] == "high"]
    summary = high.groupby("ui_type").agg(
        task_time=("task_completion_time_sec", "mean"),
        comprehension=("comprehension_score_0_3", "mean"),
        gate_open_rate=("gate_reason", lambda s: float((s == "ICM_semantic_justified").mean())),
    )
    print(summary)
