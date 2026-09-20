import streamlit as st
from subtitle_nmt_thesis.ui.state import run_single_comparison
from subtitle_nmt_thesis.ui.components.subtitle_card import get_badge_html

PRESETS = [
    {
        "label": "NPTEL CS: Binary Search Trees & Complexity",
        "text": "Today we will analyze balanced binary search trees and their asymptotic time complexity.",
        "duration": 4.0,
    },
    {
        "label": "NPTEL Physics: Thermodynamics & Entropy",
        "text": "The second law of thermodynamics states that total entropy of an isolated system always increases over time.",
        "duration": 4.5,
    },
    {
        "label": "NPTEL Math: Linear Transformations & Eigenvectors",
        "text": "An eigenvector of a square matrix represents a direction invariant under linear transformations.",
        "duration": 4.0,
    },
    {
        "label": "NPTEL Chemistry: Molecular Orbital Theory",
        "text": "In chemical bonding, the overlap of atomic orbitals leads to the formation of bonding and antibonding states.",
        "duration": 4.5,
    },
    {
        "label": "NPTEL Electrical: Frequency Response & Filters",
        "text": "The low pass filter attenuates high frequency noise while passing desired signals through the system.",
        "duration": 3.8,
    },
]


def render_playground_tab(
    model, max_cpl: int, max_cps: int, src_lang: str, tgt_lang: str
) -> None:
    st.subheader("🧪 Live Interactive Model Playground (A/B Test)")
    st.write(
        "Compare standard unconstrained translation vs. **constraint-aware NMT** (CPL/CPS limits) "
        "on individual educational sentences in real time."
    )

    preset_choice = st.selectbox(
        "Choose an educational sample preset or type below:",
        ["(Custom Input)"] + [p["label"] for p in PRESETS],
    )

    default_text = ""
    default_dur = 3.5
    if preset_choice != "(Custom Input)":
        match = next(p for p in PRESETS if p["label"] == preset_choice)
        default_text = match["text"]
        default_dur = match["duration"]

    input_text = st.text_area(
        "Source Text:",
        value=default_text
        or "In this lecture, we explore how neural translation models adapt to subtitle screen constraints.",
        height=90,
    )
    duration_sec = st.slider(
        "Estimated Cue Duration (seconds):",
        min_value=1.0,
        max_value=8.0,
        value=default_dur,
        step=0.5,
    )

    if st.button("🚀 Translate & Compare Models", type="primary", use_container_width=True):
        with st.spinner("Executing baseline and constraint-aware inference..."):
            result = run_single_comparison(
                text=input_text,
                model=model,
                max_cpl=max_cpl,
                max_cps=max_cps,
                duration_sec=duration_sec,
            )

        col_base, col_const = st.columns(2)
        with col_base:
            st.markdown("### 🔹 Baseline (Unconstrained)")
            st.info(result["baseline_text"])
            m = result["baseline_metrics"]
            st.markdown(
                f"{get_badge_html('CPL', m['max_line_cpl'], m['cpl_compliant'])}"
                f"{get_badge_html('CPS', m['cps'], m['cps_compliant'])}",
                unsafe_allow_html=True,
            )
            st.caption(f"Lines: {m['line_count']} | Total Characters: {m['char_count']}")

        with col_const:
            st.markdown("### ⚡ Constraint-Aware NMT")
            st.success(result["constrained_text"])
            m = result["constrained_metrics"]
            st.markdown(
                f"{get_badge_html('CPL', m['max_line_cpl'], m['cpl_compliant'])}"
                f"{get_badge_html('CPS', m['cps'], m['cps_compliant'])}",
                unsafe_allow_html=True,
            )
            st.caption(f"Lines: {m['line_count']} | Total Characters: {m['char_count']}")

        # Highlight difference
        base_cpl = result["baseline_metrics"]["max_line_cpl"]
        const_cpl = result["constrained_metrics"]["max_line_cpl"]
        if base_cpl > max_cpl and const_cpl <= max_cpl:
            st.success(
                f"🎯 **Constraint Success**: Baseline violated CPL limit ({base_cpl} > {max_cpl}), "
                f"whereas Constraint-Aware NMT successfully compressed output to {const_cpl} chars!"
            )
        elif const_cpl <= max_cpl:
            st.info("✅ Both outputs fit comfortably within target subtitle line length constraints.")
