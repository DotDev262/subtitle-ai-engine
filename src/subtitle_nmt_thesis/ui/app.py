import sys
from pathlib import Path
import streamlit as st
from subtitle_nmt_thesis.pipeline.run import Pipeline
from subtitle_nmt_thesis.ui.state import get_device_status, get_cached_translator
from subtitle_nmt_thesis.ui.components.playground import render_playground_tab
from subtitle_nmt_thesis.ui.components.studio import render_studio_tab
from subtitle_nmt_thesis.ui.components.metrics_view import render_metrics_tab


def build_pipeline_instance(
    max_cpl: int = 42,
    max_cps: int = 21,
    constrained_decoding: bool = True,
    src_lang: str = "en",
    tgt_lang: str = "hi",
    model_name: str = "auto",
) -> Pipeline:
    return Pipeline(
        model_name=model_name,
        max_cpl=max_cpl,
        max_cps=max_cps,
        constrained_decoding=constrained_decoding,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )


def main() -> None:
    st.set_page_config(
        page_title="Subtitle AI Engine | Thesis Showcase",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar Header & Hardware status
    device, is_cuda = get_device_status()
    device_badge = "⚡ GPU (CUDA Enabled)" if is_cuda else "💻 CPU Mode (Standard)"

    st.sidebar.title("🎬 Subtitle AI Engine")
    st.sidebar.caption("Multi-objective constraint-aware NMT for educational subtitle localization.")
    st.sidebar.info(f"**Execution Device:** {device_badge}")

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Engine Parameters")

    lang_choice = st.sidebar.selectbox(
        "Language Direction:",
        ["English -> Hindi (en -> hi)", "Hindi -> English (hi -> en)"],
    )
    src_lang = "en" if "en -> hi" in lang_choice else "hi"
    tgt_lang = "hi" if "en -> hi" in lang_choice else "en"

    model_choice = st.sidebar.selectbox(
        "Model Architecture:",
        ["auto (IndicTrans2 / Opus-MT)", "indictrans2", "Helsinki-NLP/opus-mt-en-hi"],
    )
    model_name = "auto" if "auto" in model_choice else model_choice

    max_cpl = st.sidebar.slider(
        "Max Characters Per Line (CPL):", min_value=25, max_value=65, value=42, step=1
    )
    max_cps = st.sidebar.slider(
        "Max Characters Per Second (CPS):", min_value=12, max_value=32, value=21, step=1
    )
    constrained = st.sidebar.toggle("Enable Constraint-Aware Decoding", value=True)

    # Main Tabs
    tab_play, tab_studio, tab_metrics = st.tabs([
        "🧪 Model Playground (A/B)",
        "🎬 Video & Subtitle Studio",
        "📊 Thesis Evaluation Metrics",
    ])

    with tab_play:
        translator = get_cached_translator(model_name, src_lang, tgt_lang)
        render_playground_tab(translator, max_cpl, max_cps, src_lang, tgt_lang)

    with tab_studio:
        def _pipeline_factory(
            max_cpl=max_cpl, max_cps=max_cps, src_lang=src_lang, tgt_lang=tgt_lang
        ):
            return build_pipeline_instance(
                max_cpl=max_cpl,
                max_cps=max_cps,
                constrained_decoding=constrained,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                model_name=model_name,
            )

        render_studio_tab(_pipeline_factory, max_cpl, max_cps, src_lang, tgt_lang)

    with tab_metrics:
        render_metrics_tab(max_cpl, max_cps)


def run_app() -> None:
    """CLI entrypoint executed via 'uv run ui'."""
    from streamlit.web import cli as stcli

    app_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
