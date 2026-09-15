import streamlit as st
from subtitle_nmt_thesis.ui.utils.metrics_helper import compute_cue_metrics


def format_timestamp(start: float, end: float) -> str:
    def _sec_to_ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    return f"{_sec_to_ts(start)} --> {_sec_to_ts(end)}"


def get_badge_html(label: str, value: str | int | float, is_compliant: bool) -> str:
    color = "#28a745" if is_compliant else "#dc3545"
    bg = "#e8f5e9" if is_compliant else "#ffebee"
    icon = "🟢" if is_compliant else "🔴"
    return (
        f"<span style='background-color: {bg}; color: {color}; border: 1px solid {color}; "
        f"padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 6px;'>"
        f"{icon} {label}: {value}</span>"
    )


def render_subtitle_card(
    cue_idx: int,
    timestamp: str,
    source_text: str,
    target_text: str,
    duration_sec: float,
    max_cpl: int = 42,
    max_cps: int = 21,
) -> None:
    src_metrics = compute_cue_metrics(source_text, duration_sec, max_cpl, max_cps)
    tgt_metrics = compute_cue_metrics(target_text, duration_sec, max_cpl, max_cps)

    with st.container(border=True):
        st.caption(f"**Cue #{cue_idx}** &nbsp;|&nbsp; ⏱️ `{timestamp}`")
        col_src, col_tgt = st.columns(2)
        with col_src:
            st.markdown(f"**Source ({src_metrics['char_count']} chars):**")
            st.info(source_text)
            st.markdown(
                f"{get_badge_html('CPL', src_metrics['max_line_cpl'], src_metrics['cpl_compliant'])}"
                f"{get_badge_html('CPS', src_metrics['cps'], src_metrics['cps_compliant'])}",
                unsafe_allow_html=True,
            )
        with col_tgt:
            st.markdown(f"**Localized Target ({tgt_metrics['char_count']} chars):**")
            st.success(target_text)
            st.markdown(
                f"{get_badge_html('CPL', tgt_metrics['max_line_cpl'], tgt_metrics['cpl_compliant'])}"
                f"{get_badge_html('CPS', tgt_metrics['cps'], tgt_metrics['cps_compliant'])}",
                unsafe_allow_html=True,
            )
