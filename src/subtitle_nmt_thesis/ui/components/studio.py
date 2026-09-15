import tempfile
from pathlib import Path
import streamlit as st
from subtitle_nmt_thesis.data.parser import parse_srt, write_srt, SubtitleItem
from subtitle_nmt_thesis.ui.utils.demo_loader import get_demo_asset_paths, load_demo_srt
from subtitle_nmt_thesis.ui.utils.srt_vtt import srt_to_vtt
from subtitle_nmt_thesis.ui.components.subtitle_card import render_subtitle_card, format_timestamp


def prepare_subtitles_for_studio(items: list[SubtitleItem]) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    write_srt(items, tmp_path)
    srt_str = tmp_path.read_text(encoding="utf-8")
    tmp_path.unlink(missing_ok=True)
    vtt_str = srt_to_vtt(srt_str)
    return srt_str, vtt_str


def render_studio_tab(
    pipeline_factory, max_cpl: int, max_cps: int, src_lang: str, tgt_lang: str
) -> None:
    st.subheader("🎬 Video & Subtitle Localization Studio")
    st.write(
        "Run full subtitle files through the localization pipeline, preview with video, "
        "and inspect cue compliance."
    )

    source_mode = st.radio(
        "Select Input Source:",
        ["Use Pre-loaded Demo (Recommended)", "Upload Custom Files"],
        horizontal=True,
    )

    video_bytes = None
    input_srt_content = ""

    if source_mode == "Use Pre-loaded Demo (Recommended)":
        assets = get_demo_asset_paths()
        input_srt_content = load_demo_srt("en")
        if assets["video"].exists():
            video_bytes = assets["video"].read_bytes()
        st.success("Loaded built-in educational lecture clip & sample SRT.")
    else:
        uploaded_srt = st.file_uploader("Upload Subtitle File (.srt):", type=["srt"])
        uploaded_video = st.file_uploader("Upload Matching Video (.mp4) (Optional):", type=["mp4"])
        if uploaded_srt:
            input_srt_content = uploaded_srt.getvalue().decode("utf-8")
        if uploaded_video:
            video_bytes = uploaded_video.getvalue()

    if input_srt_content:
        st.text_area("Source Subtitles Preview:", input_srt_content, height=120, disabled=True)

        if st.button(
            "✨ Run Subtitle Localization Pipeline", type="primary", use_container_width=True
        ):
            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as in_tmp:
                in_path = Path(in_tmp.name)
            in_path.write_text(input_srt_content, encoding="utf-8")

            out_path = in_path.with_name("translated_out.srt")

            pipeline = pipeline_factory(
                max_cpl=max_cpl, max_cps=max_cps, src_lang=src_lang, tgt_lang=tgt_lang
            )
            items = parse_srt(in_path)

            progress_bar = st.progress(0, text="Translating subtitle cues...")
            translated_texts = []
            for idx, item in enumerate(items):
                progress_bar.progress(
                    (idx + 1) / len(items), text=f"Localizing cue {idx + 1}/{len(items)}..."
                )
                ctx = pipeline.context_builder.build(items, idx)
                context_str = pipeline.context_builder.format_context(ctx)
                t = pipeline.translator.translate(
                    item.text,
                    item=item,
                    context=context_str,
                    constrained=pipeline.constrained_decoding,
                    max_cpl=max_cpl,
                )
                translated_texts.append(t)

            translated_items = [
                SubtitleItem(id=item.id, start=item.start, end=item.end, text=t)
                for item, t in zip(items, translated_texts)
            ]
            write_srt(translated_items, out_path)

            srt_str, vtt_str = prepare_subtitles_for_studio(translated_items)
            st.session_state["studio_results"] = {
                "source_items": items,
                "translated_items": translated_items,
                "srt_str": srt_str,
                "vtt_str": vtt_str,
            }
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            progress_bar.empty()
            st.success("Translation complete!")

    if "studio_results" in st.session_state:
        res = st.session_state["studio_results"]
        st.divider()

        # Video Player with Subtitles
        if video_bytes:
            st.markdown("### 📺 Synchronized Video Player")
            with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as vtt_file:
                vtt_file.write(res["vtt_str"].encode("utf-8"))
                vtt_file_path = vtt_file.name
            st.video(video_bytes, subtitles=vtt_file_path)
            Path(vtt_file_path).unlink(missing_ok=True)

        # Download actions
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📥 Download Localized SRT",
                res["srt_str"],
                file_name="localized_subtitles.srt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "📥 Download WebVTT Subtitles",
                res["vtt_str"],
                file_name="localized_subtitles.vtt",
                mime="text/vtt",
                use_container_width=True,
            )

        # Side by side card list
        st.markdown("### 📋 Cue-by-Cue Compliance Inspector")
        for src_it, tgt_it in zip(res["source_items"], res["translated_items"]):
            dur = max(tgt_it.end - tgt_it.start, 0.1)
            ts = format_timestamp(tgt_it.start, tgt_it.end)
            render_subtitle_card(
                cue_idx=tgt_it.id,
                timestamp=ts,
                source_text=src_it.text,
                target_text=tgt_it.text,
                duration_sec=dur,
                max_cpl=max_cpl,
                max_cps=max_cps,
            )
