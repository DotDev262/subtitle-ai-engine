# Streamlit Showcase UI Design Spec

**Date**: 2026-09-13  
**Project**: `subtitle-ai-engine` (`subtitle-nmt-thesis`)  
**Scope**: Interactive Web UI for Thesis Demonstration and Subtitle Localization  

---

## 1. Overview & Objectives

The goal of this feature is to provide an interactive, professional showcase web application for the `subtitle-ai-engine` thesis project. The UI demonstrates:
1. **Neural Machine Translation (NMT)** model capabilities (specifically English <-> Hindi via IndicTrans2 / Opus-MT).
2. **Constraint-Aware Decoding**: Comparing unconstrained baseline translations against constraint-aware decoding (CPL - Characters Per Line, CPS - Characters Per Second, logit processing, and reranking).
3. **End-to-End Subtitle & Video Localization**: Uploading or selecting pre-packaged educational video and subtitle assets, executing the translation pipeline with live progress, and previewing synchronized subtitles in a video player alongside side-by-side cue inspection.
4. **Thesis Metrics & Benchmarking**: Presenting quantifiable evaluation metrics (CPL violation rate %, CPS distributions, BLEU / chrF scores).

The UI is built using **Streamlit** for its native Python ML ecosystem support, simplicity, and built-in multimedia capabilities.

---

## 2. Architecture & File Structure

The UI subsystem is integrated into the existing package without disrupting any CLI commands or core pipeline interfaces.

```
subtitle-ai-engine/
├── assets/
│   ├── demo.mp4                    # Compact sample video (~5-10s educational clip)
│   ├── sample_en.srt               # English sample subtitles matching demo.mp4
│   └── sample_hi.srt               # Hindi sample subtitles matching demo.mp4
├── src/
│   └── subtitle_nmt_thesis/
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── app.py               # Streamlit main entrypoint & navigation
│       │   ├── state.py             # Model resource caching & session state management
│       │   ├── components/
│       │   │   ├── __init__.py
│       │   │   ├── playground.py       # Tab 1: Single-cue A/B comparison sandbox
│       │   │   ├── studio.py           # Tab 2: Full SRT pipeline, card inspector & video player
│       │   │   ├── metrics_view.py     # Tab 3: Quantitative thesis metrics dashboard
│       │   │   └── subtitle_card.py    # Reusable card component for cue comparison & flags
│       │   └── utils/
│       │       ├── __init__.py
│       │       ├── srt_vtt.py          # SRT <-> WebVTT conversion utility for st.video
│       │       └── demo_loader.py      # Helpers to load built-in demo clips & preset SRTs
└── tests/
    └── test_ui.py                   # Unit tests for SRT/VTT conversion, metrics, and UI helpers
```

---

## 3. Detailed Component Specifications

### 3.1 Sidebar (Global Configuration)
* **Model Selection**: Choose between `auto`, `indictrans2` (`ai4bharat/indictrans2-*`), or `Helsinki-NLP/opus-mt-*`.
* **Language Direction**: `English -> Hindi` or `Hindi -> English`.
* **Hardware Status Indicator**: Displays `⚡ GPU (CUDA)` or `💻 CPU Mode` based on `torch.cuda.is_available()`.
* **Decoding Parameters**:
  * Max CPL slider (default: 42, range: 20–80).
  * Max CPS slider (default: 21, range: 10–35).
  * Method toggle: Constrained Decoding (CPL Logits Processor) vs. Multi-candidate Reranking.

### 3.2 Tab 1: Interactive Model Playground (A/B Comparison)
* **Purpose**: Immediate, low-latency evaluation of single sentences/cues.
* **Inputs**:
  * Text area with sample presets (e.g. Computer Science / Math lecture sentences).
  * Cue duration slider (default: 3.0s) to calculate CPS.
* **Execution**:
  * **Baseline Model**: Unconstrained beam search (`num_beams=5`).
  * **Constraint-Aware Model**: With active CPL/CPS reranker or CPL logit processor.
* **Outputs (Side-by-Side)**:
  * Column 1: Unconstrained Baseline translation.
  * Column 2: Constraint-Aware translation.
  * Metric Badges per column:
    * Character count & Line count.
    * CPL: Flagged 🟢 Compliant or 🔴 Exceeded ($> \text{Max CPL}$).
    * CPS: Flagged 🟢 Compliant or 🔴 Exceeded ($> \text{Max CPS}$).
  * Analytical summary: Explains how the constraint controller adjusted length, avoided line wrapping, or balanced fluency vs. brevity.

### 3.3 Tab 2: Video & Subtitle Studio
* **Purpose**: Full workflow demonstration on real multimedia assets.
* **Inputs**:
  * Source mode toggle: "Use Preloaded Educational Demo" or "Upload Custom Files".
  * File uploaders for `.srt` and optional `.mp4`.
* **Processing**:
  * Run pipeline with cue-by-cue progress feedback:
    ```python
    def progress_callback(current_idx: int, total_items: int, item_text: str): ...
    ```
* **Outputs**:
  * **Synchronized Video Player**: `st.video(video_bytes, subtitles=vtt_bytes)` to preview translated subtitles directly on the video playback.
  * **Interactive Subtitle Card Inspector**:
    * Scrollable list of subtitle cards.
    * Each card displays Cue Number, Timestamp (`start --> end`), Original Text, Translated Text, CPL/CPS stats, and visual compliance pills.
  * **Export Actions**:
    * Download button for localized `.srt`.
    * Download button for converted `.vtt`.

### 3.4 Tab 3: Thesis Metrics & Evaluation Dashboard
* **Purpose**: Present rigorous academic evidence of the constraint-aware approach.
* **Metrics Displayed**:
  * **CPL Violation Rate**: % of lines exceeding target CPL (Baseline vs. Constrained).
  * **CPS Violation Rate**: % of cues exceeding reading speed thresholds.
  * **Length Ratio**: Compression / expansion ratio between source and target text.
  * **Translation Quality (if reference provided)**: BLEU and chrF++ scores calculated via `evaluate` / `sacrebleu`.
* **Visualizations**:
  * Line length distribution histogram comparing Baseline vs. Constrained models with vertical cutoff line at `Max CPL`.

---

## 4. Technical Details & Utilities

### 4.1 WebVTT Conversion (`src/subtitle_nmt_thesis/ui/utils/srt_vtt.py`)
Streamlit's `st.video(..., subtitles=...)` requires valid WebVTT (`.vtt`) format.
* Converts comma decimal separators (`00:00:01,000`) to dot decimal separators (`00:00:01.000`).
* Inserts the standard `WEBVTT` header.
* Preserves cue indices and line breaks.

### 4.2 Resource Caching (`src/subtitle_nmt_thesis/ui/state.py`)
* Use `@st.cache_resource` for model instantiations:
  ```python
  @st.cache_resource
  def get_translator(model_name: str, src_lang: str, tgt_lang: str):
      return _build_model(model_name, src_lang, tgt_lang)
  ```
* Avoid reloading heavy PyTorch weights when user adjusts UI sliders or navigates tabs.

### 4.3 Pre-packaged Demo Assets (`assets/`)
* A lightweight, royalty-free educational sample video clip (`assets/demo.mp4`, < 2MB, ~10s duration) demonstrating lecture speech.
* Matching English subtitle file (`assets/sample_en.srt`) with typical educational terminology (e.g. data structures, algorithms).
* Reference Hindi subtitle file (`assets/sample_hi.srt`).

---

## 5. Dependencies & Packaging

* **Dependencies in `pyproject.toml`**:
  * Add `streamlit` to dependencies (or optional group `ui = ["streamlit>=1.35.0"]`).
* **CLI Command**:
  * Register entry point in `[project.scripts]`:
    ```toml
    ui = "subtitle_nmt_thesis.ui.app:run_app"
    ```
  * `run_app()` invokes `streamlit.web.cli.main()` programmatically with the path to `app.py`.

---

## 6. Testing & Verification

1. **Unit Tests (`tests/test_ui.py`)**:
   * Test `srt_to_vtt` conversion accuracy with various timestamp inputs.
   * Test compliance metric calculations (CPL/CPS violations, empty strings, zero duration).
   * Test demo asset loader helpers (presence and readability of demo assets).
2. **Integration & Mock Tests**:
   * Mock translator responses to verify UI component rendering without downloading full HF models during automated tests.
3. **End-to-End Verification**:
   * Launch Streamlit app on local port (`uv run streamlit run src/subtitle_nmt_thesis/ui/app.py`).
   * Verify Tab 1 translates sample cue and displays constraint badges.
   * Verify Tab 2 runs pipeline on demo assets, shows video with subtitles, and allows downloading `.srt`.
   * Verify Tab 3 displays metrics cards correctly.
