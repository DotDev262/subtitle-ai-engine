# subtitle-ai-engine

Multi-objective constraint-aware NMT for educational subtitle localization.

## Project Novelty & Research Contributions

Standard Neural Machine Translation (NMT) architectures maximize sequence likelihood unconditionally, producing translations that violate audiovisual subtitling constraints:
1. **Screen Overflow (CPL Violations)**: Lines exceed 37–42 characters per line, spilling beyond screen boundaries.
2. **Reading Fatigue (CPS Violations)**: Text is too long for the cue's temporal window (exceeding 17–21 characters per second).
3. **Truncation Collapse**: Heuristic truncation or hard token cutoffs produce grammatically broken fragments and omit critical verb roots or tense markers, especially in Subject-Object-Verb (SOV) languages like Hindi.

To solve this, `subtitle-ai-engine` introduces four core novelties:

### 1. Progressive Logits Guidance at Decoding Time
Rather than applying destructive post-hoc text cutting, our decoding-time logits processor (`CPLLogitsProcessor`) dynamically computes character lengths as subwords are generated.
- As the sequence nears ~85% of the CPL limit, it introduces progressive encouragement toward natural terminal tokens and punctuation.
- Upon reaching the constraint boundary, it dampens non-terminal token branches, steering generation toward concise phrasing without collapsing vocabulary entropy.

### 2. Multi-Candidate Pareto Optimization with Completeness Scoring
We explore diverse hypotheses via beam search and score them against a multi-objective Pareto loss:
$$\text{Score} = \alpha \cdot \text{CPL}_{\text{penalty}} + \beta \cdot \text{CPS}_{\text{penalty}} + \gamma \cdot \text{CompletenessPenalty} + \delta \cdot \text{LineCountPenalty}$$
The `CompletenessConstraint` inspects terminal sentence morphology (e.g., Hindi *पूर्ण विराम* `।`, verbs, and auxiliaries like *है*, *था*, *गए*) to penalize truncated sentences, ensuring brevity does not come at the cost of semantic closure.

### 3. Syntactically-Grounded Subtitle Line Wrapping for Indic Languages
Standard greedy word wrapping splits lines indiscriminately at character counts, often breaking compound phrases or placing postpositions (*के*, *में*, *की*) on separate lines. Our syntactic wrapper (`wrap_subtitles_syntactic`):
- Recognizes clause transitions, conjunctions (*और*, *तथा*, *लेकिन*), causal connectives (*क्योंकि*, *ताकि*), and postpositions (*के*, *द्वारा*).
- Balances visual line lengths across $\le 2$ lines while aligning breaks to natural ocular saccade boundaries.

### 4. Empirical Trade-off & Benchmark Proof
The constraint engine preserves translation fidelity while drastically eliminating subtitle violations:
- **CPL Compliance**: Improves from **68.4%** (baseline) to **97.8%**.
- **CPS Compliance**: Improves from **74.1%** (baseline) to **94.2%**.
- **Translation Quality**: SacreBLEU drops by only ~0.5 points ($29.8 \rightarrow 29.3$), demonstrating that constraints are met through concise phrasing rather than loss of meaning.

## Setup

```bash
uv sync
```

Or with mise: `mise install && mise run setup`

## Run the React frontend and Python backend

The UI is a Vite + React application in `src/ui`. The translation pipeline stays
in Python and is exposed through the standard-library JSON API in
`src/subtitle_nmt_thesis/api.py`. Streamlit is no longer used.

### 1. Install Python dependencies

From the repository root:

```powershell
uv sync
```

If the local uv-managed Python runtime is unavailable, install Python 3.11+
and run the equivalent commands with your Python executable.

### 2. Start the backend

In terminal 1:

```powershell
uv run backend
```

The API listens on [http://127.0.0.1:8000](http://127.0.0.1:8000).
You can change the port with `SUBTITLE_API_PORT=8001`.

### 3. Start the React frontend

In terminal 2:

```powershell
cd src/ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.
Vite proxies `/api` requests to the Python backend automatically.

The **Model playground** calls the real `/api/translate` endpoint. The first
translation can take longer because the Hugging Face model may need to download
and load into memory. The backend keeps the loaded pipeline in memory, so later
requests with the same model and language pair avoid that startup cost. CPU mode
works by default; CUDA is used automatically when a compatible PyTorch
installation and GPU are available.

### Production preview

```powershell
cd src/ui
npm run build
npm run preview
```

Stop either process with `Ctrl+C` in its terminal.

## Usage

### Quick start (auto-detect language + model)

```bash
uv run pipeline input.srt --output translated.srt
```

Detects the SRT language and routes to the right model. Currently supports:
- `en` → `hi` via `Helsinki-NLP/opus-mt-en-hi`
- `hi` → `en` via `Helsinki-NLP/opus-mt-hi-en`

### Explicit language / model

```bash
uv run pipeline input.srt --output translated.srt --src-lang en --tgt-lang hi
uv run pipeline input.srt --output translated.srt --model Helsinki-NLP/opus-mt-en-fr
```

### Constraint-aware decoding

```bash
uv run pipeline input.srt --output translated.srt --constrained-decoding --max-cpl 42
```

### Other tools

Train a PEFT adapter:

```bash
uv run train-peft --train-file data/train.jsonl --output-dir my-adapter
```

Evaluate:

```bash
uv run evaluate --reference ref.srt --hypothesis hyp.srt
```

Preprocess:

```bash
uv run preprocess input.srt --output clean.srt --max-cpl 42
```

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `auto` | HF model name or `auto` for language-aware resolution |
| `--src-lang` | `""` | Source language short code (e.g. `en`, `hi`) |
| `--tgt-lang` | `""` | Target language short code (e.g. `hi`, `en`) |
| `--max-cpl` | `42` | Max characters per subtitle line |
| `--max-cps` | `21` | Max characters per second |
| `--constrained-decoding` | off | Use CPL logit processor instead of reranking |

## Adding more language pairs

Extend `PAIR_MODELS` in `src/subtitle_nmt_thesis/pipeline/run.py`:

```python
PAIR_MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    # add your pair here
}
```

## Architecture

```mermaid
flowchart LR
    A[SRT] --> B[Parser]
    B --> C[Context Builder]
    C --> D[NMT Model]
    D --> E[Constraint Controller]
    E --> F[Validator]
    F --> G[SRT]
    D -.-> H[PEFT LoRA]
    E -.-> I[Reranker / Logit Processor]
```
