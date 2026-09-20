from pathlib import Path
from subtitle_nmt_thesis.data.parser import parse_srt, write_srt, SubtitleItem
from subtitle_nmt_thesis.data.context_builder import ContextBuilder
from subtitle_nmt_thesis.models.baseline import BaselineTranslator
from subtitle_nmt_thesis.constraints.cpl import CPLConstraint
from subtitle_nmt_thesis.constraints.cps import CPSConstraint
from subtitle_nmt_thesis.constraints.controller import ConstraintController
from subtitle_nmt_thesis.constraints.reranker import Reranker
from subtitle_nmt_thesis.models.translator import Translator
from subtitle_nmt_thesis.utils.logging import setup_logger
from subtitle_nmt_thesis.utils.lang_detector import detect_language

logger = setup_logger("pipeline")

PAIR_MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
}


def _auto_resolve_model(model_name: str, src_lang: str, tgt_lang: str) -> str:
    if model_name != "auto":
        return model_name
    pair = (src_lang, tgt_lang)
    resolved = PAIR_MODELS.get(pair)
    if resolved:
        logger.info("Auto-resolved model for %s → %s: %s", src_lang, tgt_lang, resolved)
        return resolved
    logger.warning("No auto-resolved model for %s → %s, using '%s' as-is", src_lang, tgt_lang, model_name)
    return model_name


def _detect_and_route(input_path: Path, model_name: str, src_lang: str, tgt_lang: str) -> tuple[str, str, str]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    resolved_model = model_name
    resolved_src = src_lang
    resolved_tgt = tgt_lang
    if model_name == "auto":
        if not src_lang or not tgt_lang:
            detected = detect_language(input_path)
            if detected == "en":
                resolved_src, resolved_tgt = "en", "hi"
            elif detected == "hi":
                resolved_src, resolved_tgt = "hi", "en"
            else:
                raise ValueError(f"Detected unsupported language '{detected}'. Pass --src-lang/--tgt-lang explicitly.")
        resolved_model = _auto_resolve_model(model_name, resolved_src, resolved_tgt)
    elif not src_lang or not tgt_lang:
        resolved_model = _auto_resolve_model(model_name, src_lang, tgt_lang)
    return resolved_model, resolved_src, resolved_tgt


def _build_model(model_name: str, src_lang: str, tgt_lang: str):
    if model_name.startswith("ai4bharat/indictrans2") or model_name == "indictrans2":
        from subtitle_nmt_thesis.models.indictrans import IndicTransTranslator

        return IndicTransTranslator(src_lang=src_lang, tgt_lang=tgt_lang)
    return BaselineTranslator(model_name=model_name)


class Pipeline:
    def __init__(
        self,
        model_name: str = "auto",
        max_cpl: int = 42,
        max_cps: int = 21,
        window_size: int = 2,
        use_reranker: bool = True,
        num_candidates: int = 5,
        constraint_weights: dict[str, float] | None = None,
        constrained_decoding: bool = False,
        src_lang: str = "",
        tgt_lang: str = "",
        input_path: str | Path | None = None,
    ):
        resolved_model, resolved_src, resolved_tgt = _detect_and_route(
            Path(input_path) if input_path else Path("."),
            model_name,
            src_lang,
            tgt_lang,
        )
        logger.info("Initializing pipeline (model=%s, src=%s, tgt=%s, cpl=%d, cps=%d, reranker=%s, constrained=%s)",
                     resolved_model, resolved_src, resolved_tgt, max_cpl, max_cps, use_reranker, constrained_decoding)
        self.resolved_model = resolved_model
        self.src_lang = resolved_src
        self.tgt_lang = resolved_tgt
        self.max_cpl = max_cpl
        self.constrained_decoding = constrained_decoding
        self.model = _build_model(resolved_model, resolved_src, resolved_tgt)
        self.context_builder = ContextBuilder(window_size=window_size)
        if use_reranker and not constrained_decoding:
            constraints = {
                "cpl": CPLConstraint(max_cpl=max_cpl).score,
                "cps": CPSConstraint(max_cps=max_cps).score,
            }
            weights = constraint_weights or {"cpl": 1.0, "cps": 1.0}
            controller = ConstraintController(constraints=constraints, weights=weights)
            reranker = Reranker(controller=controller, num_candidates=num_candidates)
        else:
            reranker = None
        self.translator = Translator(
            model=self.model, reranker=reranker, context_builder=self.context_builder
        )

    def run(self, input_path: str | Path, output_path: str | Path) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)
        logger.info("Parsing %s", input_path)
        items = parse_srt(input_path)
        logger.info("Translating %d subtitles", len(items))
        translations = []
        for i, item in enumerate(items):
            logger.debug("[%d/%d] Translating: %s", i + 1, len(items), item.text[:50])
            ctx = self.context_builder.build(items, i)
            context_str = self.context_builder.format_context(ctx)
            translated = self.translator.translate(
                item.text, item=item, context=context_str,
                constrained=self.constrained_decoding, max_cpl=self.max_cpl,
            )
            translations.append(translated)
        translated_items = [
            SubtitleItem(id=item.id, start=item.start, end=item.end, text=t)
            for item, t in zip(items, translations)
        ]
        write_srt(translated_items, output_path)
        logger.info("Written to %s", output_path)
        return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run subtitle translation pipeline")
    parser.add_argument("input", type=str, help="Input SRT file")
    parser.add_argument("--output", type=str, default="output.srt", help="Output SRT path")
    parser.add_argument("--model", type=str, default="auto", help="Model name or 'auto'/nllb' for auto-detection + NLLB")
    parser.add_argument("--src-lang", type=str, default="", help="Source language (e.g. en, hi)")
    parser.add_argument("--tgt-lang", type=str, default="", help="Target language (e.g. hi, en)")
    parser.add_argument("--max-cpl", type=int, default=42)
    parser.add_argument("--max-cps", type=int, default=21)
    parser.add_argument("--constrained-decoding", action="store_true", help="Use CPL logit processor instead of reranking")
    args = parser.parse_args()
    pipeline = Pipeline(
        model_name=args.model,
        max_cpl=args.max_cpl,
        max_cps=args.max_cps,
        constrained_decoding=args.constrained_decoding,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        input_path=args.input,
    )
    pipeline.run(args.input, args.output)
