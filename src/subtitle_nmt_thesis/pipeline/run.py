from pathlib import Path
from subtitle_nmt_thesis.data.parser import parse_srt, write_srt, SubtitleItem
from subtitle_nmt_thesis.data.context_builder import ContextBuilder
from subtitle_nmt_thesis.data.preprocess import clean_text
from subtitle_nmt_thesis.models.baseline import BaselineTranslator
from subtitle_nmt_thesis.constraints.cpl import CPLConstraint
from subtitle_nmt_thesis.constraints.cps import CPSConstraint
from subtitle_nmt_thesis.constraints.controller import ConstraintController
from subtitle_nmt_thesis.constraints.reranker import Reranker
from subtitle_nmt_thesis.models.translator import Translator
from subtitle_nmt_thesis.utils.logging import setup_logger

logger = setup_logger("pipeline")


class Pipeline:
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-ROMANCE",
        max_cpl: int = 42,
        max_cps: int = 21,
        window_size: int = 2,
        use_reranker: bool = True,
        num_candidates: int = 5,
        constraint_weights: dict[str, float] | None = None,
    ):
        logger.info("Initializing pipeline (model=%s, cpl=%d, cps=%d, reranker=%s)",
                     model_name, max_cpl, max_cps, use_reranker)
        self.model = BaselineTranslator(model_name=model_name)
        self.context_builder = ContextBuilder(window_size=window_size)
        if use_reranker:
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
            translated = self.translator.translate(item.text, item=item)
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
    parser.add_argument("--model", type=str, default="Helsinki-NLP/opus-mt-en-ROMANCE")
    parser.add_argument("--max-cpl", type=int, default=42)
    parser.add_argument("--max-cps", type=int, default=21)
    args = parser.parse_args()
    pipeline = Pipeline(model_name=args.model, max_cpl=args.max_cpl, max_cps=args.max_cps)
    pipeline.run(args.input, args.output)
