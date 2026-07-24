from subtitle_nmt_thesis.models.baseline import BaselineTranslator
from subtitle_nmt_thesis.constraints.reranker import Reranker
from subtitle_nmt_thesis.data.context_builder import ContextBuilder


class Translator:
    def __init__(
        self,
        model: BaselineTranslator,
        reranker: Reranker | None = None,
        context_builder: ContextBuilder | None = None,
    ):
        self.model = model
        self.reranker = reranker
        self.context_builder = context_builder  # ponytail: reserved for future context-aware decoding

    def translate(self, text: str, item=None, context: str = "", constrained: bool = False, max_cpl: int = 42) -> str:
        input_text = context + text
        if constrained:
            return self.model.translate_constrained(input_text, max_cpl=max_cpl)
        if self.reranker and context:
            candidates = self.model.translate_n(input_text, n=self.reranker.num_candidates)
            ranked = self.reranker.rerank(candidates, item)
            return ranked[0][0]
        return self.model.translate(text)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Translate subtitles using the pipeline")
    parser.add_argument("input", type=str, help="Input SRT file")
    args = parser.parse_args()
    print(f"Translating {args.input} — use 'pipeline' command instead")
