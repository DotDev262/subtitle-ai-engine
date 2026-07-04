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

    def translate(self, text: str, item=None) -> str:
        if self.reranker:
            candidates = self.model.translate_n(text, n=self.reranker.num_candidates)
            ranked = self.reranker.rerank(candidates, item)
            return ranked[0][0]
        return self.model.translate(text)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Translate subtitles using the pipeline")
    parser.add_argument("input", type=str, help="Input SRT file")
    args = parser.parse_args()
    print(f"Translating {args.input} — use 'pipeline' command instead")
