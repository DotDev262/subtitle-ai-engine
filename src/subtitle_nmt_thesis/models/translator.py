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
        self.context_builder = context_builder

    def translate(self, text: str, item=None) -> str:
        if self.reranker and not item:
            candidates = self.model.translate_n(text, n=self.reranker.num_candidates)
            ranked = self.reranker.rerank(candidates, item)
            return ranked[0][0]
        return self.model.translate(text)
