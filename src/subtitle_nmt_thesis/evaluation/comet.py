from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluate import EvaluationModule


class COMETEvaluator:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "Unbabel/wmt22-comet-da"
        self._model: EvaluationModule | None = None

    def _lazy_load(self) -> None:
        if self._model is None:
            from evaluate import load
            self._model = load("comet", self.model_name)

    def score(self, sources: list[str], references: list[str], hypotheses: list[str]) -> dict:
        self._lazy_load()
        results = self._model.compute(
            sources=sources,
            references=references,
            predictions=hypotheses,
        )
        return {"comet": results["mean_score"]}
