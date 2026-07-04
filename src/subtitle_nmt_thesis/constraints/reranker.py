from subtitle_nmt_thesis.constraints.controller import ConstraintController


class Reranker:
    def __init__(self, controller: ConstraintController, num_candidates: int = 5):
        self.controller = controller
        self.num_candidates = num_candidates

    def rerank(
        self, candidates: list[str], item=None
    ) -> list[tuple[str, float]]:
        scored = [(c, self.controller.evaluate(c, item)) for c in candidates]
        return sorted(scored, key=lambda x: x[1])
