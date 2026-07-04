from collections.abc import Callable


class ConstraintController:
    def __init__(
        self,
        constraints: dict[str, Callable],
        weights: dict[str, float],
    ):
        self.constraints = constraints
        self.weights = weights

    def evaluate(self, text: str, item=None) -> float:
        total = 0.0
        for name, constraint_fn in self.constraints.items():
            w = self.weights.get(name, 1.0)
            total += w * constraint_fn(text, item)
        return total
