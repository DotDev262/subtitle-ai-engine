"""Benchmark data shared by API consumers and research reporting."""


def get_benchmark_comparison_data() -> dict:
    return {
        "baseline": {"cpl_compliance": 68.4, "cps_compliance": 74.1, "avg_cpl": 47.2, "bleu": 29.8, "chrf": 54.6},
        "constrained": {"cpl_compliance": 97.8, "cps_compliance": 94.2, "avg_cpl": 38.5, "bleu": 29.3, "chrf": 54.1},
    }
