from subtitle_nmt_thesis.ui.components.metrics_view import get_benchmark_comparison_data


def test_benchmark_data():
    data = get_benchmark_comparison_data()
    assert "baseline" in data
    assert "constrained" in data
    assert data["constrained"]["cpl_compliance"] > data["baseline"]["cpl_compliance"]
