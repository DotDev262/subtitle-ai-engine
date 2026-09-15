from unittest.mock import MagicMock
from subtitle_nmt_thesis.ui.state import get_device_status, run_single_comparison


def test_device_status():
    device, is_cuda = get_device_status()
    assert device in ("cuda", "cpu")
    assert isinstance(is_cuda, bool)


def test_run_single_comparison():
    mock_model = MagicMock()
    mock_model.translate.return_value = "Unconstrained output that is quite lengthy and detailed."
    mock_model.translate_constrained.return_value = "Constrained output."

    res = run_single_comparison(
        text="Input text",
        model=mock_model,
        max_cpl=42,
        duration_sec=3.0,
    )
    assert res["baseline_text"] == "Unconstrained output that is quite lengthy and detailed."
    assert res["constrained_text"] == "Constrained output."
    assert "baseline_metrics" in res
    assert "constrained_metrics" in res
