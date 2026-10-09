from unittest.mock import patch, MagicMock
from subtitle_nmt_thesis.ui.app import build_pipeline_instance
from subtitle_nmt_thesis.ui.state import get_device_status


@patch("subtitle_nmt_thesis.pipeline.run._build_model")
def test_pipeline_factory(mock_build):
    mock_build.return_value = MagicMock()
    pipeline = build_pipeline_instance(max_cpl=40, max_cps=20, src_lang="en", tgt_lang="hi")
    assert pipeline.max_cpl == 40
    assert pipeline.src_lang == "en"
    assert pipeline.tgt_lang == "hi"


def test_device_status_string():
    dev, is_cuda = get_device_status()
    assert dev in ("cuda", "cpu")
