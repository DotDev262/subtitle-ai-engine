from subtitle_nmt_thesis.data.download_cli import parse_args


def test_download_cli_defaults():
    args = parse_args(["--num-samples", "25"])
    assert args.num_samples == 25
    assert args.dataset == "nptel"
    assert args.split == "test"
