import torch
from subtitle_nmt_thesis.models.baseline import CPLLogitsProcessor


def test_cpl_processor_progressive_penalty():
    processor = CPLLogitsProcessor(max_cpl=40, eos_token_id=0)
    input_ids = torch.tensor([[10, 20, 30]])
    scores = torch.zeros(1, 100)
    out_scores = processor(input_ids, scores)
    # Shouldn't completely zero out vocabulary prematurely with hard -inf
    assert not torch.all(out_scores[:, 1:] == -float("inf"))
