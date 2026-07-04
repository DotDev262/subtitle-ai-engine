import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class CPLLogitsProcessor(LogitsProcessor):
    # ponytail: token→char ratio is a heuristic, not exact; refine with lang-specific constants if needed
    def __init__(self, max_cpl: int, chars_per_token: float = 2.0, eos_token_id: int = 0):
        self.max_tokens = int(max_cpl / chars_per_token)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] > self.max_tokens:
            scores[:, self.eos_token_id] += 10.0
        return scores


class BaselineTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ROMANCE", device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.eos_id = self.tokenizer.eos_token_id or 0

    def translate(self, text: str, source_lang: str = "", target_lang: str = "") -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_n(self, text: str, n: int = 5, source_lang: str = "", target_lang: str = "") -> list[str]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs, num_return_sequences=n, num_beams=n, diversity_penalty=0.3, do_sample=True
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def translate_constrained(self, text: str, max_cpl: int = 42) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        processor = CPLLogitsProcessor(max_cpl=max_cpl, eos_token_id=self.eos_id)
        outputs = self.model.generate(
            **inputs, logits_processor=LogitsProcessorList([processor]), num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline NMT model for subtitle translation")
    parser.add_argument("--model-name", type=str, default="Helsinki-NLP/opus-mt-en-ROMANCE")
    args = parser.parse_args()
    print(f"Baseline training not yet implemented (model: {args.model_name})")
