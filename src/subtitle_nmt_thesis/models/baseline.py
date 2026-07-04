import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class BaselineTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ROMANCE", device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline NMT model for subtitle translation")
    parser.add_argument("--model-name", type=str, default="Helsinki-NLP/opus-mt-en-ROMANCE")
    args = parser.parse_args()
    print(f"Baseline training not yet implemented (model: {args.model_name})")
