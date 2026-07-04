from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


def create_peft_model(
    base_model_name: str,
    r: int = 8,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q", "v"],
    )
    return get_peft_model(model, peft_config)


# ponytail: load_peft_model duplicates model load but keeps interface clean
def load_peft_model(base_model_name: str, adapter_path: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    return PeftModel.from_pretrained(model, adapter_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PEFT adapter for subtitle translation")
    parser.add_argument("--base-model", type=str, default="Helsinki-NLP/opus-mt-en-ROMANCE")
    args = parser.parse_args()
    print(f"PEFT training not yet implemented (base model: {args.base_model})")
