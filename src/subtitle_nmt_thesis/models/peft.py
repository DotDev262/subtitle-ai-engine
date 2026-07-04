import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
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


def load_parallel_data(path: str) -> Dataset:
    pairs = []
    for line in Path(path).read_text().strip().splitlines():
        if line.strip():
            pairs.append(json.loads(line))
    return Dataset.from_list(pairs)


def preprocess_function(examples: dict, tokenizer, source_key: str = "source", target_key: str = "target",
                        max_length: int = 128):
    inputs = tokenizer(examples[source_key], truncation=True, max_length=max_length, padding=False)
    targets = tokenizer(examples[target_key], truncation=True, max_length=max_length, padding=False)
    inputs["labels"] = targets["input_ids"]
    return inputs


def train(
    base_model_name: str = "Helsinki-NLP/opus-mt-en-ROMANCE",
    train_file: str = "train.jsonl",
    output_dir: str = "peft-adapter",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 128,
    lora_r: int = 8,
    lora_alpha: int = 32,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r, lora_alpha=lora_alpha, target_modules=["q", "v"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_parallel_data(train_file)
    tokenized = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        predict_with_generate=True,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PEFT adapter for subtitle translation")
    parser.add_argument("--base-model", type=str, default="Helsinki-NLP/opus-mt-en-ROMANCE")
    parser.add_argument("--train-file", type=str, default="train.jsonl", help="JSONL with source/target pairs")
    parser.add_argument("--output-dir", type=str, default="peft-adapter")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()
    train(
        base_model_name=args.base_model,
        train_file=args.train_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
