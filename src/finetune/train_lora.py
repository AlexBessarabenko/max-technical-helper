"""LoRA fine-tuning of Qwen2.5-Instruct on the corporate SFT dataset (CPU-only).

Usage:
    .venv/bin/python -m src.finetune.train_lora

Env overrides:
    FINETUNE_BASE_MODEL   — base model HF id (default: settings.finetune_base_model)
    FINETUNE_MAX_EXAMPLES — truncate dataset to N examples (smoke runs)
    FINETUNE_EPOCHS       — number of training epochs (default: 2)
    FINETUNE_OUTPUT_DIR   — adapter output dir (default: models/lora_adapter)
    FINETUNE_DTYPE        — float32 | bfloat16 | auto (default: auto — fp32 only
                            when >=12 GB RAM is available, else bfloat16)
"""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.config import get_settings

MAX_LEN = 768
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _resolve_dtype() -> torch.dtype:
    requested = os.environ.get("FINETUNE_DTYPE", "auto").lower()
    if requested != "auto":
        return getattr(torch, requested)
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 2**30
    except Exception:
        available_gb = 0.0
    # fp32 weights of a 1.5B model alone take ~6.2 GB; bf16 halves that.
    return torch.float32 if available_gb >= 12 else torch.bfloat16


def _prepare_dataset(dataset_path: str, tokenizer, max_examples: int | None):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    def to_text(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=MAX_LEN)

    dataset = dataset.map(to_text)
    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main() -> None:
    settings = get_settings()
    base_model = os.environ.get("FINETUNE_BASE_MODEL", settings.finetune_base_model)
    max_examples = int(os.environ.get("FINETUNE_MAX_EXAMPLES", "0")) or None
    epochs = float(os.environ.get("FINETUNE_EPOCHS", "2"))
    output_dir = os.environ.get("FINETUNE_OUTPUT_DIR", "models/lora_adapter")
    dtype = _resolve_dtype()

    print(f"Base model: {base_model} | dtype: {dtype} | output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = _prepare_dataset(str(settings.finetune_dataset_path), tokenizer, max_examples)
    print(f"Dataset: {len(dataset)} examples, max_len={MAX_LEN}")

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="models/lora_runs",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=1e-4,
            logging_steps=5,
            use_cpu=True,
            report_to=[],
            save_strategy="no",
            dataloader_num_workers=0,
            seed=42,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
