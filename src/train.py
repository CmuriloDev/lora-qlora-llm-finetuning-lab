import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_PATH = "dataset/train.jsonl"
TEST_PATH = "dataset/test.jsonl"
SAVE_DIR = "artifacts/lora"


def format_row(row):
    return f"### Instruction:\n{row['prompt']}\n\n### Answer:\n{row['response']}"


def execute():
    data = load_dataset("json", data_files={"train": TRAIN_PATH, "test": TEST_PATH})

    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=quant,
        device_map="auto",
    )

    model.config.use_cache = False

    lora = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        peft_config=lora,
        args=args,
        formatting_func=lambda x: [format_row(x)],
    )

    trainer.train()
    trainer.model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Training finished")


if __name__ == "__main__":
    execute()