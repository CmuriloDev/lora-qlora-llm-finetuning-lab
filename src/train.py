import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer


def train():

    model_name = "facebook/opt-350m"

    use_gpu = torch.cuda.is_available()

    print(f"GPU disponível: {use_gpu}")

    if use_gpu:
        print("Usando QLoRA (4-bit)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    else:
        print("Rodando em CPU (sem quantização)...")

        model = AutoModelForCausalLM.from_pretrained(model_name)


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset(
        "json",
        data_files="data/dataset.jsonl",
        split="train"
    )

    def format_example(example):
        text = f"### Question:\n{example['prompt']}\n### Answer:\n{example['response']}"
        return [text]  


    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1 if not use_gpu else 2,
        num_train_epochs=1,
        learning_rate=2e-4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        report_to="none",
        fp16=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_example,
        args=training_args
    )

    trainer.train()

    os.makedirs("lora-adapter", exist_ok=True)
    trainer.model.save_pretrained("lora-adapter")

    print("Treino finalizado com sucesso!")