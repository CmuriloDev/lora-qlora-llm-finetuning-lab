from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = Path("artifacts/lora")


def load():
    if not ADAPTER.exists():
        raise RuntimeError("Adapter not found")

    tok_src = ADAPTER if (ADAPTER / "tokenizer_config.json").exists() else MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quant,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()

    return model, tokenizer


def run_inference(model, tokenizer, text):
    prompt = f"### Instruction:\n{text}\n\n### Answer:\n"

    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **tokens,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    model, tokenizer = load()

    query = "I want to book a training session tomorrow evening"
    result = run_inference(model, tokenizer, query)

    print("Input:")
    print(query)
    print("\nOutput:")
    print(result)


if __name__ == "__main__":
    main()