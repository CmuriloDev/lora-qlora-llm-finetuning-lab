# LoRA & QLoRA Fine-Tuning Project

## Project Description

This project presents a complete pipeline for fine-tuning a language model using parameter-efficient approaches. Instead of retraining all model weights, techniques such as LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are applied to significantly reduce computational cost while still allowing effective adaptation.

The implementation focuses on practicality, demonstrating how it is possible to train and adapt models even in constrained environments. It also highlights how quantization and efficient training strategies can be combined in real-world scenarios.

---

## Academic Context

Course: Artificial Intelligence Topics  
Instructor: Dimmy Magalhães  
Institution: Faculdade iCEV  
Author: Carlos Murilo 

---

## Main Goals

The purpose of this project goes beyond simply executing a training script. It aims to cover the full lifecycle of a lightweight fine-tuning process, including:

- Generating a synthetic dataset aligned with a specific domain
- Structuring the dataset into training and testing splits
- Applying 4-bit quantization to reduce memory usage
- Integrating LoRA layers into the base model
- Performing supervised fine-tuning using TRL’s SFTTrainer
- Saving and organizing the trained adapter for later use

---

## Tech Stack

The project is built using modern tools from the machine learning ecosystem:

- Python as the main programming language
- PyTorch for tensor operations and model execution
- Hugging Face Transformers for model loading and tokenization
- PEFT for parameter-efficient fine-tuning (LoRA)
- TRL for supervised fine-tuning workflows
- BitsAndBytes for low-bit quantization

---

## Execution Guide

To run the project locally, first install all dependencies:

```bash
pip install -r requirements.txt

Run the full pipeline:

python main.py
Output Results

After execution, the project should produce:

Generated dataset files (train.jsonl and test.jsonl)
Training logs displayed in the terminal
Observable loss progression during training
Fine-tuned adapter stored in artifacts/lora/
```
---

## Use of AI Tools

AI-assisted tools were used during development for:

Supporting dataset generation structure
Assisting in configuration of LoRA and QLoRA parameters
Helping identify and resolve implementation issues

All outputs were reviewed, validated, and adjusted manually by Carlos Murilo.

## Release

Version 1.0