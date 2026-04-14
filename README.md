LoRA & QLoRA Fine-Tuning Project
Project Description

This repository contains an implementation of a lightweight fine-tuning workflow for language models using parameter-efficient techniques.

The focus is to show how models can be adapted with reduced computational cost by combining LoRA with quantization strategies such as QLoRA.

Academic Context

Course: Artificial Intelligence Topics
Instructor: Dimmy Magalhães
Institution: Faculdade iCEV
Author: Carlos Murilo

Main Goals
Create a synthetic dataset for training
Apply low-bit quantization (4-bit)
Configure and integrate LoRA layers
Perform supervised fine-tuning with TRL
Export the trained adapter
Tech Stack
Python
PyTorch
Hugging Face Transformers
PEFT
TRL (Transformer Reinforcement Learning)
BitsAndBytes
Execution Guide

Install the required dependencies:

pip install -r requirements.txt

Run the full pipeline:

python main.py
Output Results

After execution, the project should produce:

Generated dataset files (train.jsonl and test.jsonl)
Training logs displayed in the terminal
Observable loss progression during training
Fine-tuned adapter stored in artifacts/lora/
Use of AI Tools

AI-assisted tools were used during development for:

Supporting dataset generation structure
Assisting in configuration of LoRA and QLoRA parameters
Helping identify and resolve implementation issues

All outputs were reviewed, validated, and adjusted manually by Carlos Murilo.

Release

Version 1.0