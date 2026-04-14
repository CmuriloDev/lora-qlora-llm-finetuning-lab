# LoRA and QLoRA Fine-Tuning Lab

## Overview

This project implements a simplified pipeline for fine-tuning a language model using parameter-efficient techniques such as LoRA and QLoRA.

The goal is to demonstrate how large models can be adapted using low computational resources.

---

## Academic Information

Academic project for the course Artificial Intelligence Topics
Professor: Dimmy Magalhães  
Institution: Faculdade iCEV  
Student: Carlos Murilo  

---

## Objectives

* Generate a synthetic dataset
* Apply 4-bit quantization (QLoRA)
* Configure LoRA parameters
* Train using SFTTrainer
* Save the trained adapter

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* TRL
* BitsAndBytes

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

## Expected Output

* Dataset file created (`dataset.jsonl`)
* Training logs printed
* Loss decreasing during training
* Adapter saved in `lora-adapter/`

---

## AI-Assisted Complementary Support

AI tools were used to assist in:

* Structuring the dataset generation logic
* Clarifying configuration of LoRA and QLoRA parameters
* Debugging minor issues in training setup

All changes were reviewed, tested and validated by:

Carlos Murilo Nogueira Portela

---

## Version

v1.0
