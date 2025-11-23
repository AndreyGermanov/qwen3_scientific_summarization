# Fine Tune Qwen3-1.7B LLM for scientific summarization

This project demonstrates the application of parameter-efficient fine-tuning (PEFT) to adapt the **Qwen3-1.7B-Base** large language model for the task of scientific article summarization. Leveraging **4-bit quantization**, **LoRA (Low-Rank Adaptation)**, and **QLoRA** techniques, I successfully fine-tune the model on a subset of the PubMed summarization dataset

# Contents

`train.ipynb` - This notebook shows full Fine Tuning pipeline of the base **Qwen3-1.7B-Base** model, using QLora.

`functions.py` - Additional functions, used by previous notebook

`inference.ipynb` - This notebook shows how to use fine-tuned model for inference.

`dataset` - This folder contains the dataset, used for fine tuning.

# How to use

Using Jupyter notebook run the notebooks, described above cell by cell.

# References

Article: 

Model card: https://huggingface.co/GermanovDev/qwen3-pubmed-summarization
