

# Adversarial Reversal Generation (ARG)

<div align="center">

**A Data-Centric Strategy to Balance Helpfulness and Harmlessness in Large Language Models**

</div>

## 📖 Introduction

This repository contains the official implementation of the paper **"Adversarial Reversal Generation: A Data-Centric Strategy to Balance Helpfulness and Harmlessness in Large Language Models"** (ACL 2026 Submission).

Large Language Models (LLMs) often struggle with the "alignment tax"—the trade-off where increasing safety leads to over-refusal on safe queries. **ARG** addresses this by constructing a dataset of **bidirectional, semantically aligned instruction pairs** with flipped safety attributes (Safe  Unsafe).

By combining a **Multi-Agent Safety Reasoning Pipeline** with **Safety Contrastive Regularization (SCR)**, ARG allows models to learn precise safety decision boundaries. Empirical results show that ARG reduces harmful responses by **78%** and cuts over-refusal on safe inputs by **65%**, outperforming methods like Safe-RLHF and commercial baselines on fine-grained boundaries.

## 🚀 Key Features

* **🔄 Bidirectional Reversal:** A data augmentation method that flips safety attributes (Safe  Unsafe and Unsafe  Safe) while preserving the semantic structure of the query.


* **🤖 Multi-Agent Pipeline:** A collaborative framework involving Analysis, Reversal, Answer, and Validation agents to ensure high-quality, minimal-edit data generation.


* **🧠 SB-CoT:** **Safety Boundary-aware Chain-of-Thought** integrated into training, forcing the model to explicitly reason about intent and harm before deciding to refuse or comply.


* **📐 SCR (Safety Contrastive Regularization):** A novel training objective that enforces geometric separation between helpful and harmful intents in the latent space.



## 🛠️ Framework Overview

Figure 1: The overall framework of ARG, showing the data construction via bidirectional reversal and the multi-agent safety reasoning pipeline.

The ARG method consists of three main components:

1. **Data Generation:** Generating  pairs where  and  share semantics  but have opposite safety attributes .
2. **SB-CoT Annotation:** Generating a rationale tuple  for every response.
3. **Training:** Optimizing a unified objective consisting of SFT loss and SCR regularization terms.

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ARG-Safety.git
cd ARG-Safety

```


2. **Create a virtual environment:**
```bash
conda create -n arg_env python=3.10
conda activate arg_env

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## 🏗️ Data Construction (ARG Pipeline)

To generate the adversarial reversal dataset using the Multi-Agent pipeline:


**Pipeline Stages:**

1. **Analysis Agent:** Extracts semantic utility () and safety attribute ().
2. **Reversal Agent:** Rewrites the query to  conditioning on the inverted attribute .
3. **Answer Agent:** Generates the response  and the SB-CoT rationale.
4. 
**Validation Agent:** Accepts/rejects pairs based on semantic preservation and boundary consistency .



## 🚄 Training

We employ **Safety Contrastive Regularization (SCR)** alongside standard SFT. The training objective is defined as:

To run the fine-tuning (default backbone: **Qwen2.5-7B** ):



**Loss Components:**

* `--lambda_dir`: Controls **Safety Direction Learning** (Geometric Separation).


* `--lambda_cons`: Controls **Representation Consistency** (Manifold Regularization).


* `--lambda_mi`: Controls **Information Separation** (Feature Disentanglement).



## 📊 Evaluation

### Supported Benchmarks

We evaluate on a Hybrid Evaluation Suite:

1. **HarmBench:** Adversarial Robustness (Safety OOD).
2. **XSTest:** Over-refusal check (Boundary Precision).
3. **MT-Bench:** General Utility.
4. **ARG-Boundary:** Internal fine-grained soft-refusal metric.

### Run Evaluation

```bash
python eval/run_benchmarks.py \
    --model_path checkpoints/arg-qwen-7b \
    --benchmarks harmbench,xstest,mt_bench

```

### Main Results

Comparison based on Qwen2.5-7B backbone:

| Model | HarmBench (ASR) ⬇️ | XSTest (Over-Refusal) ⬇️ | MT-Bench (Utility) ⬆️ |
| --- | --- | --- | --- |
| **Base Model** | 58.4% | 1.2% | 7.61 |
| **Safe-RLHF** | 4.2% | 14.5% | 7.45 |
| **ARG (Ours)** | **6.5%** | **2.5%** | **7.85** |

> **Note:** ARG achieves a pareto-optimal balance, significantly lowering attack success rates without the high "safety tax" (over-refusal) seen in RLHF methods.
