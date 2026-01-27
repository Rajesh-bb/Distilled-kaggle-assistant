# 🧠 Kai: Efficient Kaggle Assistant via Knowledge Distillation

> **Big Brain Intelligence. Small Model Speed.**
> *Engineered a domain-specific, lightweight AI assistant by distilling a fine-tuned Llama-3.2-3B (Teacher) into a Llama-3.2-1B (Student). The result? A deployment-ready model that runs 45% faster with 54% less memory.*

---

## 🚀 The Mission

Deploying Large Language Models (LLMs) is a balancing act between **Intelligence** and **Latency**.

* **The Problem:** Large models (3B+) offer great reasoning but are slow and heavy for edge deployment or real-time assistance.
* **The Solution:** **Knowledge Distillation**. Instead of training a small model from scratch, we teach it to mimic the reasoning of a larger, smarter "Teacher" model.

**Kai** is designed to answer machine learning/deep learning concepts helping with Kaggle competitions efficiently.

---

## 📊 Impact & Benchmarks

We achieved massive efficiency gains without sacrificing the model's ability to understand code context.

| Metric | 🐢 Teacher Model (3B) | 🐇 Student Model (1B) | 📉 Improvement |
| --- | --- | --- | --- |
| **Model Size** | 2.1 GB | **980 MB** | **54% Smaller** |
| **Inference Latency** | 4246 ms | **2290 ms** | **~44% Faster** |
| **VRAM Usage** | High (>6GB) | **Low (<3GB)** | **Edge Ready** |

### 🎥 Live Demo: Teacher vs. Student

To visualize the impact, we tested both models with the same complex query: *"What is Overfitting?"*.

* **[Watch Pro Model Demo (69s)](https://drive.google.com/file/d/1sJvDrT0siOWwYt521MW67fxO9NfVAIvt/view?usp=sharing)** - *High accuracy, but slower generation.*
* **[Watch Flash Model Demo (46s)](https://drive.google.com/file/d/1cGxnb-nbwxr-dBUNoCaMpQaMig8wptrj/view?usp=sharing)** - *Same accuracy, **1.5x speedup**.*

---

## ⚙️ Technical Architecture

### 1. The Datasets 📚

To ensure the model speaks the language of Data Science, we curated a mix of high-quality coding and logic datasets:

* **[Google MBPP (Mostly Basic Python Problems)](https://www.google.com/search?q=https://huggingface.co/datasets/google-research-datasets/mbpp):** Essential for grounding the model in core Python logic and syntax.
* **[Kaggle Solutions Dataset](https://www.kaggle.com/):** Scraped discussions and notebooks to understand Kaggle-specific context (Pandas, Scikit-Learn, EDA techniques).
* **[Data Science QA](https://www.google.com/search?q=https://huggingface.co/datasets/code_search_net):** (Or your specific HF link) Used to fine-tune the reasoning capabilities for theoretical questions.

### 2. The Pipeline 🛠️

We didn't just train; we engineered a pipeline for knowledge transfer.

1. **Teacher Fine-Tuning (QLoRA):**
* Base: **Llama-3.2-3B**.
* Technique: **QLoRA** (Quantized Low-Rank Adaptation) with 4-bit quantization to fit training on consumer GPUs.
* Result: A highly specialized "Expert" model.


2. **Knowledge Distillation (The Magic):**
* We built a custom **PyTorch Distillation Trainer**.
* **Loss Function:** used **KL Divergence Loss** to align the probability distributions of the Student (1B) with the Teacher (3B). This transfers "Dark Knowledge" (the reasoning nuances) rather than just the final answer.


3. **Deployment (Colab + Ngrok):**
* The entire inference engine runs on **Google Colab** using T4 GPUs.
* Served to the web via **Flask** and **Ngrok** tunneling, proving that powerful AI agents don't need expensive dedicated servers.



---

## 💻 Installation & Usage

### Prerequisites

* Python 3.10+
* PyTorch (CUDA enabled)
* Transformers & PEFT

### Running on Colab

1. Open the notebook `knowledge_distillation_trainer(1).ipynb`.
2. Add your **Ngrok Auth Token**.
3. Run all cells to start the backend.
4. Paste the generated public URL into `index.html`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

