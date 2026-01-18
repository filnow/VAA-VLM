# Visual Affect Analysis: Predicting Emotions of Image Viewers with Vision–Language Models

<div align="center">
  <img src="figs/collage.png" alt="LAIGAI Images" width="600"/>
  <br>
  <em>Example images from LAI-GAI dataset</em>
</div>

<br>

<div align="center">

**Official repository for the paper: *Visual Affect Analysis: Predicting Emotions of Image Viewers with Vision–Language Models***

</div>

---

## 📑 Abstract

Vision-language models (VLMs) show promise as tools for inferring affect from visual stimuli on a large scale; it is not yet clear how closely their outputs align with human affective ratings. We benchmarked nine VLMs, ranging from state-of-the-art proprietary models to open-source models, on three psychometrically validated affective image datasets: the International Affective Picture System, the Nencki Affective Picture System, and the Library of AI-Generated Affective Images. The models performed two tasks in the zero-shot setting: (i) top-emotion classification (selecting the strongest discrete emotion elicited by an image) and (ii) continuous prediction of human ratings on 1–7 Likert scales for discrete emotion categories and affective dimensions. We also evaluated the impact of rater-conditioned prompting on the LAI-GAI dataset using de-identified participant metadata.
 
The results show good performance in discrete emotion classification, with accuracies typically ranging from 60\% to 80\% on six-emotion labels and from 60\% to 75\% on a more challenging 12-category task. The predictions of anger and surprise had the lowest accuracy in all datasets. For continuous rating prediction, models showed moderate to strong alignment with humans (r > 0.60) but also exhibited consistent biases, notably weaker performance on arousal, and a tendency to overestimate response strength. Rater-conditioned prompting resulted in only small and inconsistent changes in the predictions. Overall, VLMs capture broad affective trends but lack the nuance found in validated psychological ratings, highlighting their potential and current limitations for affective computing and mental health–related applications.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Supported Models](#-supported-models)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Task 1: Emotion Classification](#task-1-emotion-classification)
  - [Task 2: Affective Dimension Prediction](#task-2-affective-dimension-prediction)
  - [Task 3: Rater-conditioned Prompting](#task-3-rater-conditioned-prompting)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
---

## 🔭 Overview

This repository provides the codebase for evaluating **Vision-Language Models (VLMs)** on the task of **Visual Affect Analysis (VAA)** — predicting the emotional responses that images evoke in human viewers. Unlike traditional image emotion recognition that focuses on depicted emotions (e.g., a smiling face), VAA aims to understand the *affective impact* of images on observers.

### Research Questions
1. **Classification Accuracy:** How accurately do VLMs determine the top-rated discrete emotions assigned by human raters?
2. **Continuous Prediction:** How well do VLMs approximate human Likert ratings?
3. **Rater Conditioning:** Does conditioning on rater background (age, sex, country, emotional state) improve alignment with human ratings?

---

## ✨ Key Features

- 🤖 **Multi-Model Support**: Evaluate 11 VLMs including GPT-4.1, Gemini 2.5 Flash, Qwen2.5-VL, LLaVA, and more.
- 📊 **Three Evaluation Tasks**: Classification, Dimension Prediction, and Rater-conditioned Prompting.
- 🗂️ **Three Benchmark Datasets**: IAPS, NAPS, and LAI-GAI with human ground truth.
- ⚙️ **Hydra Configuration**: Flexible, composable configuration system.
- 🔧 **vLLM Optimization**: Efficient local inference for open-source models.
- 📉 **Comprehensive Metrics**: Accuracy, F1-score, MAE, Pearson correlation, Cohen's kappa.

---

## 💾 Datasets

We utilize three psychometrically validated datasets.

| Dataset | # Images | Emotions | Scale | Source |
|:---|:---:|:---|:---:|:---|
| **IAPS** | 692 | 6 basic + 2 dimensional | 1-9 | [CSEA](https://csea.phhp.ufl.edu/media/iapsmessage.html) |
| **NAPS** | 504 | 6 basic + 2 dimensional | 1-7 | [LOBI)](https://lobi.nencki.gov.pl/research/8/) |
| **LAI-GAI** | 480 | 12 discrete + 6 dimensional | 1-7 | [Affect Databases](https://affectdatabases.amu.edu.pl/) |

### ⚠️ Dataset Access & Rights

> **Important:** IAPS and NAPS are **restricted-access datasets**. We do not have redistribution rights for these images.

1.  **IAPS**: Request access from the [Center for the Study of Emotion and Attention (CSEA)](https://csea.phhp.ufl.edu/media/iapsmessage.html).
2.  **NAPS**: Request access from the [Laboratory of Brain Imaging, Nencki Institute](https://lobi.nencki.gov.pl/research/8/).
3.  **LAI-GAI**: Publicly available. You can download it directly:
    - [`affectdatabeses.amu.edu.pl`](https://affectdatabases.amu.edu.pl/)

---

## 🧠 Supported Models

We support both API-based proprietary models and open-source models via vLLM.

### Proprietary Models (API)
| Model | Config File | Description |
|---|---|---|
| **GPT-4.1** | `configs/model/gpt_4.1.yaml` | OpenAI's flagship model |
| **Gemini 2.5 Flash** | `configs/model/gemini_2.5_flash.yaml` | Google DeepMind's efficient multimodal model |

### Open-Source Models (Local via vLLM)
| Model | Params | Config File | Hardware Req |
|---|---|---|---|
| **Qwen2.5-VL** | 72B / 7B | `qwen_72b.yaml` / `qwen_7b.yaml` | H100 / RTX 4090 |
| **InternVL-3.5** | 38B / 8B | `intern_38.yaml` / `intern_8b.yaml` | H100 / RTX 4090 |
| **Gemma-3** | 27B | `gemma_27b.yaml` | H100 |
| **Kimi-VL-A3B** | 16B | `kimi_16b.yaml` | H100 |
| **GLM-4.1** | 9B | `glm_9b.yaml` | RTX 4090 |
| **LLaVA-v1.6** | 7B | `llava_mistral_7b.yaml` | RTX 4090 |

---

## ⚙️ Installation

### Requirements
- Python 3.10+
- CUDA-compatible GPU (for local inference)
- ~80GB VRAM for 72B models, ~16GB-24GB for 7B-9B models.

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/filnow/VAA-VLM.git
   cd VAA-VLM
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables (for API models):**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   ```

---

## 🚀 Usage

All experiments use [Hydra](https://hydra.cc/) for configuration management. You can override parameters via the command line or modify the YAML files in `configs/`.

### Task 1: Emotion Classification
Predict the primary emotion category evoked by an image (Zero-shot).

```bash
# Run with default config (Qwen-7B on NAPS)
python run_cls.py

# Specify model and dataset (e.g., GPT-4 on IAPS)
python run_cls.py model=gpt_4.1 dataset=iaps

# Run on LAIGAI with Gemini
python run_cls.py model=gemini_2.5_flash dataset=laigai
```

### Task 2: Affective Dimension Prediction
Predict continuous ratings (1-7/9 scale) for affective dimensions. This uses `n=50` samples with `temperature=0.5` to simulate human inter-rater variability.

```bash
# Standard regression
python run_reg.py model=qwen_7b dataset=naps

# Batch processing (more efficient for closed source models)
python run_reg_batch.py model=gpt_4.1 dataset=laigai
```

### Task 3: Rater-conditioned Prompting
Test whether participant metadata (demographics/state) improves predictions. **(LAI-GAI only)**.

```bash
# Demographics context (age, gender, country)
python run_context.py model=gemma_27b dataset=laigai \
    prompt_config_path=prompts/prompt-demographics-context.json

# Full context (demographics + pre-viewing emotional state)
python run_context_batch.py model=gemini_2.5_flash dataset=laigai \
    prompt_config_path=prompts/prompt-full-context.json
```

---

## 📊 Key Findings
*   **Accuracy:** Models achieve 60-80% accuracy on discrete classification.
*   **Difficulties:** Anger and surprise are consistently the hardest emotions to predict.
*   **Bias:** Models show moderate-to-strong correlation ($r > 0.60$) with human intensity ratings but tend to **overestimate** the strength of the emotional response.
*   **Context:** Adding rater demographics resulted in only small and inconsistent changes in predictions.

---

## 📂 Project Structure

```text
VAA-VLM/
├── configs/                   # Hydra configuration files
│   ├── dataset/               # Dataset configs (iaps, naps, laigai)
│   ├── model/                 # Model configs (gpt, qwen, internvl, etc.)
│   └── config-cls.yaml        # Base experiment configs
├── data/                      # Data resources
│   ├── emotion_definitions.json   
│   ├── human_cls/             # Human ground truth (Classification)
│   └── human_reg/             # Human ground truth (Regression)
├── prompts/                   # Prompt templates (JSON)
├── results/                   # Experiment outputs and logs
├── utils/                     
│   ├── metrics/               # Scripts for Acc, F1, MAE, Pearson
│   └── plots/                 # Visualization scripts
├── run_cls.py                 # Entry point: Classification
├── run_reg.py                 # Entry point: Regression
├── run_context.py             # Entry point: Context-aware
└── requirements.txt           # Dependencies
```

---

## 👥 Authors & Acknowledgments

**Authors:**  
Filip Nowicki¹, Hubert Marciniak¹, Jakub Łączkowski¹, Krzysztof Jassem¹, Tomasz Górecki¹, Vimala Balakrishnan²˒³, Desmond C. Ong⁴, Maciej Behnke⁵

*¹ Adam Mickiewicz University, Poznań, Poland*  
*² Universiti Malaya, Malaysia*  
*³ Korea University, Korea*  
*⁴ University of Texas at Austin, USA*  
*⁵ Cognitive Neuroscience Center, Adam Mickiewicz University, Poland*

**Funding:**  
This research was funded by the National Science Center in Poland (UMO-2020/39/B/HS6/00685) and the Excellence Initiative - Research University (ID-UB) at Adam Mickiewicz University.

---