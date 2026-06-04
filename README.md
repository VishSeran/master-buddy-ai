# 🎓 StudyBuddy-AI — DPO Preference-Aligned Study Assistant

**StudyBuddy-AI** is an AI-powered educational assistant fine-tuned with **Direct Preference Optimization (DPO)** to produce clear, structured, and beginner-friendly explanations. Rather than acting like a technical manual, it's trained to respond like a patient tutor.

---

## 📌 Overview

Most language models can answer questions — but not always in a way that's actually useful for students. StudyBuddy-AI addresses this by applying **alignment training (DPO)** to teach the model *what makes a good educational explanation*, using human preference data to distinguish helpful answers from unhelpful ones.

The model learns to prefer responses that are:
- ✅ Simple and easy to understand
- ✅ Well-structured with step-by-step reasoning
- ✅ Backed by real-world examples
- ✅ Free from unnecessary technical jargon

---

## 🎯 Objective

> *"Good explanations should feel like a patient tutor, not a technical manual."*

The core goal is to fine-tune a small language model (LLaMA 3.2 1B / Mistral 7B) using a custom preference dataset, so it consistently produces high-quality educational responses — not just correct ones.

---

## 🧠 How It Works — DPO Training

For every training example, the model is given:

| Field | Description |
|---|---|
| `prompt` | A student's question |
| `chosen` | A clear, structured, student-friendly answer |
| `rejected` | A confusing, jargon-heavy, or poorly structured answer |

The **DPOTrainer** from Hugging Face TRL optimizes the model to prefer `chosen` responses over `rejected` ones, directly aligning the model's output with human learning preferences — without needing a separate reward model.

### Example Training Pair

**Prompt:** *Explain TCP vs UDP*

| Type | Response Style |
|---|---|
| ✅ Chosen | Simple comparison, clear structure, real-world analogies |
| ❌ Rejected | Dense protocol-level jargon, no formatting, hard to follow |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Base Model | `meta-llama/Llama-3.2-1B-Instruct` (or Mistral 7B) |
| Training Framework | Hugging Face `transformers` + `trl` |
| Fine-tuning Method | Direct Preference Optimization (DPO) |
| Efficient Training | QLoRA (4-bit quantization) + LoRA adapters via `peft` |
| Dataset Format | Custom JSONL preference dataset (1000 pairs) |
| Backend | Python + PyTorch |

---

## 📁 Project Structure

```
master-buddy/
├── dataset/
│   └── dpo_study_assistant_1000.jsonl   # Preference dataset (prompt/chosen/rejected)
├── master-buddy-flow.ipynb              # Main training notebook
├── model_output/                        # Saved model checkpoints
├── .env                                 # HuggingFace token (not committed)
└── README.md
```

---

## ⚙️ Training Configuration

**QLoRA (4-bit quantization):**
- Quantization type: `nf4`
- Compute dtype: `bfloat16`
- Double quantization: enabled

**LoRA Adapter:**
- Rank (`r`): 4
- Alpha: 16
- Dropout: 0.1
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`

**DPO Hyperparameters:**
- Beta: `0.1`
- Learning rate: `1e-4`
- Epochs: 3
- Train/eval split: 80/20
- Batch size: 2 (train & eval)
- Eval strategy: every 100 steps

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/studybuddy-ai.git
cd studybuddy-ai
```

### 2. Install dependencies

```bash
pip install transformers trl peft datasets torch python-dotenv
```

### 3. Set up your HuggingFace token

Create a `.env` file in the parent directory:

```
HF_TOKEN=your_huggingface_token_here
```

### 4. Run the notebook

Open `master-buddy-flow.ipynb` in Jupyter and run all cells. A CUDA-enabled GPU is strongly recommended.

---

## 📊 Expected Outcomes

After training, StudyBuddy-AI will:
- Generate clearer, more structured explanations than the base model
- Default to teaching-style responses over dense technical answers
- Behave like a **personal AI tutor** for students across any subject

---

## 🔮 Future Improvements

- [ ] Multi-language support for wider accessibility
- [ ] RAG (Retrieval-Augmented Generation) integration for textbook knowledge
- [ ] Expand dataset with real student feedback
- [ ] Deploy as a web app (React frontend + FastAPI backend)
- [ ] Evaluate using automated preference metrics (e.g., reward model scoring)

---

## 💡 Why This Project Matters

AI alignment isn't just for safety — it's also about **usefulness**. This project demonstrates how techniques like DPO, originally developed for aligning large models to human values, can be applied in smaller, targeted domains to make AI genuinely better at a specific task. In education, where clarity matters more than complexity, this kind of preference alignment has real impact.

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
