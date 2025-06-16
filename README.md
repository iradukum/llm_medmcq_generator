# 🧠 LLM-MedMCQ Generator

---

## 🚀 Project Overview
This project builds a pipeline to transform medical text into MCQs using large language models. It includes:
- Clean data extraction from the MedQuAD dataset
- Summary generation using GPT-4
- Prompted MCQ creation and quality scoring
- Fine-tuning-ready datasets (still in progress)
- A full evaluation rubric (still in progress)
- Streamlit-based demo UI (still in progress)

---

## 📁 Directory Structure
```
llm_medmcq_generator/
├── data/               # Raw, processed, and formatted MCQ data
├── notebooks/          # Step-by-step development notebooks
├── src/                # Core logic for parsing, MCQ generation, evaluation
└── README.md           # This file
```

---

## 🧭 Weekly Goals
| Week | Focus                            | Outcome                                      |
|------|----------------------------------|----------------------------------------------|
| 1    | Data & Baseline Pipeline         | Clean QA data, baseline prompts, evaluation  |
| 2    | Model Training + Tuning          | Format MCQs, fine-tune/test small LLMs       |
| 3    | Deployment + Docs + Interview Kit| UI, deliverables, interview prep             |

---

## 📚 Resources
- [MedQuAD Dataset](https://github.com/abachaa/MedQuAD)
- [Ollama Quickstart](https://github.com/ollama/ollama)