# AmbedkarGPT-Intern-Task
AmbedkarGPT is a Retrieval-Augmented Generation (RAG) pipeline built to answer questions over Dr. B.R. Ambedkar’s writings, featuring multi-document retrieval, rigorous evaluation metrics, and comparative chunking analysis.

The project combines dense document retrieval, local LLM inference, and rigorous evaluation metrics to analyze both retrieval effectiveness and answer quality.

## Purpose of the Project
Large Language Models (LLMs) are powerful but prone to hallucination when answering factual questions.
Retrieval-Augmented Generation (RAG) mitigates this by grounding model outputs in retrieved source documents.

This project aims to:

- Build a multi-document RAG pipeline
- Evaluate retrieval quality and answer quality quantitatively
- Analyze how chunk size affects performance
- Identify failure modes in retrieval and generation
- Recommend an optimal configuration based on empirical results

## 📁 Repository Structure

```
AmbedkarGPT-Intern-Task/
│
├── app.py                     # Main interactive RAG application
├── evaluation.py              # Unified evaluation script (all metrics)
│
├── corpus/                    # Source documents (6 Ambedkar texts)
│   ├── speech1.txt
│   ├── speech2.txt
│   ├── speech3.txt
│   ├── speech4.txt
│   ├── speech5.txt
│   └── speech6.txt
│
├── test_dataset.json          # Provided evaluation dataset (25 questions)
├── test_results.json          # Output of evaluation runs
│
├── results_analysis.md        # Detailed analysis & recommendations
│
├── requirements.txt           # Runtime + evaluation dependencies
├── README.md                  # Project documentation
└── .gitignore
```

## ⚙️ Setup Instructions
### 1️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate

```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Install & Run Ollama
Make sure Ollama is installed and running:
```bash
ollama pull mistral
ollama serve
```

## 🚀 Running the Application
Start the interactive question-answering system:
```bash
python app.py
```
The system will ask for a question. 3 types of command can be done here. For closing the Q/A session, a user has to prompt 'exit'. If a user wants to rebuild the database with new information, then it has to type 'rebuild'. Besides these 2 commands, every other inputs will be treated as a question and the system will retrieve relevant chunks and generates a context-grounded answer.

## 📊 Evaluation
### Run Full Evaluation
```bash
python evaluation.py
```
This runs:
### 🔹 Retrieval Metrics
- Hit@K

- Mean Reciprocal Rank (MRR)

- Precision@K

### 🔹 Answer Quality Metrics

- Answer Relevance (embedding similarity)

- Faithfulness (context consistency)

- ROUGE-L (lexical overlap)

### 🔹 Semantic Metrics
- Semantic Cosine Similarity

- BLEU Score

### 🔹 Comparative Chunking Analysis
- Small chunks (200–300 chars)

- Medium chunks (500–600 chars)

- Large chunks (800–1000 chars)
