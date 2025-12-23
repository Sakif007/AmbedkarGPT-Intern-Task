## 📊 Results Analysis & Recommendations

### Overview
This document presents a detailed analysis of the evaluation results obtained for the AmbedkarGPT Retrieval-Augmented Generation (RAG) system. The evaluation focuses on three major aspects:

1. Retrieval Performance  
2. Answer Quality  
3. Semantic Similarity  

Additionally, a comparative chunking analysis is conducted to understand the impact of chunk size on system performance. Based on these findings, common failure modes are identified and an optimal configuration is recommended.

---

### 1️⃣ Retrieval Performance Analysis

#### Metrics Used

- **Hit@K**: Measures whether at least one relevant document appears in the top-K retrieved results.
- **MRR (Mean Reciprocal Rank)**: Measures how early the first relevant document appears.
- **Precision@K**: Measures how many retrieved documents are relevant among the top-K.

| Chunk Size | Hit@K | MRR | Precision@K |
|------|------|--------|-------|
| Small (200–300) | 0.80 | 0.71 | 0.32 |
| Medium (500–600) | 0.88 | 0.77 | 0.32 |
| Large (800–1000) | **0.88** | **0.81** | **0.35** |

#### Observations

- Large chunks achieve the **highest MRR and Precision@K**, indicating better ranking of relevant documents.
- Small chunks often fragment important semantic information across multiple chunks, reducing retrieval effectiveness.
- Medium chunks provide a reasonable balance but slightly underperform large chunks in ranking quality.

---

### 2️⃣ Answer Quality Analysis

#### Metrics Used

- **Answer Relevance**: Cosine similarity between the generated answer and the ground-truth answer embeddings.
- **Faithfulness**: Cosine similarity between the generated answer and the retrieved context embeddings.
- **ROUGE-L**: Measures longest common subsequence overlap between generated and reference answers.

| Chunk Size | Answer Relevance | Faithfulness | ROUGE-L |
|------|------------------|-------------|--------|
| Small | 0.530 | **0.629** | 0.280 |
| Medium | 0.560 | 0.587 | 0.260 |
| Large | **0.570** | 0.579 | **0.282** |

#### Observations

- Small chunks exhibit slightly higher faithfulness since answers remain closely tied to short retrieved passages.
- Large chunks generate more **relevant and well-structured answers**, reflected in higher relevance and ROUGE-L scores.
- Lower ROUGE-L values across all settings indicate frequent paraphrasing, which is expected for generative models.

---

### 3️⃣ Semantic Similarity Analysis

#### Metrics Used

- **Semantic Cosine Similarity**: Measures semantic alignment between generated and ground-truth answers using embeddings.
- **BLEU Score**: Measures n-gram overlap with reference answers.

| Chunk Size | Semantic Cosine | BLEU |
|------|------------------|------|
| Small | 0.530 | 0.058 |
| Medium | 0.560 | 0.058 |
| Large | **0.571** | **0.078** |

#### Observations

- Semantic cosine similarity improves consistently with larger chunk sizes.
- BLEU scores remain low across all settings, indicating that answers are semantically correct but lexically paraphrased.
- Embedding-based semantic metrics provide a more reliable assessment of answer correctness than lexical overlap metrics alone.

---

### 4️⃣ Comparative Chunking Analysis

| Chunk Size | Strengths | Weaknesses |
|------|----------|------------|
| Small | High faithfulness, precise context | Fragmented semantics, lower retrieval accuracy |
| Medium | Balanced performance | Slightly weaker ranking than large chunks |
| Large | Best retrieval and semantic performance | Possible inclusion of irrelevant surrounding text |

#### Overall Trend

- Increasing chunk size improves retrieval ranking, answer relevance, and semantic coherence.
- The trade-off is a small reduction in faithfulness due to larger contextual scope.

---

### 5️⃣ Common Failure Modes

- **Source Ambiguity**: Similar themes across documents lead to correct answers but incorrect source attribution.
- **Context Overload**: Large chunks may include irrelevant information.
- **Paraphrasing Effects**: Correct answers expressed differently reduce ROUGE and BLEU scores.
- **Ground Truth Granularity**: Some questions are answerable from multiple documents while ground truth lists only one.

---

### 6️⃣ Recommended Configuration

#### Optimal Setup

- **Chunk Size**: 800–1000 characters  
- **Chunk Overlap**: ~200 characters  
- **Top-K Retrieval**: 3  
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2  
- **LLM**: Mistral (via Ollama)

#### Rationale

- Maximizes retrieval accuracy and ranking quality.
- Produces the most semantically aligned and relevant answers.
- Balances contextual completeness with acceptable faithfulness.

---

### 7️⃣ Final Conclusion

This evaluation demonstrates that **chunking strategy plays a critical role** in RAG system performance. Larger chunks consistently improve retrieval effectiveness and semantic quality, while embedding-based metrics provide a more accurate measure of answer correctness than surface-level overlap metrics.

All results are reproducible using `evaluation.py`, with raw metrics stored in `test_results.json`.
