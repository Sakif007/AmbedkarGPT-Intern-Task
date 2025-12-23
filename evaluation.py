import json
import numpy as np
from pathlib import Path
from datetime import datetime

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

from app import (
    build_vectorstore,
    build_retriever,
    answer_with_ollama
)
from langchain_community.embeddings import HuggingFaceEmbeddings


# ================= CONFIG =================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test_dataset.json"

TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKING_CONFIGS = {
    "small":  {"size": 250, "overlap": 50},
    "medium": {"size": 550, "overlap": 100},
    "large":  {"size": 900, "overlap": 200},
}

# Main evaluation Loop

def evaluate():
    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    
    test_data = raw["test_questions"]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    final_results = {}

    for name, cfg in CHUNKING_CONFIGS.items():
        print(f"\n🔹 Evaluating {name.upper()} chunks")

        chroma = build_vectorstore(
            recreate=True,
            chunk_size=cfg["size"],
            chunk_overlap=cfg["overlap"],
            persist_suffix=name
        )
        retriever = build_retriever(chroma)

        hits, rr, precision = [], [], []
        relevance, faithfulness, rouge_l, cosine_sim, bleu = [], [], [], [], []

        for item in test_data:
            question = item["question"]
            gt_answer = item["ground_truth"]
            gt_sources = set(item["source_documents"])

            docs = retriever.invoke(question)
            retrieved_sources = {d.metadata["source"] for d in docs}

            hits.append(int(len(gt_sources & retrieved_sources) > 0))

            ranks = [i + 1 for i, d in enumerate(docs) if d.metadata["source"] in gt_sources]
            rr.append(1 / ranks[0] if ranks else 0)

            precision.append(len(gt_sources & retrieved_sources) / TOP_K)

            context = "\n".join(d.page_content for d in docs)
            answer = answer_with_ollama(context, question)

            # -------- Answer Quality --------
            rel = cosine_similarity(
                embeddings.embed_documents([answer]),
                embeddings.embed_documents([gt_answer])
            )[0][0]

            faithful = cosine_similarity(
                embeddings.embed_documents([answer]),
                embeddings.embed_documents([context])
            )[0][0]

            relevance.append(rel)
            faithfulness.append(faithful)
            rouge_l.append(rouge.score(gt_answer, answer)["rougeL"].fmeasure)
            cosine_sim.append(rel)

            bleu.append(sentence_bleu(
                [gt_answer.split()],
                answer.split(),
                smoothing_function=smoothie
            ))

        final_results[name] = {
            "Hit@K": np.mean(hits),
            "MRR": np.mean(rr),
            "Precision@K": np.mean(precision),
            "Answer Relevance": np.mean(relevance),
            "Faithfulness": np.mean(faithfulness),
            "ROUGE-L": np.mean(rouge_l),
            "Semantic Cosine": np.mean(cosine_sim),
            "BLEU": np.mean(bleu),
        }

    print("\n================ FINAL COMPARISON ================")
    for name, metrics in final_results.items():
        print(f"\n🔸 {name.upper()} CHUNKS")
        for k, v in metrics.items():
            print(f"{k:18s}: {v:.4f}")
    

        # ================= SAVE RESULTS =================

    output = {
        "metadata": {
            "model": "mistral",
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "num_questions": len(test_data),
            "chunking_strategies": {
                name: {
                    "chunk_size": CHUNKING_CONFIGS[name]["size"],
                    "chunk_overlap": CHUNKING_CONFIGS[name]["overlap"]
                }
                for name in CHUNKING_CONFIGS
            },
            "generated_at": datetime.now().isoformat()
        },
        "results": final_results
    }

    output_path = BASE_DIR / "test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n test_results.json saved at: {output_path}")



if __name__ == "__main__":
    evaluate()
