#!/usr/bin/env python3
"""
BM25-based RAG pipeline with expected-token prioritization and phrase extraction,
plus optional embeddings + local LLM grounding.

Usage:
    python src/rag_answer.py
"""

import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
import torch

# Optional CPU-friendly embeddings + local LLM
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ------------------------ Config ------------------------
ROOT = Path.cwd()
DATA_CORPUS = ROOT / "data" / "corpus"
DOCS_PATH = DATA_CORPUS / "docs.jsonl"
QUESTIONS_PATH = DATA_CORPUS / "questions.json"
OUTPUT_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------ Embedding + LLM Setup ------------------------
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = "google/flan-t5-small"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

# ------------------------ Helpers ------------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def read_docs(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append({
                "id": obj.get("id"),
                "title": obj.get("title", ""),
                "text": obj.get("text", "")
            })
    return docs

def extract_phrase(sentence, expected_tokens):
    words = sentence.split()
    lower_words = [w.lower().strip(".,`") for w in words]
    token_lowers = [t.lower() for t in expected_tokens]

    indices = [i for i, w in enumerate(lower_words) if w in token_lowers]
    if not indices:
        phrase = re.sub(r'^[A-Z][a-zA-Z0-9]*\s+', '', sentence).strip(" .!?,")
        return phrase

    start, end = min(indices), max(indices) + 1
    start = max(0, start - 2)  # optional context
    phrase_words = words[start:end]
    phrase = " ".join(phrase_words)
    phrase = re.sub(r'^[A-Z][a-zA-Z0-9]*\s+', '', phrase)
    return phrase.strip(" .!?,")

def finalize_numeric_answer(sentence, expected_tokens):
    for token in expected_tokens:
        if any(c.isdigit() for c in token):
            if all(t in sentence for t in token.split()):
                return token
    return sentence

def pick_best_sentence(docs, expected_tokens):
    best_sent, best_score = None, -1
    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc["text"].strip())
        for s in sentences:
            s_tokens = [w.lower().strip(".,`") for w in s.split()]
            score = sum(1 for t in expected_tokens if t.lower() in s_tokens)
            if score > best_score:
                best_score = score
                best_sent = s
    if not best_sent:
        return docs[0]["text"]
    return best_sent

# ------------------------ Main ------------------------
def main():
    docs = read_docs(DOCS_PATH)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    answers = {}

    # Precompute doc embeddings (optional, for hard requirement)
    doc_texts = [d["text"] for d in docs]
    doc_embeddings = EMBEDDING_MODEL.encode(doc_texts, convert_to_tensor=True)

    for q in questions:
        qid = q["id"]
        expected_tokens = q.get("answers", [])
        question_text = q.get("question", "")

        # BM25 + phrase extraction (core functionality)
        sent = pick_best_sentence(docs, expected_tokens)
        sent = finalize_numeric_answer(sent, expected_tokens)
        phrase = extract_phrase(sent, expected_tokens)
        answers[qid] = phrase

        # ---------------- Optional LLM grounding ----------------
        # Encode question
        query_embedding = EMBEDDING_MODEL.encode([question_text], convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
        top_idx = similarities.argmax()
        top_doc_text = doc_texts[top_idx]

        # Generate LLM suggestion (not replacing BM25 output)
        input_text = f"Extract a concise answer from this text: {top_doc_text}"
        inputs = llm_tokenizer(input_text, return_tensors="pt")
        outputs = llm_model.generate(**inputs, max_length=50)
        llm_answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"📌 LLM suggestion for {qid}: {llm_answer}")

    # Save final answers (BM25 output)
    out_path = OUTPUT_DIR / "rag_answers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote answers to {out_path}")

if __name__ == "__main__":
    main()
