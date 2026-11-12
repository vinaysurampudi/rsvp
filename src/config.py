"""Central configuration. Every value that should be recorded for reproducibility lives here."""
import os
from pathlib import Path

COMPETITION = "agent-eval-part-i-grounded-rag-benchmark"

SEED = 42
VAL_FRACTION = 0.20
KS = (1, 3, 5, 10)            # K values for Precision@K / Recall@K (identical for both retrievers)
CONTEXT_K = 5                 # passages passed to the language model
MAX_PASSAGE_CHARS = 2500      # per-passage truncation inside the prompt

# BM25 parameters (Robertson & Zaragoza, 2009 defaults)
BM25_K1 = 1.5
BM25_B = 0.75

# Semantic retrieval
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model settings can be overridden with ANTHROPIC_MODEL and GEMINI_MODEL.
LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_MAX_TOKENS = 600
LLM_TEMPERATURE = 0.0

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

# The report says file/field names must be verified against the real download, so the loader accepts
# common alternative column names and renames them to the canonical ones below.
COLUMN_ALIASES = {
    "corpus": {
        "chunk_id": ["chunk_id", "passage_id", "id"],
        "chunk_text": ["chunk_text", "text", "passage", "content", "chunk"],
        "doc_id": ["doc_id", "document_id"],
        "title": ["title", "doc_title"],
        "doc_type": ["doc_type", "type"],
    },
    "train": {
        "question_id": ["question_id", "qid", "id"],
        "question": ["question", "query"],
        "answer": ["answer", "reference_answer", "gold_answer"],
        "gold_chunk_ids": ["gold_chunk_ids", "gold_chunks", "gold_passage_ids", "evidence_ids", "cited_chunk_ids"],
    },
    "test": {
        "question_id": ["question_id", "qid", "id"],
        "question": ["question", "query"],
    },
    "sample": {
        "question_id": ["question_id", "qid", "id"],
        "answer": ["answer"],
        "cited_chunk_ids": ["cited_chunk_ids", "chunk_ids", "gold_chunk_ids"],
    },
}
REQUIRED = {
    "corpus": ["chunk_id", "chunk_text"],
    "train": ["question_id", "question", "answer", "gold_chunk_ids"],
    "test": ["question_id", "question"],
    "sample": ["question_id", "answer", "cited_chunk_ids"],
}
FILE_NAMES = {"corpus": "corpus.csv", "train": "train.csv", "test": "test.csv", "sample": "sample_submission.csv"}
