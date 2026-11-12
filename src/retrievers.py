"""BM25 baseline and semantic retrieval (Report sections 4.2 and 4.3). Both expose search(query, k)."""
from __future__ import annotations

import hashlib
import re
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config as C

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case, alphanumeric tokens, English stop-words removed. Shared by BM25 and the lexical metrics."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def _topk(scores: np.ndarray, ids: list[str], k: int) -> list[tuple[str, float]]:
    k = min(k, len(ids))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(ids[i], float(scores[i])) for i in idx]


class BM25Retriever:
    name = "bm25"

    def __init__(self, corpus: pd.DataFrame, k1: float = C.BM25_K1, b: float = C.BM25_B):
        from rank_bm25 import BM25Okapi
        self.ids = corpus["chunk_id"].tolist()
        t0 = time.perf_counter()
        self.index = BM25Okapi([tokenize(t) or ["<empty>"] for t in corpus["chunk_text"]], k1=k1, b=b)
        self.info = {
            "retriever": self.name, "library": f"rank-bm25 {_ver('rank-bm25')}", "variant": "BM25Okapi",
            "k1": k1, "b": b, "tokenisation": "lowercase alphanumeric regex + sklearn English stop-words",
            "index_build_seconds": round(time.perf_counter() - t0, 3), "n_passages": len(self.ids),
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = np.asarray(self.index.get_scores(q), dtype=float) if q else np.zeros(len(self.ids))
        return _topk(scores, self.ids, k)


class SemanticRetriever:
    """Dense retrieval: normalised sentence embeddings + cosine similarity (exact inner-product search).

    backend="sbert"  -> sentence-transformers model (the real semantic retriever, use this for the report)
    backend="tfidf"  -> TF-IDF cosine. NOT dense/semantic; only an offline fallback for smoke-testing the pipeline.
    """
    name = "semantic"

    def __init__(self, corpus: pd.DataFrame, backend: str = "sbert", model_name: str = C.EMBED_MODEL,
                 cache_dir: Path | None = None, batch_size: int = 64):
        self.ids = corpus["chunk_id"].tolist()
        self.backend = backend
        texts = corpus["chunk_text"].tolist()
        t0 = time.perf_counter()
        if backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            key = hashlib.sha1((model_name + "".join(self.ids) + "".join(texts)).encode()).hexdigest()[:12]
            cache = (cache_dir / f"emb_{key}.npy") if cache_dir else None
            if cache and cache.exists():
                self.emb = np.load(cache)
            else:
                self.emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                             show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, self.emb)
            dim = int(self.emb.shape[1])
            self.info = {
                "retriever": self.name, "backend": "sentence-transformers", "model": model_name,
                "library": f"sentence-transformers {_ver('sentence-transformers')}", "vector_dim": dim,
                "normalisation": "L2 (cosine == dot product)", "index": "exact brute-force matrix product",
                "max_seq_length": int(getattr(self.model, "max_seq_length", 0)),
            }
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
            self.emb = self.vec.fit_transform(texts)  # sparse, L2-normalised
            self.info = {"retriever": self.name, "backend": "tfidf (OFFLINE FALLBACK - not dense)",
                         "library": f"scikit-learn {_ver('scikit-learn')}", "vector_dim": int(self.emb.shape[1]),
                         "normalisation": "L2", "index": "sparse matrix product"}
        else:
            raise ValueError("backend must be 'sbert' or 'tfidf'")
        self.info.update(index_build_seconds=round(time.perf_counter() - t0, 3), n_passages=len(self.ids))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.backend == "sbert":
            q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            scores = self.emb @ q
        else:
            scores = np.asarray((self.emb @ self.vec.transform([query]).T).todense()).ravel()
        return _topk(np.asarray(scores, dtype=float), self.ids, k)


def build_retriever(name: str, corpus: pd.DataFrame, backend: str, model: str, cache_dir: Path):
    if name == "bm25":
        return BM25Retriever(corpus)
    if name == "semantic":
        return SemanticRetriever(corpus, backend=backend, model_name=model, cache_dir=cache_dir)
    raise ValueError(name)
