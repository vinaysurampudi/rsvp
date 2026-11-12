"""Evaluation measures (Report section 4.5). Definitions used for edge cases are stated in each docstring."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

from .retrievers import tokenize


# ------------------------------------------------------------------ retrieval
def retrieval_metrics(retrieved: list[str], gold: set[str], ks: tuple[int, ...]) -> dict:
    """Precision@K = |top-K ∩ gold| / K.  Recall@K = |top-K ∩ gold| / |gold|.
    MRR = 1 / rank of the first gold passage within the retrieved list (0 if none).
    Note: with |gold| < K, Precision@K cannot reach 1.0 - compare retrievers at equal K, not across K."""
    out = {}
    for k in ks:
        hits = len(set(retrieved[:k]) & gold)
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = hits / len(gold) if gold else math.nan
        out[f"Hit@{k}"] = float(hits > 0)
    rr = 0.0
    for rank, cid in enumerate(retrieved, 1):
        if cid in gold:
            rr = 1.0 / rank
            break
    out[f"MRR@{len(retrieved)}"] = rr
    out["MRR"] = rr
    return out


# ------------------------------------------------------------------ citations
def citation_metrics(cited: list[str], gold: set[str], abstained: bool) -> dict:
    """Citation precision = |cited ∩ gold| / |cited|.  Citation recall = |cited ∩ gold| / |gold|.
    Empty citation set (incl. abstention) -> both 0.0, so a missing citation is penalised, never ignored.
    Abstention is reported separately."""
    c = set(cited)
    inter = len(c & gold)
    return {
        "citation_precision": inter / len(c) if c else 0.0,
        "citation_recall": inter / len(gold) if gold and c else 0.0,
        "n_citations": len(c),
    }


# ------------------------------------------------------------------ answer correctness (lexical)
def normalize_answer(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_overlap(pred: str, ref: str) -> dict:
    """Token F1 (SQuAD-style) and reference-token recall. Lexical proxies; abstentions score 0.
    Reference recall is more forgiving than F1 when reference answers are long."""
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return {"token_f1": float(p == r), "ref_recall": float(p == r)}
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return {"token_f1": 0.0, "ref_recall": 0.0}
    prec, rec = common / len(p), common / len(r)
    return {"token_f1": 2 * prec * rec / (prec + rec), "ref_recall": rec}


# ------------------------------------------------------------------ faithfulness (lexical proxy)
def lexical_faithfulness(answer: str, evidence_texts: list[str], threshold: float = 0.6) -> dict:
    """Proxy only. An answer sentence counts as 'supported' if >= threshold of its content tokens occur in the
    supplied evidence. faithfulness = supported / total sentences; unsupported_claims = unsupported sentences.
    Treat as an estimate and check a sample by hand (Report 4.5); use the LLM judge for a stronger estimate."""
    ev = set()
    for t in evidence_texts:
        ev.update(tokenize(t))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if tokenize(s)]
    if not sents:
        return {"faithfulness": math.nan, "unsupported_claims": 0, "n_claims": 0}
    unsupported = 0
    for s in sents:
        toks = tokenize(s)
        if sum(t in ev for t in toks) / len(toks) < threshold:
            unsupported += 1
    return {"faithfulness": 1 - unsupported / len(sents), "unsupported_claims": unsupported, "n_claims": len(sents)}


# ------------------------------------------------------------------ LLM judge (optional, stronger estimate)
JUDGE_SYSTEM = """You are a strict evaluator of question-answering outputs. Return ONE JSON object only:
{"correctness": <0, 0.5 or 1>, "n_claims": <int>, "n_unsupported_claims": <int>}
- correctness: 1 = candidate answer conveys the same meaning as the reference answer; 0.5 = partially correct or incomplete; 0 = wrong or contradictory.
- n_claims: number of distinct factual claims in the candidate answer.
- n_unsupported_claims: number of those claims NOT supported by the evidence passages (judge ONLY against the passages, not world knowledge)."""


def llm_judge(generator, question: str, reference: str, candidate: str, evidence_texts: list[str]) -> dict:
    from .generator import _parse_json
    ev = "\n\n".join(f"[{i+1}] {t[:2000]}" for i, t in enumerate(evidence_texts))
    user = f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}\n\nEvidence passages:\n{ev}"
    try:
        d = _parse_json(generator._call(JUDGE_SYSTEM, user, max_tokens=200))
        n, u = int(d.get("n_claims", 0)), int(d.get("n_unsupported_claims", 0))
        return {"judge_correctness": float(d.get("correctness", math.nan)),
                "judge_faithfulness": (1 - u / n) if n else math.nan, "judge_unsupported_claims": u}
    except Exception:
        return {"judge_correctness": math.nan, "judge_faithfulness": math.nan, "judge_unsupported_claims": math.nan}
