"""Final inference on test.csv with the chosen retriever. Run ONLY after the method has been fixed on validation data."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import config as C
from .data_prep import detect_id_format, format_ids, load_processed, parse_id_list
from .generator import ExtractiveGenerator, build_generator
from .retrievers import build_retriever


def run(args) -> None:
    if args.generator == "none":
        raise SystemExit(
            "Submission needs an answer generator. Choose --generator extractive "
            "or set a provider API key and use --generator google or --generator anthropic."
        )
    data = load_processed(Path(args.processed_dir))
    corpus, test, sample = data["corpus"], data["test"], data["sample"]
    text_of = dict(zip(corpus["chunk_id"], corpus["chunk_text"]))
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.id_format or detect_id_format(sample["cited_chunk_ids"], data["train"]["gold_chunk_ids"])
    print(f"cited_chunk_ids will be written with format: {fmt!r}  (override with --id-format)")

    retriever = build_retriever(args.retriever, corpus, args.semantic_backend, args.embed_model, res_dir / "cache")
    gen = build_generator(args.generator, args.llm_model)
    fallback = ExtractiveGenerator()

    partial = res_dir / "submission_partial.csv"
    done = pd.read_csv(partial, dtype=str, keep_default_na=False) if partial.exists() else pd.DataFrame(
        columns=["question_id", "answer", "cited_chunk_ids"])
    done_ids = set(done["question_id"])
    rows = done.to_dict("records")
    todo = test[~test["question_id"].isin(done_ids)]
    print(f"{len(done_ids)} already done, {len(todo)} to go")

    n_fallback = 0
    for i, (_, q) in enumerate(todo.iterrows(), 1):
        ids = [c for c, _ in retriever.search(q["question"], C.CONTEXT_K)]
        passages = [(c, text_of[c]) for c in ids]
        g = gen.generate(q["question"], passages)
        if g.error or not g.answer:
            g, n_fallback = fallback.generate(q["question"], passages), n_fallback + 1
        answer = g.answer if not g.abstained else C.ABSTAIN_TOKEN.replace("_", " ").capitalize()
        rows.append({"question_id": q["question_id"], "answer": answer, "cited_chunk_ids": format_ids(g.cited_ids, fmt)})
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"  {i}/{len(todo)}")
    sub = pd.DataFrame(rows)
    # order + completeness checks against the official sample file
    sub = sub.set_index("question_id").reindex(sample["question_id"]).reset_index()
    assert sub["answer"].notna().all(), "Some test question_ids have no answer"
    valid = set(corpus["chunk_id"])
    bad = sum(1 for v in sub["cited_chunk_ids"] if any(c not in valid for c in parse_id_list(v)))
    out = res_dir / "submission.csv"
    sub[["question_id", "answer", "cited_chunk_ids"]].to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(sub)}  LLM-failure fallbacks={n_fallback}  rows citing unknown ids={bad}")
