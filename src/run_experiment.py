"""Controlled comparison of BM25 vs semantic retrieval on the SAME validation questions (Report 3.2, 4.6)."""
from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .data_prep import load_processed
from .generator import build_generator
from .retrievers import build_retriever


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def system_info() -> dict:
    def v(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": {p: v(p) for p in ["pandas", "numpy", "scikit-learn", "rank-bm25", "sentence-transformers",
                                           "anthropic", "torch"]}}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def run(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_processed(Path(args.processed_dir))
    corpus, val = data["corpus"], data["val"]
    text_of = dict(zip(corpus["chunk_id"], corpus["chunk_text"]))
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    ks = C.KS
    max_k = max(max(ks), C.CONTEXT_K)

    generator = build_generator(args.generator, args.llm_model) if args.generator != "none" else None
    cfg = {"seed": args.seed, "ks": ks, "context_k": C.CONTEXT_K, "n_validation_questions": len(val),
           "n_generation_questions": min(args.n_gen, len(val)) if generator else 0,
           "generator": generator.info if generator else None,
           "judge": bool(args.judge and generator and args.generator in ("anthropic", "google")),
           "retrievers": {}, "system": system_info()}

    retrievers = {}
    for name in args.retrievers:
        print(f"Building {name} index ...")
        retrievers[name] = build_retriever(name, corpus, args.semantic_backend, args.embed_model, res_dir / "cache")
        cfg["retrievers"][name] = retrievers[name].info

    # ---- 1. retrieval evaluation on ALL validation questions
    rows, ranked = [], {}
    for name, r in retrievers.items():
        print(f"Retrieval evaluation: {name} on {len(val)} questions")
        for _, q in val.iterrows():
            gold = set(q["gold_list"])
            t0 = time.perf_counter()
            hits = r.search(q["question"], max_k)
            lat = time.perf_counter() - t0
            ids = [c for c, _ in hits]
            ranked[(name, q["question_id"])] = ids
            m = M.retrieval_metrics(ids[: max(ks)], gold, ks)
            rows.append({"retriever": name, "question_id": q["question_id"], "retrieval_seconds": lat, **m})
    ret = pd.DataFrame(rows)
    ret.to_csv(res_dir / "retrieval_per_question.csv", index=False)
    metric_cols = [c for c in ret.columns if c not in ("retriever", "question_id") and not c.startswith("MRR@")]
    ret_sum = ret.groupby("retriever")[metric_cols].mean().reset_index()
    ret_sum.insert(1, "n_questions", ret.groupby("retriever").size().values)
    ret_sum.to_csv(res_dir / "retrieval_summary.csv", index=False)
    print("\n" + md_table(ret_sum) + "\n")

    # ---- 2. generation evaluation (same sampled questions for every retriever)
    gen_sum = pd.DataFrame()
    probe_sum = pd.DataFrame()
    if generator:
        rng = random.Random(args.seed)
        sample_idx = sorted(rng.sample(range(len(val)), min(args.n_gen, len(val))))
        sample = val.iloc[sample_idx]
        probe_ids = set(sample["question_id"].iloc[: args.abstention_probe])
        use_judge = cfg["judge"]
        grows, prows = [], []
        for name, r in retrievers.items():
            print(f"Generation evaluation: {name} on {len(sample)} questions")
            for i, (_, q) in enumerate(sample.iterrows(), 1):
                gold = set(q["gold_list"])
                ids = ranked[(name, q["question_id"])][: C.CONTEXT_K]
                passages = [(c, text_of[c]) for c in ids]
                g = generator.generate(q["question"], passages)
                ev_texts = [t for _, t in passages]
                row = {"retriever": name, "question_id": q["question_id"], "question": q["question"],
                       "reference_answer": q["answer"], "generated_answer": g.answer, "cited_ids": json.dumps(g.cited_ids),
                       "abstained": int(g.abstained), "invalid_citations": g.invalid_citations,
                       "generation_seconds": g.latency, "error": g.error,
                       "retrieval_seconds": float(ret[(ret.retriever == name) & (ret.question_id == q["question_id"])]["retrieval_seconds"].iloc[0])}
                row["total_seconds"] = row["retrieval_seconds"] + g.latency
                row.update(M.citation_metrics(g.cited_ids, gold, g.abstained))
                row.update(M.answer_overlap("" if g.abstained else g.answer, q["answer"]))
                if g.abstained or g.error:
                    row.update(faithfulness=math.nan, unsupported_claims=0, n_claims=0)
                else:
                    row.update(M.lexical_faithfulness(g.answer, ev_texts))
                if use_judge and not g.abstained and not g.error:
                    row.update(M.llm_judge(generator, q["question"], q["answer"], g.answer, ev_texts))
                grows.append(row)

                # RQ4 probe: withhold ALL gold passages from the context and check that the system abstains
                if q["question_id"] in probe_ids:
                    deep = r.search(q["question"], C.CONTEXT_K + len(gold))
                    kept = [c for c, _ in deep if c not in gold][: C.CONTEXT_K]
                    pg = generator.generate(q["question"], [(c, text_of[c]) for c in kept])
                    prows.append({"retriever": name, "question_id": q["question_id"], "abstained_without_evidence": int(pg.abstained),
                                  "answer": pg.answer, "cited_ids": json.dumps(pg.cited_ids), "error": pg.error})
                if i % 10 == 0:
                    print(f"  {i}/{len(sample)}")
        gen = pd.DataFrame(grows)
        gen.to_csv(res_dir / "generation_per_question.csv", index=False)
        num = [c for c in gen.columns if gen[c].dtype.kind in "fi" and c not in ("n_citations",)]
        gen_sum = gen.groupby("retriever")[num].mean().reset_index()
        gen_sum.insert(1, "n_questions", gen.groupby("retriever").size().values)
        gen_sum.insert(2, "n_errors", gen.groupby("retriever")["error"].apply(lambda s: int((s != "").sum())).values)
        gen_sum["unsupported_claims_total"] = gen.groupby("retriever")["unsupported_claims"].sum().values
        gen_sum.to_csv(res_dir / "generation_summary.csv", index=False)
        print("\n" + md_table(gen_sum) + "\n")
        if prows:
            probe = pd.DataFrame(prows)
            probe.to_csv(res_dir / "abstention_probe_per_question.csv", index=False)
            probe_sum = probe.groupby("retriever")["abstained_without_evidence"].agg(["mean", "size"]).reset_index()
            probe_sum.columns = ["retriever", "abstention_rate_when_gold_withheld", "n_probe_questions"]
            probe_sum.to_csv(res_dir / "abstention_probe_summary.csv", index=False)

    (res_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    with open(res_dir / "comparison_tables.md", "w") as f:
        f.write("# Results (generated by run_experiment.py)\n\n## Retrieval (all validation questions)\n\n" + md_table(ret_sum) + "\n")
        if len(gen_sum):
            f.write("\n## Generation (sampled validation questions)\n\n" + md_table(gen_sum) + "\n")
        if len(probe_sum):
            f.write("\n## RQ4 - abstention when gold evidence is withheld\n\n" + md_table(probe_sum) + "\n")
        f.write("\nMetric definitions: see README.md. Lexical faithfulness/correctness are proxies; judge_* columns "
                "(if present) come from an LLM judge and should be spot-checked by hand.\n")
    print(f"All outputs written to {res_dir}/")
