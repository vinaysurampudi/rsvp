"""Command-line entry point.  python main.py <command> --help"""
import argparse
from pathlib import Path
import subprocess
import sys

from src import config as C


def main():
    p = argparse.ArgumentParser(description="Evidence-grounded QA on AgentEval Part I (RAG)")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seed", type=int, default=C.SEED)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="download the Kaggle files into --raw-dir")
    sub.add_parser("inspect", help="print file/column/row/missing/duplicate report only")
    pp = sub.add_parser("prepare", help="inspect + clean + validate + split")
    pp.add_argument("--val-fraction", type=float, default=C.VAL_FRACTION)

    def add_common(sp):
        sp.add_argument("--semantic-backend", choices=["sbert", "tfidf"], default="sbert",
                        help="tfidf = offline smoke-test fallback (NOT dense); use sbert for real results")
        sp.add_argument("--embed-model", default=C.EMBED_MODEL)
        sp.add_argument("--generator", choices=["google", "anthropic", "extractive", "none"], default="google")
        sp.add_argument("--llm-model", default=None, help="override the selected provider's default model")

    ev = sub.add_parser("evaluate", help="BM25 vs semantic on the validation split")
    add_common(ev)
    ev.add_argument("--retrievers", nargs="+", default=["bm25", "semantic"], choices=["bm25", "semantic"])
    ev.add_argument("--n-gen", type=int, default=100, help="validation questions used for (costly) generation eval")
    ev.add_argument("--judge", action="store_true", help="add LLM-judge correctness/faithfulness (Google or Anthropic generator)")
    ev.add_argument("--abstention-probe", type=int, default=20, help="questions for the RQ4 withheld-evidence probe")

    sm = sub.add_parser("submit", help="inference on test.csv -> results/submission.csv")
    add_common(sm)
    sm.add_argument("--retriever", choices=["bm25", "semantic"], required=True)
    sm.add_argument("--id-format", default=None, help="override cited_chunk_ids format: json | ',' | ', ' | ';' | '|' | ' '")
    sub.add_parser("ui", help="open the local browser interface")

    a = p.parse_args()
    raw, proc, res = Path(a.raw_dir), Path(a.processed_dir), Path(a.results_dir)

    if a.cmd == "ui":
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
        except FileNotFoundError as e:
            raise SystemExit("Streamlit is missing. Install dependencies with: python -m pip install -r requirements.txt") from e
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"Streamlit exited with status {e.returncode}.") from e
        return

    from src import data_prep
    if a.cmd == "download":
        data_prep.download(raw)
    elif a.cmd == "inspect":
        import json
        print(json.dumps(data_prep.inspect(data_prep.load_raw(raw)), indent=2))
    elif a.cmd == "prepare":
        data_prep.prepare(raw, proc, res, a.seed, a.val_fraction)
    elif a.cmd == "evaluate":
        from src import run_experiment
        run_experiment.run(a)
    elif a.cmd == "submit":
        from src import make_submission
        make_submission.run(a)


if __name__ == "__main__":
    main()
