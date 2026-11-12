"""Local Streamlit interface for the AgentEval RAG workflow."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from src import config as C
from src import data_prep, make_submission, run_experiment

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
REQUIRED_UPLOADS = ("corpus.csv", "train.csv", "test.csv", "sample_submission.csv")


def show_action(title: str, action) -> None:
    """Run a workflow step and show its output or a readable error."""
    output = io.StringIO()
    try:
        with st.spinner(title), contextlib.redirect_stdout(output):
            action()
    except (Exception, SystemExit) as error:
        if output.getvalue().strip():
            st.code(output.getvalue(), language="text")
        st.error(str(error) or f"{title} did not complete.")
        return

    if output.getvalue().strip():
        st.code(output.getvalue(), language="text")
    st.success(f"{title} complete.")


def prepared_files_exist() -> bool:
    files = ("corpus_clean.csv", "train_split.csv", "val_split.csv",
             "test_clean.csv", "sample_submission.csv")
    return all((PROCESSED_DIR / filename).is_file() for filename in files)


def render_dataset_tab() -> None:
    st.subheader("Step 1 — Get the Kaggle dataset")
    st.write("The project needs the corpus, labelled questions, test questions, and sample submission CSVs.")
    if st.button("Download from Kaggle", help="Requires Kaggle API credentials and accepted competition rules."):
        show_action("Kaggle download", lambda: data_prep.download(RAW_DIR))

    st.markdown("**Or upload files you downloaded manually**")
    uploaded = st.file_uploader(
        "Select the four competition CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Keep the original filenames: corpus.csv, train.csv, test.csv, sample_submission.csv.",
    )
    if st.button("Save uploaded files", disabled=not uploaded):
        names = {file.name for file in uploaded}
        missing = sorted(set(REQUIRED_UPLOADS) - names)
        if missing:
            st.error(f"Missing required file(s): {', '.join(missing)}")
        else:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            for file in uploaded:
                if file.name in REQUIRED_UPLOADS:
                    (RAW_DIR / file.name).write_bytes(file.getvalue())
            st.success(f"Saved all four CSV files in {RAW_DIR}.")

    present = sorted(path.name for path in RAW_DIR.glob("*.csv"))
    st.caption(f"Raw data folder: {RAW_DIR}")
    st.write("CSV files found:", ", ".join(present) if present else "None yet")
    if st.button("Check dataset files"):
        show_action(
            "Dataset inspection",
            lambda: st.json(data_prep.inspect(data_prep.load_raw(RAW_DIR))),
        )


def render_prepare_tab() -> None:
    st.subheader("Step 2 — Check, clean, and split the data")
    st.write("This creates the cleaned corpus and a reproducible train/validation split.")
    val_fraction = st.slider("Validation split (%)", min_value=10, max_value=40, value=20) / 100
    seed = st.number_input("Random seed", min_value=0, value=C.SEED, step=1)

    if st.button("Prepare dataset"):
        show_action(
            "Dataset preparation",
            lambda: data_prep.prepare(RAW_DIR, PROCESSED_DIR, RESULTS_DIR, int(seed), val_fraction),
        )

    if prepared_files_exist():
        st.success(f"Prepared data is ready in {PROCESSED_DIR}.")
    else:
        st.info("Prepare the dataset before running evaluation or submission.")


def render_evaluate_tab() -> None:
    st.subheader("Step 3 — Compare retrieval methods")
    st.write("BM25 and semantic retrieval are evaluated on the validation questions.")
    semantic_backend = st.selectbox(
        "Semantic search backend",
        ("sbert", "tfidf"),
        help="SBERT is the real dense semantic method. TF-IDF is an offline smoke-test fallback.",
    )
    generator = st.selectbox(
        "Answer generation",
        ("none", "extractive", "google", "anthropic"),
        help="Choose Google Gemini or Anthropic for API-based generation; none evaluates retrieval only.",
    )
    n_gen = st.number_input("Questions for answer-generation evaluation", min_value=1, value=100, step=10)
    probe = st.number_input("Questions for evidence-withheld probe", min_value=0, value=20, step=5)
    judge = st.checkbox(
        "Add LLM judge (extra API calls)",
        disabled=generator not in ("google", "anthropic"),
    )

    if generator == "google" and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        st.warning("Enter your Google API key in the sidebar to use Gemini.")
    if generator == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY in PowerShell before starting this app to use Anthropic.")
    if semantic_backend == "tfidf":
        st.warning("TF-IDF is only for an offline smoke test; use SBERT for reported semantic results.")

    if st.button("Run evaluation", disabled=not prepared_files_exist()):
        args = SimpleNamespace(
            seed=C.SEED,
            processed_dir=str(PROCESSED_DIR),
            results_dir=str(RESULTS_DIR),
            semantic_backend=semantic_backend,
            embed_model=C.EMBED_MODEL,
            generator=generator,
            llm_model=None,
            retrievers=["bm25", "semantic"],
            n_gen=int(n_gen),
            judge=judge,
            abstention_probe=int(probe),
        )
        show_action("Evaluation", lambda: run_experiment.run(args))

    for filename, title in (
        ("retrieval_summary.csv", "Retrieval results"),
        ("generation_summary.csv", "Answer-generation results"),
        ("abstention_probe_summary.csv", "Evidence-withheld results"),
    ):
        path = RESULTS_DIR / filename
        if path.is_file():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path), use_container_width=True)


def render_submission_tab() -> None:
    st.subheader("Step 4 — Create the Kaggle submission")
    st.write("Choose the retriever using validation results, then generate answers for test questions.")
    retriever = st.selectbox("Retriever to use", ("semantic", "bm25"))
    semantic_backend = st.selectbox("Semantic backend", ("sbert", "tfidf"), key="submission_backend")
    generator = st.selectbox(
        "Answer generator",
        ("extractive", "google", "anthropic", "none"),
        key="submission_generator",
    )
    if generator == "none":
        st.warning("The submission requires generated answers. Choose extractive, Gemini, or Anthropic.")
    if generator == "google" and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        st.warning("Enter your Google API key in the sidebar to use Gemini.")
    if generator == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY in PowerShell before starting this app to use Anthropic.")

    if st.button("Create submission", disabled=not prepared_files_exist() or generator == "none"):
        args = SimpleNamespace(
            processed_dir=str(PROCESSED_DIR),
            results_dir=str(RESULTS_DIR),
            retriever=retriever,
            semantic_backend=semantic_backend,
            embed_model=C.EMBED_MODEL,
            generator=generator,
            llm_model=None,
            id_format=None,
        )
        show_action("Submission creation", lambda: make_submission.run(args))

    path = RESULTS_DIR / "submission.csv"
    if path.is_file():
        st.success(f"Submission file: {path}")
        st.dataframe(pd.read_csv(path).head(20), use_container_width=True)
        st.download_button("Download submission.csv", path.read_bytes(), file_name="submission.csv")


st.set_page_config(page_title="AgentEval RAG", page_icon="🔎", layout="wide")
google_api_key = st.sidebar.text_input(
    "Google Gemini API key",
    value=os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", "")),
    type="password",
    help="Used only for Gemini calls in this app session; not saved to project files.",
)
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("AgentEval — Evidence-Grounded QA")
st.caption("Compare BM25 and semantic retrieval, inspect evidence-grounded answers, and prepare a Kaggle submission.")
st.info("Start with **Dataset**, then **Prepare**, **Evaluate**, and finally **Submission**. "
        "You can run retrieval evaluation without an API key.")

dataset_tab, prepare_tab, evaluate_tab, submission_tab = st.tabs(
    ("1 · Dataset", "2 · Prepare", "3 · Evaluate", "4 · Submission")
)
with dataset_tab:
    render_dataset_tab()
with prepare_tab:
    render_prepare_tab()
with evaluate_tab:
    render_evaluate_tab()
with submission_tab:
    render_submission_tab()
