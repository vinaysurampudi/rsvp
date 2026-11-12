"""Dataset download, inspection, validation, cleaning and train/validation split (Report section 3.3)."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


# ----------------------------------------------------------------------------- download
def download(raw_dir: Path) -> None:
    """Step 1: download the permitted files with the Kaggle CLI and record the download date."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", C.COMPETITION, "-p", str(raw_dir)]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "Kaggle CLI not found. Install project dependencies with: "
            "python -m pip install -r requirements.txt"
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "Kaggle download failed. Check that (1) Kaggle CLI dependencies are installed "
            "(`python -m pip install -r requirements.txt`), (2) kaggle.json exists in "
            "`C:\\Users\\<you>\\.kaggle\\` (Kaggle > Settings > API > Create New Token), "
            "and (3) you have clicked 'Join/Accept rules' on the competition page.\n"
            f"Alternative: download manually from https://www.kaggle.com/competitions/{C.COMPETITION}/data "
            f"and unzip into {raw_dir}"
        )
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
        print("Extracted", z.name)
    (raw_dir / "DOWNLOAD_INFO.json").write_text(json.dumps({
        "competition": C.COMPETITION,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "rules_url": f"https://www.kaggle.com/competitions/{C.COMPETITION}/rules",
    }, indent=2))


# ----------------------------------------------------------------------------- loading
def _find(raw_dir: Path, filename: str) -> Path | None:
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.name.lower() == filename.lower():
            return p
    return None


def _canonicalise(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in C.COLUMN_ALIASES[kind].items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canon
                break
    df = df.rename(columns=rename)
    missing = [c for c in C.REQUIRED[kind] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[{kind}] required columns {missing} not found. Columns present: {list(df.columns)}.\n"
            f"Edit COLUMN_ALIASES in src/config.py to map them."
        )
    return df


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read all CSVs as strings (so IDs are never converted to numbers). Empty cells stay as ''."""
    out, problems = {}, []
    for kind, fname in C.FILE_NAMES.items():
        path = _find(raw_dir, fname)
        if path is None:
            problems.append(fname)
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        out[kind] = _canonicalise(df, kind)
        out[kind].attrs["source_file"] = str(path)
    if problems:
        present = sorted(p.name for p in raw_dir.rglob("*") if p.is_file())
        raise SystemExit(f"Missing files {problems} in {raw_dir}. Files found: {present}")
    return out


# ----------------------------------------------------------------------------- text / id helpers
_WS = re.compile(r"\s+")


def clean_text(s) -> str:
    """Unicode-normalise, remove control chars / zero-width chars, collapse whitespace. Case is preserved."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = "".join(ch for ch in s if ch in "\n\t " or unicodedata.category(ch)[0] != "C")
    return _WS.sub(" ", s).strip()


def parse_id_list(value) -> list[str]:
    """Parse gold/cited ID lists in any common format: JSON list, python list, or , ; | space separated."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null", "[]"}:
            return []
        if s[0] in "[(":
            try:
                items = json.loads(s)
            except Exception:
                try:
                    items = ast.literal_eval(s)
                except Exception:
                    items = re.split(r"[,;|\s]+", s.strip("[]()"))
        elif re.search(r"[,;|]", s):
            items = re.split(r"\s*[,;|]\s*", s)
        else:
            items = s.split()
        if isinstance(items, (str, int)):
            items = [items]
    out, seen = [], set()
    for it in items:
        t = str(it).strip().strip("'\"")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def detect_id_format(*series: pd.Series) -> str:
    """Infer how cited_chunk_ids should be written from sample_submission (or train gold ids)."""
    for ser in series:
        for v in ser:
            s = str(v).strip()
            if not s:
                continue
            if s.startswith("["):
                return "json"
            for sep in (";", "|", ","):
                if sep in s:
                    return sep + " " if sep + " " in s else sep
            if " " in s:
                return " "
    return ","


def format_ids(ids: list[str], fmt: str) -> str:
    return json.dumps(ids) if fmt == "json" else fmt.join(ids)


# ----------------------------------------------------------------------------- inspection
def inspect(dfs: dict[str, pd.DataFrame]) -> dict:
    """Step 2: file names, columns, row counts, missing values, duplicate identifiers."""
    rep: dict = {}
    id_col = {"corpus": "chunk_id", "train": "question_id", "test": "question_id", "sample": "question_id"}
    for kind, df in dfs.items():
        rep[kind] = {
            "file": df.attrs.get("source_file"),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "empty_values_per_column": {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns},
            "duplicate_ids": int(df[id_col[kind]].duplicated().sum()),
        }
    corp = dfs["corpus"]
    rep["corpus"]["duplicate_chunk_text_rows"] = int(corp["chunk_text"].duplicated().sum())
    lens = corp["chunk_text"].str.split().str.len()
    rep["corpus"]["chunk_length_words"] = {k: float(v) for k, v in lens.describe().round(1).to_dict().items()}
    for c in ("doc_id", "doc_type"):
        if c in corp.columns:
            rep["corpus"][f"unique_{c}"] = int(corp[c].nunique())
    tr = dfs["train"]
    rep["train"]["gold_ids_per_question"] = {
        k: float(v) for k, v in tr["gold_chunk_ids"].map(lambda x: len(parse_id_list(x))).describe().round(2).to_dict().items()
    }
    rep["overlap_train_test_question_ids"] = int(len(set(tr["question_id"]) & set(dfs["test"]["question_id"])))
    return rep


# ----------------------------------------------------------------------------- cleaning + validation
def clean_and_validate(dfs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Steps 3 & 6: clean text, validate links question_id -> gold_chunk_ids -> chunk_id (-> doc_id), log every exclusion."""
    log: dict = {}

    # corpus
    corpus = dfs["corpus"].copy()
    n0 = len(corpus)
    corpus["chunk_id"] = corpus["chunk_id"].str.strip()
    corpus["chunk_text"] = corpus["chunk_text"].map(clean_text)
    for c in ("title", "doc_id", "doc_type"):
        if c in corpus.columns:
            corpus[c] = corpus[c].map(clean_text)
    empty = (corpus["chunk_text"] == "") | (corpus["chunk_id"] == "")
    log["corpus_dropped_empty_text_or_id"] = int(empty.sum())
    corpus = corpus[~empty]
    log["corpus_dropped_duplicate_chunk_id"] = int(corpus["chunk_id"].duplicated().sum())
    corpus = corpus.drop_duplicates("chunk_id", keep="first").reset_index(drop=True)
    # NOTE: duplicate chunk *texts* with different ids are kept: they may be distinct gold targets.
    log["corpus_rows_before"], log["corpus_rows_after"] = n0, len(corpus)
    valid_ids = set(corpus["chunk_id"])

    # labelled questions
    train = dfs["train"].copy()
    m0 = len(train)
    train["question"] = train["question"].map(clean_text)
    train["answer"] = train["answer"].map(clean_text)
    train["gold_list"] = train["gold_chunk_ids"].map(parse_id_list)
    bad = (train["question"] == "") | (train["question_id"].str.strip() == "")
    log["train_dropped_empty_question_or_id"] = int(bad.sum())
    train = train[~bad]
    log["train_dropped_duplicate_question_id"] = int(train["question_id"].duplicated().sum())
    train = train.drop_duplicates("question_id", keep="first")
    log["train_questions_with_empty_answer"] = int((train["answer"] == "").sum())

    unknown = train["gold_list"].map(lambda ids: [i for i in ids if i not in valid_ids])
    log["train_gold_ids_not_in_corpus"] = int(unknown.map(len).sum())
    train["gold_list"] = train["gold_list"].map(lambda ids: [i for i in ids if i in valid_ids])
    no_gold = train["gold_list"].map(len) == 0
    log["train_dropped_no_valid_gold_chunk"] = int(no_gold.sum())
    train = train[~no_gold].reset_index(drop=True)
    train["gold_chunk_ids"] = train["gold_list"].map(json.dumps)
    train = train.drop(columns=["gold_list"])
    log["train_rows_before"], log["train_rows_after"] = m0, len(train)

    # test
    test = dfs["test"].copy()
    test["question_id"] = test["question_id"].str.strip()
    test["question"] = test["question"].map(clean_text)
    log["test_rows"] = len(test)
    log["test_empty_questions"] = int((test["question"] == "").sum())  # kept: every test id needs an answer row
    log["test_ids_overlapping_train"] = int(len(set(test["question_id"]) & set(train["question_id"])))

    return {"corpus": corpus, "train": train, "test": test, "sample": dfs["sample"]}, log


def split(train: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 4: reproducible train/validation split (seed recorded in the log)."""
    tr, va = train_test_split(train, test_size=val_fraction, random_state=seed, shuffle=True)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def prepare(raw_dir: Path, out_dir: Path, results_dir: Path, seed: int = C.SEED, val_fraction: float = C.VAL_FRACTION) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_dir)
    report = inspect(raw)
    (results_dir / "data_inspection_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:4000])

    clean, log = clean_and_validate(raw)
    tr, va = split(clean["train"], val_fraction, seed)
    log.update(seed=seed, val_fraction=val_fraction, n_train=len(tr), n_val=len(va))
    (results_dir / "data_cleaning_log.json").write_text(json.dumps(log, indent=2))

    clean["corpus"].to_csv(out_dir / "corpus_clean.csv", index=False)
    tr.to_csv(out_dir / "train_split.csv", index=False)
    va.to_csv(out_dir / "val_split.csv", index=False)
    clean["test"].to_csv(out_dir / "test_clean.csv", index=False)
    clean["sample"].to_csv(out_dir / "sample_submission.csv", index=False)
    print("\nCleaning log:\n" + json.dumps(log, indent=2))
    print(f"\nSaved cleaned data to {out_dir}/ and reports to {results_dir}/")


def load_processed(proc_dir: Path) -> dict[str, pd.DataFrame]:
    files = [
        ("corpus", "corpus_clean.csv"),
        ("train", "train_split.csv"),
        ("val", "val_split.csv"),
        ("test", "test_clean.csv"),
        ("sample", "sample_submission.csv"),
    ]
    missing = [filename for _, filename in files if not (proc_dir / filename).is_file()]
    if missing:
        raise SystemExit(
            f"Prepared dataset files are missing from '{proc_dir}': {', '.join(missing)}. "
            "Run `python main.py prepare` after placing corpus.csv, train.csv, "
            "test.csv, and sample_submission.csv in data/raw."
        )
    d = {
        key: pd.read_csv(proc_dir / filename, dtype=str, keep_default_na=False)
        for key, filename in files
    }
    for k in ("train", "val"):
        d[k]["gold_list"] = d[k]["gold_chunk_ids"].map(parse_id_list)
    return d
