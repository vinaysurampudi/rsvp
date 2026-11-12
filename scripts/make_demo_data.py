"""Creates a SMALL SYNTHETIC dataset in the schema described in the report, ONLY to smoke-test the pipeline offline.
Never report results from this data."""
import argparse, json, random
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", default="data_demo/raw")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rnd = random.Random(0)

products = ["Nimbus", "Quartz", "Helix", "Orbit", "Vertex", "Lumen", "Cobalt", "Zephyr", "Atlas", "Ember"]
facts = [("default port", "listens on port {n} by default"), ("timeout", "the request timeout is {n} seconds"),
         ("max connections", "supports up to {n} concurrent connections"), ("retry limit", "retries failed calls {n} times"),
         ("cache size", "allocates {n} megabytes of cache"), ("log retention", "keeps logs for {n} days")]
corpus, qs = [], []
for pi, prod in enumerate(products):
    for fi, (topic, tmpl) in enumerate(facts):
        n = rnd.randint(2, 9000)
        cid = f"c{pi:02d}{fi}"
        text = f"  {prod} service configuration. The {prod} server {tmpl.format(n=n)}.  This setting can be changed in the {prod.lower()}.conf file.\u00a0 "
        corpus.append({"chunk_id": cid, "doc_id": f"d{pi:02d}", "title": f"{prod} guide", "doc_type": "manual", "chunk_text": text})
        qs.append({"question_id": f"q{pi:02d}{fi}", "question": f"What is the {topic} for {prod}?",
                   "answer": f"The {prod} server {tmpl.format(n=n)}.", "gold_chunk_ids": json.dumps([cid])})
# messy rows to exercise cleaning: duplicate chunk id, empty text, unknown gold id
corpus.append(dict(corpus[0])); corpus.append({"chunk_id": "cEMPTY", "doc_id": "d99", "title": "", "doc_type": "", "chunk_text": "   "})
qs.append({"question_id": "qBAD", "question": "Unanswerable?", "answer": "x", "gold_chunk_ids": json.dumps(["does_not_exist"])})
rnd.shuffle(qs)
labelled, test = qs[:-12], qs[-12:]
pd.DataFrame(corpus).to_csv(out / "corpus.csv", index=False)
pd.DataFrame(labelled).to_csv(out / "train.csv", index=False)
pd.DataFrame([{"question_id": "t" + t["question_id"], "question": t["question"]} for t in test]).to_csv(out / "test.csv", index=False)
pd.DataFrame([{"question_id": "t" + t["question_id"], "answer": "", "cited_chunk_ids": "c000;c001"} for t in test]).to_csv(out / "sample_submission.csv", index=False)
print("demo data written to", out)
