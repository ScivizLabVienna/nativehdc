"""G4 (PLAN-nativehdc): doc-level HV store + SQLite doc_sentences; leave-one-out sentence retrieval on
held-out TinyStories docs. Gate: top-1 doc accuracy > random baseline with margin, named asserts exit 0.
SQLite = ground truth; the HV ItemStore is a rebuildable index (JSONL/DB doctrine, risk #543)."""
import os, sqlite3, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "..", "runs")
DB = os.path.join(RUNS, "g4_docs.db")
N_DOCS, N_HELD, DIM = 1024, 64, 4096

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


if __name__ == "__main__":
    from datasets import load_dataset
    from holovec import VSA
    from holovec.retrieval import ItemStore
    from harness import tkz
    from holovec.encoders import NGramEncoder

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)  # HF cache, offline OK
    docs = []
    for row in ds:
        sents = [s.strip() for s in row["text"].replace("\n", " ").split(". ") if len(tkz(s)) >= 3]
        if len(sents) >= 4:
            docs.append(sents)
        if len(docs) >= N_DOCS:
            break
    print(f"docs={len(docs)} sentences={sum(len(d) for d in docs)}")

    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE doc_sentences(doc INTEGER, idx INTEGER, sent TEXT)")
    con.executemany("INSERT INTO doc_sentences VALUES(?,?,?)",
                    [(d, i, s) for d, ds_ in enumerate(docs) for i, s in enumerate(ds_)])
    con.commit()

    import random
    random.seed(1337)
    held = {d: random.randrange(len(docs[d])) for d in random.sample(range(len(docs)), N_HELD)}

    m = VSA.create("FHRR", dim=DIM, seed=7)
    enc = NGramEncoder(m, n=1, seed=7)
    store, labels = ItemStore(m), {}
    for d, sents in enumerate(docs):  # held-out sentences are never indexed — pure leave-one-out
        for i, s in enumerate(sents):
            if (d, i) not in [(k, v) for k, v in held.items()]:
                lab = f"{d}\t{i}"
                store.add(lab, enc.encode(tkz(s))); labels[lab] = d

    hits = top5 = 0
    for d, i in held.items():
        q = enc.encode(tkz(docs[d][i]))
        got = store.query(q, k=5)
        if labels[got[0][0]] == d:
            hits += 1
        if any(labels[g[0]] == d for g in got):
            top5 += 1
    acc, acc5 = hits / len(held), top5 / len(held)
    rnd = 1.0 / len(docs)
    print(f"leave-one-out top-1 doc acc={acc:.3f} top-5={acc5:.3f} random=1/{len(docs)}={rnd:.5f}")

    n_db = con.execute("SELECT COUNT(*) FROM doc_sentences").fetchone()[0]
    check("db_row_count_matches", n_db == sum(len(d) for d in docs))
    check("retrieval_beats_random_10x", acc > 10 * rnd)  # PLAN gate: "> random" — margin required, no soft pass
    check("cites_doc_from_sqlite", all(con.execute("SELECT sent FROM doc_sentences WHERE doc=? AND idx=?", (d, i)).fetchone() for d, i in list(held.items())[:8]))

    bad = [k for k, v in checks.items() if not v]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} gates PASS" + (f" | FAILED: {bad}" if bad else " -> G4 gate OK"))
    sys.exit(1 if bad else 0)
