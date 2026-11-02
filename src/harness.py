"""G3 harness (PLAN-nativehdc G3): HDC gates ALWAYS run before the model; the small LM never calls tools.
Pipeline: detect_adversarial -> facts gate -> generate -> abstain check. One JSONL trace line per ask().
Gate verify: src/g3_test.py (named asserts, exit 0 = pass)."""
import json, re, sqlite3, time
from collections import Counter

from holovec import VSA
from holovec.encoders import NGramEncoder
from holovec.retrieval import ItemStore

REFUSAL = "Answerable: no"  # system-level protocol (risk #536 notes), never the LM's wording


def norm(t):
    return " ".join((t or "").lower().replace(",", " ").replace(".", " ").split())


# Content-word tokenizer: function words collide on every query and saturated HV sim (probe-measured),
# so gates run on stemmed content words only. Limitation: single-synonym near-collisions sit at the
# threshold band; mitigate with longer queries or alias ingestion at write time (see hdc skill doctrine).
STOP = frozenset("what who where when how why which do does did is are was were be am the a an of to in "
                 "on at for me you your i it its this that with and or as if my we our there here tell "
                 "about say color name".split())


def _stem(w):
    for s in ("ing", "ed", "ly"):
        if len(w) > len(s) + 2 and w.endswith(s):
            return _stem(w[:-len(s)])
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def tkz(t):
    return [_stem(w) for w in norm(t).split() if w not in STOP]


class Harness:
    def __init__(self, generate, db_path, exemplar_file, dim=4096, seed=7,
                 adv_thr=0.28, fact_thr=0.30, trace_path=None):  # thresholds probe-calibrated, see g3_test margins
        self.gen, self.trace_path = generate, trace_path
        self.adv_thr, self.fact_thr = adv_thr, fact_thr
        m = VSA.create("FHRR", dim=dim, seed=seed)  # GHRR m=1 == FHRR kernel (G0-verified)
        self.enc = NGramEncoder(m, n=1, seed=seed)  # word-unigram bundle: order-tolerant, selective
        v = lambda t: self.enc.encode(tkz(t))
        self.adv, self.n_adv = ItemStore(m), 0
        for i, line in enumerate(open(exemplar_file, encoding="utf-8")):
            if line.strip():
                self.adv.add(f"adv:{i}", v(line)); self.n_adv += 1
        self.facts, self.n_facts = ItemStore(m), 0
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE IF NOT EXISTS triples(s TEXT,p TEXT,o TEXT,doc TEXT)")
        for s, p, o, doc in con.execute("SELECT s,p,o,doc FROM triples"):
            self.facts.add(f"{s}\t{p}\t{o}\t{doc}", v(f"{s} {p} {o}")); self.n_facts += 1
        self.con = con

    def _top(self, store, text):
        if not (self.n_adv if store is self.adv else self.n_facts):
            return None, -1.0
        label, sim = store.query(self._vec(text), k=1)[0]
        return label, float(sim)

    _vec = lambda self, t: self.enc.encode(tkz(t))

    @staticmethod
    def _degenerate(out):
        if len(out.strip()) < 2:
            return True
        w = norm(out).split()
        c = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
        return bool(c) and max(c.values()) >= 3  # any 4-gram repeated 3+ times = token-loop

    def ask(self, q):
        t0 = time.time(); rec = {"q": q}
        _, a_sim = self._top(self.adv, q)
        rec["adv_sim"] = round(a_sim, 3)
        if a_sim >= self.adv_thr:
            return self._done(rec, q, "blocked_adversarial", None, None, t0)
        lab, f_sim = self._top(self.facts, q)
        rec["fact_sim"] = round(f_sim, 3)
        if f_sim < self.fact_thr or not self.n_facts:
            return self._done(rec, q, "refused_ungrounded", None, None, t0)
        s, p, o, doc = lab.split("\t")
        cite = self.con.execute("SELECT doc FROM triples WHERE s=? AND p=? AND o=?", (s, p, o)).fetchone()
        rec["cite"] = doc if cite else None  # ground truth stays in SQLite, HV store only indexes it
        try:
            ans = self.gen(q)
        except Exception as e:  # fail closed on generator failure, never leak a crash as an answer
            return self._done(rec, q, f"refused_generator_error:{e.__class__.__name__}", None, None, t0)
        if self._degenerate(ans):
            return self._done(rec, q, "refused_model_degenerate", None, None, t0)
        return self._done(rec, q, "answered", ans, rec.get("cite"), t0)

    def _done(self, rec, q, decision, answer, cite, t0):
        rec.update({"decision": decision, "answerable": decision == "answered",
                    "answer": answer, "cites_doc": cite, "refusal": None if decision == "answered" else REFUSAL,
                    "t_ms": int((time.time() - t0) * 1000)})
        if self.trace_path:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")  # risk R3: observability-first, one line per step
        return rec
