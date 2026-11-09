"""E1 champion content encoder (G14; gate = src/g10_probe_nav.py --content).
lm+tok = landmark-keyed persistence merge bins (top-15 sentence-kNN clouds) + hard top-idf token
bins, composed by random-feature phases: item HV = exp(i*arg sum_b contrib(b)); contrib is iid per-dim
gaussian seeded by blake2b(bin label) — same seam family as ghrr_attn.hash_phasors, but bins carry
shared labels so cross-item cosine = token-overlap kernel (graded similarity; id-hash is flat).
Frozen g10 bind/unbind/pose algebra applies unchanged. Corpus stats (df/idf, kNN) are encoder inputs:
recompute on corpus rebuild (bins are pure functions of text + corpus index).
Reference impl + all measured numbers: S:/research/e1-topology-hv (p2b gate PASS 2026-11-08);
this module is the nativehdc-owned port, parity asserted by the g10 --content gate."""
import hashlib
import math
import sqlite3

import numpy as np
import torch

from ghrr_attn import hash_phasors

STOP = set("a an and are as at be but by for from had has he her his i if in is it its me my no not "
           "of on or our said so she than that the their them then there they this to too us was we "
           "were what when where which while you your with".split())
_GCACHE = {}


def tk(s):
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split()
            if w not in STOP and len(w) > 2}


def bin_contrib(bintext, dim):
    """deterministic iid per-dim contribution for a bin label (blake2b-seeded)."""
    key = (bintext, dim)
    if key not in _GCACHE:
        seed = int.from_bytes(hashlib.blake2b(bintext.encode(), digest_size=8).digest(), "little")
        g = np.random.default_rng(seed).standard_normal(2 * dim)
        _GCACHE[key] = (g[:dim] + 1j * g[dim:]).astype(np.complex64)
    return _GCACHE[key]


def rf_encode(binlists, dim):
    """champion composition: bins -> unit-modulus HV; unnormalized cos = overlap kernel."""
    out = np.empty((len(binlists), dim), dtype=np.complex64)
    for r, bl in enumerate(binlists):
        z = sum((bin_contrib(b, dim) for b in bl), 0j)
        out[r] = np.exp(1j * np.arctan2(z.imag, z.real))
    return out


def mst_edges(S):
    """Prim MST on similarity submatrix; returns list of (j, sim, parent)."""
    m = S.shape[0]
    intree, best, par = np.zeros(m, bool), S[0].copy(), np.zeros(m, int)
    out = []
    intree[0] = True
    for _ in range(m - 1):
        j = int(np.argmax(np.where(intree, -np.inf, best)))
        out.append((j, float(best[j]), int(par[j])))
        intree[j] = True
        best = np.maximum(best, S[j])
    return out


def landmark_bins(S, pts, sig_top, nbins=8):
    """bin key = (merge-scale bucket, top-1 idf token of lexically-smaller endpoint)."""
    bucket = lambda d: f"{min(int(d * nbins), nbins - 1)}"  # noqa: E731
    tok_of = {int(p): sig_top[int(p)] for p in pts}
    mb = []
    for j, w, k in mst_edges(S):
        lm = tok_of[min(int(pts[j]), int(pts[k]))]
        mb.append(f"lm{bucket(1.0 - w)}:{lm}")
    return mb


def load_sentences(db, n_docs=300, seed=7):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    by_doc = {}
    for d, i, s in con.execute("select doc, idx, sent from doc_sentences"):
        by_doc.setdefault(d, []).append((i, s))
    rng = np.random.default_rng(seed)
    docs = sorted(by_doc)
    out = []
    for di in sorted(rng.choice(len(docs), size=n_docs, replace=False)):
        for i, s in sorted(by_doc[docs[di]])[:10]:
            if tk(s):
                out.append((docs[di], i, s))
    con.close()
    return out


def corpus_bins(sigs):
    """(lm+tok bin lists, top-idf-token list) for a corpus of token sets — exact champion arm spec
    (p2b r8: nb=15 cloud, 8 token bins + always-kept top token)."""
    N = len(sigs)
    dfreq = {}
    for s in sigs:
        for w in s:
            dfreq[w] = dfreq.get(w, 0) + 1
    wt = lambda w: math.log(N / dfreq[w])  # noqa: E731
    vocab = sorted(dfreq)
    Xd = np.zeros((N, len(vocab)), dtype=np.float32)
    vi = {w: i for i, w in enumerate(vocab)}
    for r, s in enumerate(sigs):
        Xd[r, [vi[w] for w in s]] = [wt(w) for w in s]
        Xd[r] /= np.linalg.norm(Xd[r])
    S = Xd @ Xd.T

    def top_tok(r):
        s = sorted(sigs[r], key=lambda w: (-wt(w), w))
        return s[0] if s else "_"

    sig_top = np.array([top_tok(r) for r in range(N)])
    out = []
    for i in range(N):
        nb = np.argsort(-S[i])[:15][1:]
        pts = np.concatenate(([i], nb))
        a = landmark_bins(S[np.ix_(pts, pts)], pts, sig_top)
        tok = ["w:" + w for w in sorted(sigs[i], key=lambda w: (-wt(w), w))[:8]]
        out.append(a + [f"w:{top_tok(i)}"] + tok)
    return out

