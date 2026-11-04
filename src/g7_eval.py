"""G7: zero-shot ProofWriter depth-stratified 3-choice likelihood eval. Spec: docs/EVALS-nativehdc.md."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PQ_PATH = os.path.join(HERE, "..", "data", "proofwriter_test.parquet")
HF_URL = "https://huggingface.co/api/datasets/tasksource/proofwriter/parquet/default/test/0.parquet"
ANS = ["True", "False", "Unknown"]


def load_rows(per_depth=500):
    import urllib.request, pyarrow.parquet as pq
    if not os.path.exists(PQ_PATH):
        os.makedirs(os.path.dirname(PQ_PATH), exist_ok=True)
        print("downloading test parquet..."); urllib.request.urlretrieve(HF_URL, PQ_PATH)
    tb = pq.read_table(PQ_PATH, columns=["theory", "question", "answer", "config"]).to_pylist()
    want = {f"depth-{d}" for d in range(6)}
    out, cnt = [], collections.Counter()
    for r in tb:  # single pass, first-N per stratum (file order is generated-random)
        if r["config"] in want and cnt[r["config"]] < per_depth:
            cnt[r["config"]] += 1; out.append(r)
    return out


def main():
    from transformers import AutoTokenizer
    from model import TinyLM
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    ckpt = os.environ.get("GHRR_CKPT", os.path.join(HERE, "..", "runs", "ghrr_best.pt"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained("gpt2")
    m = TinyLM(attn_type=os.environ.get("GHRR_ARM", "ghrr"), tau=float(os.environ.get("GHRR_TAU", "4")), hv_dim=int(os.environ.get("GHRR_HV", "4096")))  # ckpt carries no tau/arm; envs must match training
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval().to(dev)
    a1 = [tk.encode(" " + a, return_tensors="pt")[0, 0].item() for a in ANS]  # first-token scoring only: keeps input <= ctx
    rows = load_rows()[:limit] if limit else load_rows()
    acc, n = collections.Counter(), collections.Counter()
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
        for i, r in enumerate(rows):
            ids = tk.encode(r["theory"] + f"\nQuestion: {r['question']}\nAnswer:")[-(m.ctx - 8):] or [tk.eos_token_id]
            x = torch.tensor([ids], device=dev)
            lg = torch.log_softmax(m(x)[0][0, -1].float(), dim=-1)
            pred = ANS[max(range(3), key=lambda k: lg[a1[k]].item())]
            acc[r["config"]] += pred == r["answer"]; n[r["config"]] += 1
            if limit and i < 8: print(f"  {r['config']} gold={r['answer']:7s} pred={pred:7s} scores={[round(lg[a].item(),2) for a in a1]}")
    agg = sum(acc.values()) / max(sum(n.values()), 1)
    for k in sorted(n): print(f"{k}: acc {acc[k]}/{n[k]} = {acc[k]/n[k]:.3f}")
    print(f"OVERALL n={sum(n.values())} acc={agg:.3f} (chance=0.333; gate per spec: ghrr > vanilla by >1pp + shallower depth slope)")


if __name__ == "__main__":
    main()
