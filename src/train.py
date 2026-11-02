"""G2 bake-off trainer: identical budget/seed/init for ghrr and vanilla arms. Only kernel differs."""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import TinyLM

# smoke-only knobs via env (full bake-off uses defaults): BAKEOFF_B batch size; CPU forced by CUDA_VISIBLE_DEVICES=""
B = int(os.environ.get("BAKEOFF_B", 16))
CTX = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def batches(tok, n_steps):
    for _ in range(n_steps):
        i = torch.randint(len(tok) - CTX - 1, (B,))
        x = torch.stack([tok[j : j + CTX] for j in i]).long()
        y = torch.stack([tok[j + 1 : j + CTX + 1] for j in i]).long()
        yield x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)


def evaluate(m, val):
    m.eval(); tot = 0.0
    with torch.inference_mode(), torch.autocast(DEV, dtype=torch.bfloat16):
        for j in range(20):
            s = j * B * (CTX + 1)
            xs = torch.stack([val[s + b * (CTX + 1) : s + b * (CTX + 1) + CTX] for b in range(B)]).long().to(DEV)
            ys = torch.stack([val[s + b * (CTX + 1) + 1 : s + b * (CTX + 1) + CTX + 1] for b in range(B)]).long().to(DEV)
            _, loss = m(xs, ys)
            tot += loss.item() * B * CTX
    m.train()
    return math.exp(min(tot / (20 * B * CTX), 20))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["ghrr", "vanilla"], required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: 300M-token budget")
    ap.add_argument("--lr", type=float, default=6e-4)
    args = ap.parse_args()

    d = os.path.join(HERE, "..", "data")
    tr = torch.from_numpy(np.asarray(np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="r")))
    va = torch.from_numpy(np.asarray(np.memmap(os.path.join(d, "val.bin"), dtype=np.uint16, mode="r")))

    torch.manual_seed(1337); np.random.seed(1337)
    m = TinyLM(d_model=384, n_layer=6, n_heads=8, hv_dim=4096, attn_type=args.attn).to(DEV)
    steps = args.steps or int(300e6 // (B * CTX))
    runs = os.path.join(HERE, "..", "runs"); os.makedirs(runs, exist_ok=True)
    logf = open(os.path.join(runs, f"{args.attn}.jsonl"), "a")
    print(f"arm={args.attn} params={sum(p.numel() for p in m.parameters())/1e6:.1f}M steps={steps}", flush=True)

    groups = [dict(params=[p for p in m.parameters() if p.ndim >= 2], weight_decay=0.1),
              dict(params=[p for p in m.parameters() if p.ndim < 2], weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))

    t0, done, best = time.time(), 0, float("inf")
    for it, (x, y) in enumerate(batches(tr, steps), 1):
        lr = args.lr * it / 500 if it <= 500 else \
            args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(it / steps, 1))))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(DEV, dtype=torch.bfloat16):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        done += B * CTX
        if it % 100 == 0 or it == steps:
            rec = dict(arm=args.attn, step=it, loss=round(loss.item(), 4), lr=round(lr, 6),
                       toks_M=round(done / 1e6, 2), tok_s=int(done / (time.time() - t0)))
            if it % 1000 == 0 or it == steps:
                rec["ppl"] = round(evaluate(m, va), 3)
                if rec["ppl"] < best:
                    best = rec["ppl"]
                    torch.save(m.state_dict(), os.path.join(runs, f"{args.attn}_best.pt"))
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
    print(f"DONE arm={args.attn} best_ppl={best:.3f} tokens={done/1e6:.1f}M t={(time.time()-t0)/3600:.2f}h")
