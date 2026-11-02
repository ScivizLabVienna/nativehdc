"""Stream TinyStories -> GPT-2 BPE uint16 token bins (train/val). Stops at TRAIN_TOKENS."""
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TRAIN_TOKENS, VAL_TOKENS = 300_000_000, 4_000_000

tok = AutoTokenizer.from_pretrained("gpt2")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
ds = ds.filter(lambda s: len(s["text"]) <= 800, batched=False)

def emit(path, n_target, gen):
    buf, total = [], 0
    with open(path, "wb") as f:
        for sample in gen:
            ids = tok.encode(sample["text"], add_special_tokens=False)
            if not ids:
                continue
            buf.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
            if total >= n_target:
                break
            if len(buf) % 2000 == 0:
                f.write(np.concatenate(buf).tobytes()); buf = []
                print(f"{path}: {total/1e6:.1f}M tok", flush=True)
        if buf:
            f.write(np.concatenate(buf).tobytes())
    return total

os.makedirs(OUT, exist_ok=True)
tr = emit(os.path.join(OUT, "train.bin"), TRAIN_TOKENS, iter(ds))
va = emit(os.path.join(OUT, "val.bin"), VAL_TOKENS, load_dataset("roneneldan/TinyStories", split="validation", streaming=True))
print(f"DONE train={tr/1e6:.1f}M val={va/1e6:.1f}M")
