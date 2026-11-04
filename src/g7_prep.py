"""G7b prep: ProofWriter train split -> uint16 BPE bins (docs/EVALS-nativehdc.md). Run once."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, urllib.request, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data", "proofwriter_ft")
URL = "https://huggingface.co/api/datasets/tasksource/proofwriter/parquet/default/train/0.parquet"
N_TRAIN, N_VAL, SEED = 30000, 2000, 1337  # ~4x the tokens one 5k-step arm needs; rest is replay buffer

def main():
    src = os.path.join(HERE, "..", "data", "proofwriter_train.parquet")
    if not os.path.exists(src):
        print("downloading train parquet..."); urllib.request.urlretrieve(URL, src)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("gpt2")
    rows = pq.read_table(src).to_pylist()  # keep all cols; 'split' col if present filters official split
    rows = [r for r in rows if str(r.get("split", "train")) == "train"] if rows and "split" in rows[0] else rows
    random.Random(SEED).shuffle(rows)
    texts = [(f"{(r['theory'] or '').strip()}\nQuestion: {r['question']}\nAnswer: {r['answer']}.\n") for r in rows]
    ids = [x for line in texts[:N_TRAIN + N_VAL] for x in (tk.encode(line) + [tk.eos_token_id])]
    n_tr = sum(len(tk.encode(l)) + 1 for l in texts[:N_TRAIN])
    arr = np.array(ids, dtype=np.uint16); os.makedirs(D, exist_ok=True)
    arr[:n_tr].tofile(os.path.join(D, "train.bin")); arr[n_tr:].tofile(os.path.join(D, "val.bin"))
    print(f"train.bin {n_tr} tok, val.bin {len(ids)-n_tr} tok -> {D}")

if __name__ == "__main__":
    main()
