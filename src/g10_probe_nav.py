"""G10 probe: is the HDC layer fit for GPS-denied drone nav / Hermes memory roles?
Measures what the decision actually turns on: encode rate, place-memory capacity
at BOTH champion geometries (D=1024 and 4096), and recall tick latency. CPU-only."""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ghrr_attn import hash_phasors, bind, unbind

torch.set_num_threads(8)


def enc_rate(D, n=4096):
    hash_phasors(torch.arange(512), D)  # warm
    t0 = time.perf_counter()
    hash_phasors(torch.arange(1_000_000, 1_000_000 + n), D)
    return n / (time.perf_counter() - t0)


def pose_recall(K, D):
    sig = hash_phasors(torch.arange(K), D)
    pose = hash_phasors(9_000_000 + torch.arange(K) * 7, D)
    mem = bind(sig, pose).sum(0)
    r = unbind(mem.unsqueeze(0).expand(K, D), sig)
    top = (r @ pose.conj().T).real.argmax(1)
    return float((top == torch.arange(K)).float().mean())


if __name__ == "__main__":
    for D in (1024, 4096):
        print(f"encode rate @D={D}: {enc_rate(D):,.0f} novel-places/s")
    for D in (1024, 4096):
        print("loop-closure recall @D=%d: " % D + " ".join(
            f"K{k}:{pose_recall(k, D):.2f}" for k in (64, 128, 256, 512, 1024)))
    sig = hash_phasors(torch.arange(128), 1024)
    mem = bind(sig, hash_phasors(9_000_000 + torch.arange(128) * 7, 1024)).sum(0)
    q = sig[3:4].expand(128, 1024)
    t0 = time.perf_counter()
    for _ in range(2000):
        unbind(mem.expand(128, 1024), q)
    dt = (time.perf_counter() - t0) / 2000
    print(f"recall tick (K=128 places, D=1024): {dt*1e3:.2f} ms -> ~{1/dt:,.0f} Hz on 8 CPU threads")
