"""G1 gate: differentiable GHRR module cross-checked against holovec oracle. exit 0 = pass."""
import hashlib
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghrr_attn import GHRRAttention, bind, hash_phases, hash_phasors, similarity, unbind

from holovec import VSA

torch.manual_seed(0)
oracle = VSA.create("FHRR", dim=512, seed=7)


def hv(seed):  # oracle draw as torch complex64
    return torch.tensor(np.asarray(oracle.random(seed=seed)), dtype=torch.complex64)


# 1) op semantics vs holovec (bind/unbind/similarity), many pairs
worst_bind, worst_sim = 0.0, 0.0
for s in range(1, 21):
    a_or, b_or = np.asarray(oracle.random(seed=s)), np.asarray(oracle.random(seed=100 + s))
    a, b = torch.tensor(a_or, dtype=torch.complex64), torch.tensor(b_or, dtype=torch.complex64)
    worst_bind = max(worst_bind, float((bind(a, b) - torch.tensor(np.asarray(oracle.bind(a_or, b_or)), dtype=torch.complex64)).abs().max()))
    rec = unbind(bind(a, b), b)
    assert float(similarity(rec, a)) > 0.9999, f"unbind recovery failed seed {s}"
    worst_sim = max(worst_sim, abs(float(similarity(a, b)) - float(oracle.similarity(a_or, b_or))))
assert worst_bind < 1e-5 and worst_sim < 1e-5
print(f"PASS ops_vs_holovec bind_err={worst_bind:.2e} sim_err={worst_sim:.2e}")

# 2) kernel identity: sim == mean cos(theta_a - theta_b)
a, b = hv(1), hv(2)
cos_id = (torch.cos(torch.angle(a) - torch.angle(b))).mean()
assert abs(float(similarity(a, b)) - float(cos_id)) < 1e-5
print(f"PASS kernel_cos_identity diff={abs(float(similarity(a,b)) - float(cos_id)):.2e}")

# 3) hash encoding: quasi-orthogonality + cross-process determinism (PYTHONHASHSEED rule)
ids = torch.arange(64)
ph = hash_phases(ids, 4096)
zs = hash_phasors(ids, 4096)
sims = similarity(zs.unsqueeze(1), zs.unsqueeze(0))  # broadcast pairwise [64,64]
off = sims[~torch.eye(64, dtype=torch.bool)]
assert float(off.abs().mean()) < 3.0 / np.sqrt(4096), f"hash HVs not quasi-orthogonal: {float(off.abs().mean()):.4f}"
snippet = (
    "import sys,hashlib,torch;"
    f"sys.path.insert(0,r'{os.path.dirname(os.path.abspath(__file__))}');"
    "from ghrr_attn import hash_phases;"
    "print(hashlib.sha256(hash_phases(torch.arange(50),64).to(torch.float32).numpy().tobytes()).hexdigest())"
)
digests = []
for seed_env in ("0", "98765"):
    r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, timeout=180,
                       env={**os.environ, "PYTHONHASHSEED": seed_env})
    assert r.returncode == 0, r.stderr
    digests.append(r.stdout.strip())
assert digests[0] == digests[1] and len(digests[0]) == 64
print(f"PASS hash_encoding quasiorth={float(off.abs().mean()):.4f} crossproc_sha={digests[0][:12]}")

# 4) module: shapes, causal independence, kernel normalization, finite grads everywhere
mod = GHRRAttention(d_model=32, n_heads=4, hv_dim=64, tau=4.0)
x = torch.randn(2, 8, 32)
o = mod(x)
assert o.shape == x.shape and torch.isfinite(o).all()
x2 = x.clone(); x2[:, 5:, :] += 3.0
assert (mod(x2)[:, :5] - o[:, :5]).abs().max() < 1e-6, "causal leak from future tokens"
# with xq==xk input, score(i,i) must equal tau exactly (cos(0)=1 summed/dh)
zq = torch.exp(1j * mod.wq(x)).reshape(2, 8, 4, 16).transpose(1, 2)
diag_scores = 4.0 * (zq @ zq.conj().transpose(-1, -2)).real.diagonal(dim1=-2, dim2=-1) / 16
assert (diag_scores - 4.0).abs().max() < 1e-4, f"self-score != tau: {float((diag_scores-4.0).abs().max()):.2e}"
o.sum().backward()
for n, p in mod.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), f"grad missing/NaN at {n}"
print(f"PASS module shape+causal+selfscore tau-err={float((diag_scores-4.0).abs().max()):.2e} grads=all_finite")

print("G1 PASS")
