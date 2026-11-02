"""G0 gate: env sanity — CUDA torch live + holovec GHRR algebra asserts. exit 0 = pass."""
import numpy as np
import torch
from holovec import VSA

assert torch.cuda.is_available(), "torch.cuda.is_available() False"
dev_name = torch.cuda.get_device_name(0)
x = torch.randn(256, 256, dtype=torch.cfloat, device="cuda")  # force a real kernel launch
y = (x @ x.conj()).abs().sum().item()
assert np.isfinite(y), "cuda complex matmul not finite"

# GHRR bind/unbind exact recovery + non-commutativity (holovec oracle, per PLAN G0)
m3 = VSA.create("GHRR", dim=256, matrix_size=3, diagonality=0.0, seed=42)
a, b = m3.random(seed=1), m3.random(seed=2)
rec = m3.unbind(m3.bind(a, b), b)
s_rec = float(m3.similarity(rec, a))
s_ab_ba = float(m3.similarity(m3.bind(a, b), m3.bind(b, a)))
assert s_rec > 0.999, f"GHRR bind/unbind recovery sim {s_rec:.4f} <= 0.999"
assert s_ab_ba < 0.3, f"GHRR non-commutativity check failed: sim(ab,ba)={s_ab_ba:.4f}"

# m=1 must behave like FHRR: binding commutes (element-wise complex mul)
m1 = VSA.create("GHRR", dim=256, matrix_size=1, seed=7)
c, d = m1.random(seed=3), m1.random(seed=4)
s_comm = float(m1.similarity(m1.bind(c, d), m1.bind(d, c)))
assert s_comm > 0.999, f"m=1 GHRR should commute (FHRR-equiv), got {s_comm:.4f}"

print(f"G0 PASS cuda={dev_name} torch={torch.__version__} rec_sim={s_rec:.6f} "
      f"nc_sim={s_ab_ba:.4f} m1_comm_sim={s_comm:.6f}")
