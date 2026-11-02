"""GHRR-attention (m=1 scalar path) — differentiable FHRR-kernel attention + hash encodings.

Paper arXiv 2405.09689v2 §II-B/§IV-E; oracle semantics verified vs holovec in g1_test.py:
  bind = elementwise complex mul, unbind = conj (exact), sim = mean_d Re[conj(a)b] = mean cos(θa-θb).
Only the score function deviates from vanilla attention here: dot-product -> phase kernel.
V path stays real so the G2 bake-off changes exactly ONE factor vs the baseline.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn


def hash_phases(ids: torch.Tensor, dim: int) -> torch.Tensor:
    """token_id -> deterministic phases [0,2pi), shape ids.shape+(dim,).
    blake2b(id||counter) stream; NEVER builtin hash() (PYTHONHASHSEED rule)."""
    flat = ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    out = np.empty((flat.size, dim), dtype=np.float64)
    for i, tid in enumerate(flat):  # shortcut: per-id loop; rebuild as batch kernel if vocab sweep >1e6 ids/s
        h, ctr, j = hashlib.blake2b, 0, 0
        while j < dim:
            d = hashlib.blake2b(
                int(tid).to_bytes(8, "little", signed=True) + ctr.to_bytes(4, "little"),
                digest_size=64,
            ).digest()
            u = np.frombuffer(d[: (dim - j) * 8] or d, dtype=np.uint64)[: dim - j].astype(np.float64)
            out[i, j : j + u.size] = (u / 2.0**64) * 2.0 * np.pi
            j += u.size
            ctr += 1
    return torch.tensor(out.reshape(ids.shape + (dim,)), dtype=torch.float64)


def hash_phasors(ids: torch.Tensor, dim: int) -> torch.Tensor:
    """unit-modulus FHRR hypervectors exp(i*phase), complex64."""
    return torch.exp(1j * hash_phases(ids, dim).to(torch.float32))


# --- differentiable ops (mirror holovec FHRR semantics exactly) ---
def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def unbind(c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return c * b.conj()  # unit modulus => conj == inverse, exact recovery


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.conj() * b).real.mean(dim=-1)


class GHRRAttention(nn.Module):
    """Multi-head attention with the m=1 GHRR/FHRR kernel on Q/K; real V (single-factor swap vs vanilla)."""

    def __init__(self, d_model: int, n_heads: int, hv_dim: int, tau: float = 4.0):
        super().__init__()
        assert hv_dim % n_heads == 0
        self.h, self.dh, self.tau = n_heads, hv_dim // n_heads, tau
        self.wq = nn.Linear(d_model, hv_dim)   # -> phases (radians)
        self.wk = nn.Linear(d_model, hv_dim)
        self.wv = nn.Linear(d_model, hv_dim)
        self.wo = nn.Linear(hv_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,d_model]
        B, T, _ = x.shape
        zq = torch.exp(1j * self.wq(x))  # phasors: soft-RFF features of the projected state
        zk = torch.exp(1j * self.wk(x))
        v = self.wv(x)
        zq = zq.reshape(B, T, self.h, self.dh).transpose(1, 2)
        zk = zk.reshape(B, T, self.h, self.dh).transpose(1, 2)
        v = v.reshape(B, T, self.h, self.dh).transpose(1, 2)
        # score(i,j) = tau * mean_d cos(θq_i,d - θk_j,d) = tau * Re[zq conj(zk)]/dh
        scores = self.tau * (zq @ zk.conj().transpose(-1, -2)).real / self.dh
        mask = torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype).triu(1)
        p = torch.softmax(scores + mask, dim=-1)
        o = (p @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(o)
