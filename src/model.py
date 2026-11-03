"""G2 bake-off LM: pre-LN transformer; attention kernel switch ghrr|vanilla with IDENTICAL param shapes."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    """Same wq/wk/wv/wo shapes for both arms. ghrr: softmax(tau*mean cos(theta_q-theta_k));
    vanilla: softmax(q.k/sqrt(dh)). Fixed random per-(token,slot,head) phase offsets on the GHRR
    arm's q/k angles (research note 4: PEs fixed + vary across QKV/heads; scalar-per-head cheap
    variant — full-D lambda table is a later ablation)."""

    def __init__(self, d_model, n_heads, hv_dim, attn_type, vocab, seed=1337, tau=4.0):
        super().__init__()
        assert attn_type in ("ghrr", "vanilla") and hv_dim % n_heads == 0
        self.type, self.h, self.dh, self.tau = attn_type, n_heads, hv_dim // n_heads, tau
        self.wq = nn.Linear(d_model, hv_dim)
        self.wk = nn.Linear(d_model, hv_dim)
        self.wv = nn.Linear(d_model, hv_dim)
        self.wo = nn.Linear(hv_dim, d_model)
        if attn_type == "ghrr":  # fixed phase biases: [vocab, 3*qkv, heads] radians
            g = torch.Generator().manual_seed(seed)
            pb = torch.empty(vocab, 3, n_heads).uniform_(0, 2 * math.pi, generator=g)
            self.register_buffer("phase_bias", pb, persistent=False)

    def forward(self, x, ids):
        B, T, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        shp = lambda t: t.reshape(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = shp(q), shp(k), shp(v)
        if self.type == "ghrr":
            pb = self.phase_bias[ids]  # [B,T,3,H]; q/k only, v stays real in BOTH arms (single-factor swap)
            zq = torch.exp(1j * (q + pb[:, :, 0].transpose(1, 2).unsqueeze(-1)))
            zk = torch.exp(1j * (k + pb[:, :, 1].transpose(1, 2).unsqueeze(-1)))
            scores = self.tau * (zq @ zk.conj().transpose(-1, -2)).real / self.dh
        else:
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        mask = torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype).triu(1)
        p = torch.softmax(scores + mask, dim=-1)
        return self.wo((p @ v).transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.hv_dim, cfg.attn_type, cfg.vocab, cfg.seed + cfg.i, cfg.tau)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x, ids):
        x = x + self.attn(self.ln1(x), ids)
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, d_model=384, n_layer=6, n_heads=8, hv_dim=4096, vocab=50257, ctx=64,
                 attn_type="ghrr", seed=1337, tau=4.0):
        super().__init__()
        self.ctx = ctx
        cfgs = []
        for i in range(n_layer):
            c = nn.Module(); c.d_model, c.n_heads, c.hv_dim = d_model, n_heads, hv_dim
            c.attn_type, c.vocab, c.seed, c.i, c.tau = attn_type, vocab, seed, i, tau
            cfgs.append(c)
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList(Block(c) for c in cfgs)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x, idx)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                               ignore_index=-1)
        return logits, loss
