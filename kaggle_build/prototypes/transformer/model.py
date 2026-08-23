"""GPT-2-style decoder-only transformer (baseline, paper-backed).

Transcripcion fiel del decoder-only LM de Radford et al. 2019 ("Language
Models are Unsupervised Multitask Learners", OpenAI GPT-2), el mismo canon
que usan los benchmarks de associative recall de la literatura:

- Arora et al. 2023, "Simple linear attention language models balance the
  recall-throughput tradeoff", arXiv:2302.11181 (MQAR benchmark).
- Okpekpe & Orvieto 2025, "Revisiting Associative Recall in Modern
  Recurrent Models", arXiv:2508.19029 (reproducen MQAR con un transformer
  GPT-2-style 2L d=64).

Propiedades GPT-2 mantenidas explicitas:
- pre-norm LayerNorm (eps=1e-5) antes de atencion y antes del MLP
- embeddings posicionales absolutos aprendidos
- GELU en el MLP (nn.GELU erf; GPT-2 usa tanh-approx, diferencia numerica
  despreciable para este benchmark)
- sin bias en las proyecciones de atencion (c_attn, c_proj)
- proyeccion residual escalada 1/sqrt(2*n_layers) (init GPT-2)
- LayerNorm final antes del head
- head sin atar al embedding: GPT-2 ata lm_head<->wte, pero a d=64 con
  VOCAB=89 el tying degrada copy_reverse (EM 1.0 -> 0.84; A/B en
  _trash/exp_gpt2_copy_ab.py), asi que el default queda untied para
  preservar el comportamiento del baseline historico

Regimen de datos (documentado, no un bug): el harness de este repo entrena
con 400 samples por task y, en las tasks MQAR, supervisa UN solo token por
sample (el token post-ANSWER; mark_after_marker=False en tests/common_smoke.py).
La literatura entrena associative recall con 10^4-10^6 tokens y CE sobre
toda la secuencia (Okpekpe & Orvieto 2025: seq 256, batch 128-512,
50 epochs). Por eso este baseline NO aprende mqar_1hop en el regimen del
repo (EM ~ chance) pero SI copy_reverse (supervision de span completo): es
un efecto de regimen de datos, no de implementacion. El mismo codigo llega
a EM=1.0 en copy_reverse y reproduce los numeros de la literatura cuando
recibe el regimen de datos de los papers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Scaled dot-product self-attention con mascara causal (GPT-2)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    """Transformer block GPT-2: pre-LN + attn, pre-LN + MLP, residual."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int,
                 dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """Vanilla decoder-only Transformer, GPT-2-style (baseline).

    Standalone baseline para el benchmark local. Cualquier prototipo debe
    igualar o superar a este modelo en cada task con los mismos datos.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, ff_mult: int = 2, dropout: float = 0.0,
                 max_len: int = 256, tie_weights: bool = False,
                 eps: float = 1e-5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, ff_mult, dropout, eps)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # GPT-2 ata lm_head<->wte, pero a d=64 con VOCAB=89 el tying degrada
        # copy_reverse (EM 1.0 -> 0.84, A/B en _trash/exp_gpt2_copy_ab.py).
        # Se mantiene sin atar para preservar el comportamiento del baseline
        # historico (mismo resultado en las 5 tasks).
        if tie_weights:
            self.head.weight = self.tok.weight
        self.apply(self._init_weights)
        # Scale residual projections como GPT-2.
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.drop(self.tok(x) + self.pos(pos))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
