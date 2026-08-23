"""Standalone wave-memory LM: interference accumulator in C^{DxD}.

Memory per block: M in C^{D x D}. Fixed unit-phasor key codebook (frozen
buffer, v1). Structural marker-driven interface (identical to
delta_forget): BOS/SEP/FORGET/QUERY => next non-marker is a KEY; after a
KEY => next non-marker is a VALUE. Writes at VALUE tokens with the pending
KEY's phasor; erases at the KEY following FORGET_ID (alpha=1 fixed, value
re-read from M); reads at the KEY following QUERY_ID; the readout persists
until ANSWER. No learned gates, no attention, no RoPE.
"""

import math

import torch
import torch.nn as nn

MARKER_MIN = 9
FORGET_ID = 6
QUERY_ID = 4


def make_codebook(vocab_size: int, d_model: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(vocab_size, d_model, generator=g) * 2.0 * math.pi
    return torch.complex(torch.cos(theta), torch.sin(theta))


class WaveMemState:
    __slots__ = ('M', 'r', 'prev', 'pending_id', 'expect_value')

    def __init__(self, M, r, prev, pending_id, expect_value):
        self.M = M
        self.r = r
        self.prev = prev
        self.pending_id = pending_id
        self.expect_value = expect_value


class WaveMemBlock(nn.Module):
    def __init__(self, d_model: int, read_proj: str = 'complex'):
        super().__init__()
        self.d_model = d_model
        self.read_proj = read_proj
        self.v_proj = nn.Linear(d_model, d_model)
        out_dim = 2 * d_model if read_proj == 'complex' else d_model
        self.out_proj = nn.Linear(out_dim, d_model)

    def forward(self, x, M, w_cur, w_pending, write, erase, read):
        # v real (O03, decision pre-registrada): canal imaginario del valor = 0
        v_c = torch.complex(self.v_proj(x), torch.zeros_like(self.v_proj(x)))
        B = M.size(0)
        if write.any() or erase.any():
            delta = torch.zeros_like(M)
            if write.any():
                delta[write] = w_pending[write].unsqueeze(-1) * v_c[write].unsqueeze(1)
            if erase.any():
                wh = torch.einsum('bd,bdj->bj', torch.conj(w_cur[erase]), M[erase])
                wh = wh / self.d_model
                if self.read_proj == 're':
                    wh = torch.complex(wh.real, torch.zeros_like(wh.real))
                delta[erase] = -w_cur[erase].unsqueeze(-1) * wh.unsqueeze(1)
            M = M + delta
        r = None
        if read.any():
            rr = torch.einsum('bd,bdj->bj', torch.conj(w_cur[read]), M[read])
            rr = rr / self.d_model
            if self.read_proj == 're':
                rr = torch.complex(rr.real, torch.zeros_like(rr.real))
            r = (read, rr)
        return M, r, v_c


class WaveMemLM(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int,
                 n_layers: int = 2, read_proj: str = 'complex',
                 codebook_seed: int = 0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.read_proj = read_proj
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('codebook', make_codebook(vocab_size, d_model, codebook_seed),
                             persistent=False)
        self.blocks = nn.ModuleList([WaveMemBlock(d_model, read_proj)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = self.embedding.weight.device
        M = [torch.zeros(batch_size, self.d_model, self.d_model,
                         dtype=torch.complex64, device=device)
             for _ in range(self.n_layers)]
        r = [torch.zeros(batch_size, self.d_model, dtype=torch.complex64,
                         device=device) for _ in range(self.n_layers)]
        prev = torch.zeros(batch_size, dtype=torch.long, device=device)
        pending_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        expect_value = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return WaveMemState(M, r, prev, pending_id, expect_value)

    def decode_step(self, tok, state):
        B = tok.size(0)
        x = self.embedding(tok)
        tok_is_marker = tok < MARKER_MIN
        cur_is_key = ~tok_is_marker
        prev_forget = state.prev == FORGET_ID
        prev_query = state.prev == QUERY_ID
        write = state.expect_value & cur_is_key
        erase = prev_forget & cur_is_key
        read = prev_query & cur_is_key
        w_cur = self.codebook[tok]
        w_pending = self.codebook[state.pending_id]
        M = state.M
        r_persist = state.r
        for b, block in enumerate(self.blocks):
            M_b = M[b]
            M_b, r_new, _ = block(x, M_b, w_cur, w_pending, write, erase, read)
            M[b] = M_b
            if r_new is not None:
                ridx, rr = r_new
                rp = r_persist[b].clone()
                rp[ridx] = rr
                r_persist[b] = rp
        h = x
        for b, block in enumerate(self.blocks):
            rb = r_persist[b]
            if self.read_proj == 'complex':
                feat = torch.cat([rb.real, rb.imag], dim=-1)
            else:
                feat = rb.real
            h = h + block.out_proj(feat)
        logits = self.head(self.norm(h))
        state.prev = tok
        state.expect_value = ~tok_is_marker & ~state.expect_value
        state.pending_id = torch.where(cur_is_key & ~write, tok, state.pending_id)
        return logits, state

    def forward(self, input_ids):
        B, T = input_ids.shape
        if T == 0:
            return torch.empty(B, 0, self.vocab_size, device=input_ids.device,
                               dtype=self.embedding.weight.dtype)
        state = self.init_state(B, input_ids.device)
        outs = []
        for t in range(T):
            logits_t, state = self.decode_step(input_ids[:, t], state)
            outs.append(logits_t)
        return torch.stack(outs, dim=1)

    def prefill(self, input_ids, state=None):
        B, T = input_ids.shape
        if state is None:
            state = self.init_state(B, input_ids.device)
        outs = []
        for t in range(T):
            logits_t, state = self.decode_step(input_ids[:, t], state)
            outs.append(logits_t)
        if outs:
            logits = torch.stack(outs, dim=1)
        else:
            logits = torch.empty(B, 0, self.vocab_size, device=input_ids.device,
                                 dtype=self.embedding.weight.dtype)
        return logits, state