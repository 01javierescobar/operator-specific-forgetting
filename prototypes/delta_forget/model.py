"""Standalone delta-rule LM: sequential delta rule in R^{DxD}.

Comparator arm for wave_mem with an IDENTICAL structural marker interface:
BOS/SEP/FORGET/QUERY => next non-marker is a KEY; after a KEY => next
non-marker is a VALUE. Writes at VALUE tokens binding the pending KEY's
projection (K(emb_key)) to the value projection (V(emb_value)); erases at
the KEY following FORGET_ID with beta=1 forced (projection (I - uu^T),
u = k/||k||); reads at the KEY following QUERY_ID with the learned q; the
readout persists until ANSWER. No conv1d, no RoPE, no learned erase
amplitude, no learned write gate (beta=1 fixed): the operator is the only
difference vs wave_mem (key channel and read projection).
"""

import torch
import torch.nn as nn

MARKER_MIN = 9
FORGET_ID = 6
QUERY_ID = 4


class DeltaState:
    __slots__ = ('S', 'r', 'prev', 'pending_id', 'expect_value')

    def __init__(self, S, r, prev, pending_id, expect_value):
        self.S = S
        self.r = r
        self.prev = prev
        self.pending_id = pending_id
        self.expect_value = expect_value


class DeltaBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, x_pending, S, write, erase, read):
        k_cur = self.k_proj(x)
        if write.any() or erase.any():
            delta = torch.zeros_like(S)
            if write.any():
                k_pend = self.k_proj(x_pending[write])
                v = self.v_proj(x[write])
                kS = torch.einsum('bd,bdj->bj', k_pend, S[write])
                delta[write] = (v - kS).unsqueeze(1) * k_pend.unsqueeze(2)
            if erase.any():
                k_e = self.k_proj(x[erase])
                kS = torch.einsum('bd,bdj->bj', k_e, S[erase])
                n2 = (k_e * k_e).sum(dim=1).clamp(min=1e-8).unsqueeze(1)
                delta[erase] = -(kS / n2).unsqueeze(1) * k_e.unsqueeze(2)
            S = S + delta
        r = None
        if read.any():
            q = self.q_proj(x[read])
            r = (read, torch.einsum('bd,bdj->bj', q, S[read]))
        return S, r, k_cur


class DeltaForgetLM(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int,
                 n_layers: int = 2):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DeltaBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = self.embedding.weight.device
        S = [torch.zeros(batch_size, self.d_model, self.d_model, device=device)
             for _ in range(self.n_layers)]
        r = [torch.zeros(batch_size, self.d_model, device=device)
             for _ in range(self.n_layers)]
        prev = torch.zeros(batch_size, dtype=torch.long, device=device)
        pending_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        expect_value = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return DeltaState(S, r, prev, pending_id, expect_value)

    def decode_step(self, tok, state):
        x = self.embedding(tok)
        tok_is_marker = tok < MARKER_MIN
        cur_is_key = ~tok_is_marker
        prev_forget = state.prev == FORGET_ID
        prev_query = state.prev == QUERY_ID
        write = state.expect_value & cur_is_key
        erase = prev_forget & cur_is_key
        read = prev_query & cur_is_key
        x_pending = self.embedding(state.pending_id)
        S = state.S
        r_persist = state.r
        for b, block in enumerate(self.blocks):
            S_b = S[b]
            S_b, r_new, _ = block(x, x_pending, S_b, write, erase, read)
            S[b] = S_b
            if r_new is not None:
                ridx, rr = r_new
                rp = r_persist[b].clone()
                rp[ridx] = rr
                r_persist[b] = rp
        h = x
        for b, block in enumerate(self.blocks):
            h = h + block.out_proj(r_persist[b])
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