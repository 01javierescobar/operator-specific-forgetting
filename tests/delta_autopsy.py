"""Autopsia delta_forget (auditoría O03-N1, condición bloqueante antes de O04).

Preguntas:
1. ¿delta entrena FR con receta distinta? Sweep lr x grad-clip a n_pairs=8
   (régimen más fácil): lr {1e-3, 5e-4, 2.5e-4} x clip {1.0, None}.
2. ¿Dónde muere el gradiente? Normas de gradiente por componente
   (embedding, k_proj, v_proj, q_proj, out_proj, norm, head) por época.
3. ¿El mecanismo del NaN? Norma ||S||_F por capa por época (crecimiento
   de la memoria = overflow).

Salida: outputs/wave_mem/delta_autopsy.json
"""

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tests.common_smoke import (ForgetRetrieveDataset, collate_fn, evaluate,
                                ANSWER_ID, PAD_ID, VOCAB_FORGET_SIZE)
from tests.wave_mem_n1 import build_delta

N_EPOCHS = 40
N_PAIRS = (8, 8)
N_FORGET = (1, 2)
TRAIN_N = 600
VALID_N = 150
BATCH = 32


def component_grad_norms(model):
    raw = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            raw[name] = None
            continue
        raw[name] = round(float(p.grad.norm().item()), 6)
    comps = ['embedding', 'head', 'norm']
    for i in range(len(model.blocks)):
        for c in ('k_proj', 'v_proj', 'q_proj', 'out_proj'):
            comps.append(f'blocks.{i}.{c}')
    agg = {}
    for comp in comps:
        vals = [v for k, v in raw.items() if k.startswith(comp) and v is not None]
        agg[comp] = round(sum(v ** 2 for v in vals) ** 0.5, 6) if vals else None
    return agg


def run_autopsy(lr, clip, seed=1, epochs=N_EPOCHS):
    dev = torch.device('cpu')
    m = build_delta(VOCAB_FORGET_SIZE, 32, 64, 2)
    col = partial(collate_fn, answer_marker_id=ANSWER_ID, mark_after_marker=False,
                  prefix_answer=False)
    tr = ForgetRetrieveDataset(TRAIN_N, seed=seed, n_pairs_range=N_PAIRS,
                               n_forget_range=N_FORGET)
    va = ForgetRetrieveDataset(VALID_N, seed=seed + 100, n_pairs_range=N_PAIRS,
                               n_forget_range=N_FORGET)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, collate_fn=col)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False, collate_fn=col)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    peak_em = 0.0
    nan_epoch = None
    S_norm = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        m.train()
        ep_loss = 0.0
        ep_tok = 0
        grad_norms_ep = {}
        for x, y, tm in tl:
            yt = torch.where(tm, y, torch.full_like(y, PAD_ID))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB_FORGET_SIZE),
                                   yt.reshape(-1), ignore_index=PAD_ID)
            loss.backward()
            grad_norms_ep = component_grad_norms(m)
            if clip is not None:
                nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
            ep_loss += loss.item() * (yt != PAD_ID).sum().item()
            ep_tok += (yt != PAD_ID).sum().item()
            if not math.isfinite(loss.item()):
                nan_epoch = ep
                break
        if nan_epoch:
            break
        val = evaluate(m, vl, dev, VOCAB_FORGET_SIZE)
        peak_em = max(peak_em, val['exact_match'])
        with torch.no_grad():
            m.init_state(1, dev)
            xb, yb, tmb = next(iter(vl))
            st = m.init_state(xb.size(0), dev)
            for t in range(xb.size(1)):
                _, st = m.decode_step(xb[:, t], st)
            S_norm.append([round(float(s.norm().item()), 3) for s in st.S])
        hist.append({
            'epoch': ep, 'train_loss': ep_loss / max(ep_tok, 1),
            'valid_em': val['exact_match'], 'grad': grad_norms_ep})
    return {
        'lr': lr, 'clip': clip, 'nan_epoch': nan_epoch,
        'final_em': val['exact_match'] if not nan_epoch else 0.0,
        'peak_em': peak_em, 'loss_trace': [h['train_loss'] for h in hist],
        'em_trace': [h['valid_em'] for h in hist],
        'grad_ep1': hist[0]['grad'] if hist else None,
        'grad_mid': hist[len(hist) // 2]['grad'] if len(hist) > 1 else None,
        'grad_last': hist[-1]['grad'] if hist else None,
        'S_norm_first': S_norm[0] if S_norm else None,
        'S_norm_last': S_norm[-1] if S_norm else None,
        'seconds': round(time.time() - t0, 1),
    }


def main():
    combos = [(1e-3, 1.0), (1e-3, None), (5e-4, 1.0), (5e-4, None),
              (2.5e-4, 1.0), (2.5e-4, None)]
    out = {'variant': 'delta_autopsy', 'n_pairs': 8, 'epochs': N_EPOCHS,
           'seed': 1, 'runs': []}
    for lr, clip in combos:
        print(f'=== autopsy lr={lr} clip={clip} ===', flush=True)
        r = run_autopsy(lr, clip)
        print(f'  nan_ep={r["nan_epoch"]} final_em={r["final_em"]:.3f} '
              f'peak_em={r["peak_em"]:.3f} t={r["seconds"]}s', flush=True)
        out['runs'].append(r)
    os.makedirs('outputs/wave_mem', exist_ok=True)
    with open('outputs/wave_mem/delta_autopsy.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print('Escrito: outputs/wave_mem/delta_autopsy.json')


if __name__ == '__main__':
    main()