"""Diagnostico complementario de la autopsia delta (auditoria O03-N1):
1. Norma ||S||_F del wave_mem entrenado (debe ser acotada, ~n*|v|) vs delta.
2. delta con write correctivo a paso reducido (beta=0.1) a n_pairs=8:
   si ||S||_F deja de explotar, el punto de fallo es el WRITE (paso-1 LMS),
   no la receta (lr/clip ya barridos).
"""

import json
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from tests.common_smoke import (ForgetRetrieveDataset, collate_fn, evaluate,
                                ANSWER_ID, PAD_ID, VOCAB_FORGET_SIZE)
from tests.wave_mem_n1 import build_wave, build_delta


def S_norm_of(model, xb, dev):
    st = model.init_state(xb.size(0), dev)
    with torch.no_grad():
        for t in range(xb.size(1)):
            _, st = model.decode_step(xb[:, t], st)
    return st


def patch_delta_beta(model, beta):
    """Sustituye el write correctivo por la version a paso beta (diagnostico)."""
    for blk in model.blocks:
        orig = blk.forward

        def fwd(x, x_pending, S, write, erase, read, _blk=blk, _beta=beta):
            k_cur = _blk.k_proj(x)
            if write.any() or erase.any():
                delta = torch.zeros_like(S)
                if write.any():
                    k_pend = _blk.k_proj(x_pending[write])
                    v = _blk.v_proj(x[write])
                    kS = torch.einsum('bd,bdj->bj', k_pend, S[write])
                    delta[write] = (_beta * (v - kS)).unsqueeze(1) * k_pend.unsqueeze(2)
                if erase.any():
                    k_e = _blk.k_proj(x[erase])
                    kS = torch.einsum('bd,bdj->bj', k_e, S[erase])
                    n2 = (k_e * k_e).sum(dim=1).clamp(min=1e-8).unsqueeze(1)
                    delta[erase] = -(kS / n2).unsqueeze(1) * k_e.unsqueeze(2)
                S = S + delta
            r = None
            if read.any():
                q = _blk.q_proj(x[read])
                r = (read, torch.einsum('bd,bdj->bj', q, S[read]))
            return S, r, k_cur

        blk.forward = fwd
    return model


def main():
    dev = torch.device('cpu')
    col = partial(collate_fn, answer_marker_id=ANSWER_ID, mark_after_marker=False,
                  prefix_answer=False)
    res = {}

    # 1) S norm del wave entrenado (ckpt N1) vs delta random (misma batch)
    xb, _, _ = next(iter(DataLoader(
        ForgetRetrieveDataset(32, seed=5, n_pairs_range=(16, 24), n_forget_range=(1, 2)),
        batch_size=32, collate_fn=col)))
    for arm, build in [('wave_complex', build_wave('complex')),
                       ('wave_re', build_wave('re'))]:
        m = build(VOCAB_FORGET_SIZE, 64, 64, 2)
        ck = torch.load(f'outputs/n1_{arm}/cache/forget_retrieval_seed1_dm64_L2_ep80.pt',
                        map_location=dev)
        m.load_state_dict(ck['model']); m.eval()
        st = S_norm_of(m, xb, dev)
        res[f'{arm}_S_norm'] = [round(float(s.norm().item()), 1) for s in st.M]
    md = build_delta(VOCAB_FORGET_SIZE, 32, 64, 2)
    st = S_norm_of(md, xb, dev)
    res['delta_random_S_norm'] = [round(float(s.norm().item()), 1) for s in st.S]
    print(res)

    # 2) delta beta=0.1 a n_pairs=8: entrena? S acotada?
    m = patch_delta_beta(build_delta(VOCAB_FORGET_SIZE, 32, 64, 2), 0.1)
    tr = ForgetRetrieveDataset(600, seed=1, n_pairs_range=(8, 8), n_forget_range=(1, 2))
    va = ForgetRetrieveDataset(150, seed=101, n_pairs_range=(8, 8), n_forget_range=(1, 2))
    tl = DataLoader(tr, batch_size=32, shuffle=True, collate_fn=col)
    vl = DataLoader(va, batch_size=32, shuffle=False, collate_fn=col)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    peak = 0.0
    s_norms = []
    for ep in range(1, 41):
        m.train()
        for x, y, tm in tl:
            yt = torch.where(tm, y, torch.full_like(y, PAD_ID))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB_FORGET_SIZE),
                                   yt.reshape(-1), ignore_index=PAD_ID)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if not math.isfinite(loss.item()):
                res['beta01_nan_epoch'] = ep
                break
        val = evaluate(m, vl, dev, VOCAB_FORGET_SIZE)
        peak = max(peak, val['exact_match'])
        st = S_norm_of(m, xb, dev)
        s_norms.append([round(float(s.norm().item()), 1) for s in st.S])
        if ep in (5, 10, 20, 40):
            print(f'  beta=0.1 ep={ep} valEM={val["exact_match"]:.3f} '
                  f'S_norm={s_norms[-1]}')
    res['beta01_peak_em'] = peak
    res['beta01_S_norm_first'] = s_norms[0]
    res['beta01_S_norm_last'] = s_norms[-1]

    with open('outputs/wave_mem/delta_autopsy_beta.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=str)
    print('Escrito: outputs/wave_mem/delta_autopsy_beta.json')


if __name__ == '__main__':
    main()