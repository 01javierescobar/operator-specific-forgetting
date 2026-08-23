"""Fuga pooled (suma de numeradores / suma de denominadores) del probe de
drift sobre modelos entrenados: metrica robusta a la cola pesada de ||v||^2
del modelo entrenado (p90/p50 ~ 5.6) que infla la media de ratios."""

import math
import sys
import torch

sys.path.insert(0, '.')
from tests.common_smoke import FORGET_ID, QUERY_ID
from tests.wave_mem_n1 import build_wave, make_fr_probe_dataset, v_true_for

dev = torch.device('cpu')


def pooled(m, x, seqs, v_erased, dlt, mode):
    B, T = x.shape
    is_re = m.read_proj == 're'
    pos_forget = torch.tensor([s.index(FORGET_ID) for s in seqs], dtype=torch.long)
    pos_qkey = torch.tensor([s.index(QUERY_ID) + 1 for s in seqs], dtype=torch.long)
    tok_key = torch.tensor([s[int(pos_forget[i]) + 1] for i, s in enumerate(seqs)],
                           dtype=torch.long)
    ph = torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))
    st = m.init_state(B, dev)
    rows = torch.zeros(B, m.d_model, dtype=torch.complex64)
    with torch.no_grad():
        for t in range(T):
            if (pos_forget == t - 1).any():
                idx = (pos_forget == t - 1).nonzero(as_tuple=False).squeeze(-1)
                wd = ph * m.codebook[tok_key[idx]]
                for b in range(m.n_layers):
                    M_b = st.M[b]
                    M_new = M_b.clone()
                    if mode == 'reread':
                        wh = torch.einsum('bd,bdj->bj', torch.conj(wd), M_b[idx]) / m.d_model
                        if is_re:
                            wh = torch.complex(wh.real, torch.zeros_like(wh.real))
                        M_new[idx] = M_b[idx] - wd.unsqueeze(-1) * wh.unsqueeze(1)
                    else:
                        vts = v_true_for(m, v_erased[idx])
                        M_new[idx] = M_b[idx] - wd.unsqueeze(-1) * vts[b].unsqueeze(1)
                    st.M[b] = M_new
                prev = st.prev.clone()
                prev[idx] = tok_key[idx]
                st.prev = prev
            _, st = m.decode_step(x[:, t], st)
            if (pos_qkey == t).any():
                qm = pos_qkey == t
                rr = rows.clone()
                rr[qm] = st.r[0][qm]
                rows = rr
    v0 = v_true_for(m, v_erased)[0]
    if is_re:
        rows = rows.real
        v0 = v0.real
    return float(rows.abs().pow(2).sum().item() / v0.abs().pow(2).sum().item())


c = 19 / 64
print('leyes (c=19/64):')
print(f'  reread_re sin4(d)+c(1-cos2d)/4: 0 / {math.sin(0.3) ** 4 + c * (1 - math.cos(0.6)) / 4:.4f} / '
      f'{math.sin(0.5) ** 4 + c * (1 - math.cos(1.0)) / 4:.4f}')
print(f'  represent_cplx c+2(1-cosd): {c:.4f} / {c + 2 * (1 - math.cos(0.3)):.4f} / {c + 2 * (1 - math.cos(0.5)):.4f}')
print(f'  represent_re c/2+(1-cosd)^2: {c / 2:.4f} / {c / 2 + (1 - math.cos(0.3)) ** 2:.4f} / {c / 2 + (1 - math.cos(0.5)) ** 2:.4f}')
print()
for arm, rp in (('wave_complex', 'complex'), ('wave_re', 're')):
    for seed in (1, 2, 3):
        m = build_wave(rp)(89, 64, 64, 2)
        ck = torch.load(f'outputs/n1_{arm}/cache/forget_retrieval_seed{seed}'
                        f'_dm64_L2_ep80.pt', map_location=dev)
        m.load_state_dict(ck['model'])
        m.eval()
        line = []
        for dlt in (0.0, 0.3, 0.5):
            x, y, seqs, v_erased = make_fr_probe_dataset(
                320, seed=2000 + seed, kind='erased', n_pairs_range=(16, 24))
            f_rr = pooled(m, x.to(dev), seqs, v_erased, dlt, 'reread')
            f_rp = pooled(m, x.to(dev), seqs, v_erased, dlt, 'represent')
            line.append(f'd={dlt} rr={f_rr:.4f} rp={f_rp:.4f}')
        print(f'{arm} s{seed}:', ' | '.join(line), flush=True)