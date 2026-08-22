"""Diagnostico del exceso de fuga del canal re a delta>=0.4 en modelos
entrenados (O04): mide E|c|^2 y E[c^2] del crosstalk de la fila w_j^H M/D
en el modelo entrenado (isotropia del piso)."""

import sys
import torch

sys.path.insert(0, '.')
from tests.wave_mem_n1 import (build_wave, make_fr_probe_dataset,
                               v_true_for)
from prototypes.wave_mem.model import WaveMemLM

dev = torch.device('cpu')
for arm, rp in (('wave_complex', 'complex'), ('wave_re', 're')):
    for seed in (1, 2, 3):
        m = build_wave(rp)(89, 64, 64, 2)
        ck = torch.load(f'outputs/n1_{arm}/cache/forget_retrieval_seed{seed}'
                        f'_dm64_L2_ep80.pt', map_location=dev)
        m.load_state_dict(ck['model'])
        m.eval()
        x, y, seqs, v_erased = make_fr_probe_dataset(320, seed=2000 + seed,
                                                     kind='erased',
                                                     n_pairs_range=(16, 24))
        from tests.common_smoke import FORGET_ID, QUERY_ID
        pos_forget = [s.index(FORGET_ID) + 1 for s in seqs]
        pos_qkey = [s.index(QUERY_ID) + 1 for s in seqs]
        st = m.init_state(320, dev)
        rows = []
        vjs = []
        with torch.no_grad():
            T = x.size(1)
            for t in range(T):
                _, st = m.decode_step(x[:, t], st)
            for i, s in enumerate(seqs):
                # fila de M al final: w^H M/D en la fila de la clave borrada
                w = m.codebook[s[pos_forget[i]]]
                M0 = st.M[0]
                row = torch.einsum('d,de->e', torch.conj(w), M0[i]) / m.d_model
                if rp == 're':
                    row = row.real
                vj = v_true_for(m, torch.tensor([v_erased[i].item()]))[0]
                if rp == 're':
                    vj = vj.real
                rows.append(row)
                vjs.append(vj)
        R = torch.stack(rows)
        V = torch.stack(vjs)
        c = R - V
        Ec2 = (c.abs().pow(2).sum(dim=1) / V.pow(2).sum(dim=1)).mean().item()
        # crosstalk bruto y anisotropia compleja
        cb = (c.conj() * c).sum(dim=1)
        print(f'{arm} s{seed}: E|c|^2/|v|^2={Ec2:.4f}  '
              f'|E[c^2]|/E|c|^2={cb.abs().mean().item() / c.abs().pow(2).mean().item():.4f}  '
              f'vnorms: mean={V.pow(2).sum(1).float().mean().item():.2f} '
              f'p90/p50={V.pow(2).sum(1).float().quantile(0.9).item() / V.pow(2).sum(1).float().median().item():.2f}')